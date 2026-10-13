import math
import threading
from typing import Any, Callable, Dict

from loguru import logger


class TaskQueueFullError(ValueError):
    pass


def _coerce_task_limit(value: Any, name: str) -> int:
    """
    Parse a concurrency or queue limit from config into an integer.

    A TOML number may be written as `max_concurrent_tasks = 5` or as `"5"`. A string
    would reach the comparison in `add_task` and become a TypeError that does not name the config key, and
    a bad `max_queued_tasks` only shows up after concurrency slots are full. `app/services/webui_task.py`
    already normalizes the same `max_queued_tasks` key, and this lifts that into the single constructor entry.

    Values that are not integers always name the config key: `0.5` would be silently truncated by `int()` to 0 (tasks only queue,
    and no worker runs them), and TOML `inf` / `-inf` / `nan` are not valid limits either.
    0 and negatives are kept as-is: 0 means do not run tasks for now, and a negative sends the request straight into the queue branch.
    """
    if isinstance(value, bool):
        # bool is a subclass of int, but `true` is obviously not the limit the user wanted.
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if isinstance(value, float):
        # A non-finite value is not a limit, and a fraction would truncate into the meaning "0: queue only, do not run".
        if not math.isfinite(value):
            raise ValueError(f"{name} must be a finite integer, got {value!r}")
        if not value.is_integer():
            raise ValueError(f"{name} must be an integer, got {value!r}")
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be an integer, got {value!r}") from None


class TaskManager:
    def __init__(self, max_concurrent_tasks: int, max_queued_tasks: int = 100):
        self.max_concurrent_tasks = _coerce_task_limit(
            max_concurrent_tasks, "max_concurrent_tasks"
        )
        self.max_queued_tasks = _coerce_task_limit(max_queued_tasks, "max_queued_tasks")
        self.current_tasks = 0
        self.lock = threading.Lock()
        self.queue = self.create_queue()

    def create_queue(self):
        raise NotImplementedError()

    def add_task(self, func: Callable, *args: Any, **kwargs: Any):
        with self.lock:
            if self.current_tasks < self.max_concurrent_tasks:
                logger.info(
                    f"add task: {func.__name__}, current_tasks: {self.current_tasks}"
                )
                # Reserve a concurrency slot before the thread starts. The old code incremented inside the thread, so back-to-back requests
                # could all see current_tasks=0 before a child thread took the lock and exceed the concurrency
                # limit. If startup fails, the slot is rolled back so later requests can still be scheduled.
                self.current_tasks += 1
                try:
                    self.execute_task(func, *args, **kwargs)
                except Exception:
                    self.current_tasks -= 1
                    raise
            else:
                queue_size = self.queue_size()
                # Queue only after concurrency is full. The queue must have a cap, or an anonymous endpoint can keep
                # stacking task objects and request parameters until memory or third-party API cost runs away.
                if queue_size >= self.max_queued_tasks:
                    logger.warning(
                        f"reject task: {func.__name__}, queue_size: {queue_size}, "
                        f"max_queued_tasks: {self.max_queued_tasks}"
                    )
                    raise TaskQueueFullError("task queue is full, please try again later")

                logger.info(
                    f"enqueue task: {func.__name__}, current_tasks: {self.current_tasks}, "
                    f"queue_size: {queue_size}"
                )
                self.enqueue_new_task({"func": func, "args": args, "kwargs": kwargs})

    def execute_task(self, func: Callable, *args: Any, **kwargs: Any):
        thread = threading.Thread(
            target=self.run_task, args=(func, *args), kwargs=kwargs
        )
        thread.start()

    def run_task(self, func: Callable, *args: Any, **kwargs: Any):
        try:
            func(*args, **kwargs)  # call the function here, passing *args and **kwargs.
        finally:
            self.task_done()

    def check_queue(self):
        with self.lock:
            self._check_queue_locked()

    def _check_queue_locked(self):
        """Dispatch one queued task while the caller holds ``self.lock``."""
        if (
            self.current_tasks < self.max_concurrent_tasks
            and not self.is_queue_empty()
        ):
            task_info = self.dequeue()
            if task_info is None:
                # dequeue() may skip and discard queue entries that no longer
                # pass current validation (see RedisTaskManager.dequeue) and
                # return None once nothing usable is left, even though
                # is_queue_empty() was False a moment earlier.
                return
            func = task_info["func"]
            args = task_info.get("args", ())
            kwargs = task_info.get("kwargs", {})
            # Reserve the freed slot before another add_task can claim it.
            self.current_tasks += 1
            try:
                self.execute_task(func, *args, **kwargs)
            except Exception:
                self.current_tasks -= 1
                self.enqueue(task_info)
                raise

    def task_done(self):
        with self.lock:
            self.current_tasks -= 1
            self._check_queue_locked()

    def enqueue(self, task: Dict):
        raise NotImplementedError()

    def enqueue_new_task(self, task: Dict):
        """Admit new work; shared queues may need their own atomic limit check."""
        self.enqueue(task)

    def dequeue(self):
        raise NotImplementedError()

    def is_queue_empty(self):
        raise NotImplementedError()

    def queue_size(self):
        raise NotImplementedError()
