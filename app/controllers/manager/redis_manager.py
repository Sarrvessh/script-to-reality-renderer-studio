import inspect
import json
from typing import Dict

import redis
from loguru import logger
from pydantic import ValidationError

from app.controllers.manager.base_manager import TaskManager, TaskQueueFullError
from app.models import const
from app.models.schema import AudioRequest, SubtitleRequest, VideoParams
from app.services import state as sm
from app.services import task as tm

FUNC_MAP = {
    "start": tm.start,
    # 'start_test': tm.start_test
}

_ADMIT_QUEUED_TASK_SCRIPT = """
if redis.call("LLEN", KEYS[1]) >= tonumber(ARGV[1]) then
    return 0
end
redis.call("RPUSH", KEYS[1], ARGV[2])
return 1
"""


class RedisTaskManager(TaskManager):
    def __init__(
        self,
        max_concurrent_tasks: int,
        redis_url: str,
        max_queued_tasks: int = 100,
    ):
        self.redis_client = redis.Redis.from_url(redis_url)
        super().__init__(max_concurrent_tasks, max_queued_tasks=max_queued_tasks)

    def create_queue(self):
        return "task_queue"

    def resume_queued_tasks(self):
        """Dispatch persisted queue entries when a new API process starts.

        Queued Redis items survive a restart, but no worker calls check_queue
        until an active task finishes. Fill each available local worker slot so
        an idle process does not leave old requests stuck in processing.
        """
        for _ in range(max(0, self.max_concurrent_tasks - self.current_tasks)):
            self.check_queue()

    def enqueue(self, task: Dict):
        self.redis_client.rpush(self.queue, self._serialize_task(task))

    def enqueue_new_task(self, task: Dict):
        # Manager locks are process-local. Keep capacity validation and admission
        # in one Redis operation so other API workers cannot claim the same slot.
        admitted = self.redis_client.eval(
            _ADMIT_QUEUED_TASK_SCRIPT,
            1,
            self.queue,
            self.max_queued_tasks,
            self._serialize_task(task),
        )
        if not admitted:
            raise TaskQueueFullError("task queue is full, please try again later")

    @staticmethod
    def _serialize_task(task: Dict) -> str:
        task_with_serializable_params = task.copy()
        # task.copy() copies only the outer dict; rewriting nested kwargs in place would also replace the caller's
        # VideoParams with a dict. Later logs or retries may still read the original task,
        # so kwargs is copied separately and serialization has no unexpected side effect.
        task_kwargs = task.get("kwargs", {})
        task_with_serializable_params["kwargs"] = task_kwargs.copy()

        if "params" in task_kwargs and isinstance(
            task_kwargs["params"], (VideoParams, AudioRequest, SubtitleRequest)
        ):
            task_with_serializable_params["kwargs"]["params"] = task_kwargs[
                "params"
            ].model_dump(warnings=False)

        # Convert the function object to its name
        task_with_serializable_params["func"] = task["func"].__name__
        return json.dumps(task_with_serializable_params)

    def dequeue(self):
        # Loop instead of popping once: a task may have passed the checks that existed when it was queued, but checks and
        # FUNC_MAP members change across deploys (for example VideoParams gains a ge=1 constraint, or an entry
        # function is removed), so the queue can still hold entries written with the old schema or that can no longer be parsed.
        # lpop is destructive, and once an item is popped it cannot be put back; that task has permanently left the queue
        # and must not be treated as if it were still there. Raising from here (check_queue calls
        # this method while holding the lock, and the exception follows task_done → run_task's finally and kills the worker; after that
        # no task is running, nobody calls check_queue again, and later tasks stay forever
        # in processing) is worse than dropping it here and trying the next one, which keeps the contract "return one usable task
        # or the queue is really empty".
        while True:
            task_json = self.redis_client.lpop(self.queue)
            # Only an lpop that pops nothing means the queue is empty. An empty string (or empty bytes) is also
            # an unusable entry, and a usable task may still be behind it, so take the discard
            # path below instead of treating it as "end of queue" and returning.
            if task_json is None:
                return None

            task_info = None
            try:
                task_info = json.loads(task_json)
                # Convert the function name back to a function object. If the name is missing or no longer in FUNC_MAP, do not
                # index directly, or a KeyError would skip the params discard path below.
                task_info["func"] = FUNC_MAP[task_info["func"]]
                task_kwargs = task_info["kwargs"]
                if not isinstance(task_kwargs, dict):
                    raise ValueError("queued task has no keyword argument mapping")
                # When args is missing entirely, keep check_queue's default; null or any other non-
                # array form makes check_queue raise TypeError while expanding `*args`, and that path
                # requeues the entry and lets the exception escape the worker, so it must be stopped here.
                if not isinstance(task_info.get("args", []), list):
                    raise ValueError("queued task positional arguments are not a list")
                # A persisted request can outlive a callable's signature. Fail
                # it before dispatch: Python argument-binding errors happen
                # before start() enters its pipeline failure handler.
                inspect.signature(task_info["func"]).bind(
                    *task_info.get("args", []), **task_kwargs
                )
            except (TypeError, ValueError, KeyError) as e:
                logger.error(f"dropping unusable queued task: {e}")
                # Same as the params validation failure path below: if a usable task_id can be read, mark
                # this task that has permanently left the queue as failed, or the API/WebUI will keep showing
                # it as processing. If the payload itself cannot be parsed, or task_id is not a string
                # (for example a JSON array), there is no record to write back and it can only be dropped — handing a non-string
                # to patch_task makes redis raise DataError and breaks the discard loop.
                stale_kwargs = (
                    task_info.get("kwargs") if isinstance(task_info, dict) else None
                )
                task_id = (
                    stale_kwargs.get("task_id")
                    if isinstance(stale_kwargs, dict)
                    else None
                )
                if isinstance(task_id, str) and task_id:
                    sm.state.patch_task(
                        task_id,
                        state=const.TASK_STATE_FAILED,
                        failed_stage="dequeue",
                        error=f"discarded stale queued task: {e}",
                    )
                continue

            if "params" in task_kwargs and isinstance(task_kwargs["params"], dict):
                try:
                    params_model = VideoParams
                    # /audio and /subtitle use short-form request models without
                    # video_subject. Preserve those schemas at the queue boundary;
                    # full VideoParams can also stop at either of these stages.
                    if "video_subject" not in task_kwargs["params"]:
                        if task_kwargs.get("stop_at") == "audio":
                            params_model = AudioRequest
                        elif task_kwargs.get("stop_at") == "subtitle":
                            params_model = SubtitleRequest
                    task_kwargs["params"] = params_model(**task_kwargs["params"])
                except ValidationError as e:
                    logger.error(
                        "dropping queued task with params that fail current "
                        f"request model validation (queued under an older, more "
                        f"permissive schema, or corrupted): {e}"
                    )
                    # The task state record is created before enqueue and defaults to processing; if this queue item is only
                    # dropped and the state record is left alone, the API/WebUI will keep showing the task as
                    # running and it will never become failed. Use patch_task instead of update_task,
                    # so a task the user already deleted is not created again.
                    # If task_id is not a string, same as above: there is no record to write back, so skip the state update.
                    task_id = task_kwargs.get("task_id")
                    if isinstance(task_id, str) and task_id:
                        sm.state.patch_task(
                            task_id,
                            state=const.TASK_STATE_FAILED,
                            failed_stage="dequeue",
                            error=f"discarded stale queued task: {e}",
                        )
                    continue

            return task_info

    def is_queue_empty(self):
        return self.redis_client.llen(self.queue) == 0

    def queue_size(self):
        return self.redis_client.llen(self.queue)
