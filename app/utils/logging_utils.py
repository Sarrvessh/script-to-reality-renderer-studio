import functools
import os
import threading

from loguru import logger


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
LOG_RECORD_FORMAT = (
    "<green>{time:%Y-%m-%d %H:%M:%S}</> | "
    "<level>{level}</> | "
    '"{file.path}:{line}":<blue> {function}</> '
    "- <level>{message}</>\n"
)
# Loguru's default terminal handler id at startup is 0. On a WebUI reload only this
# base terminal output may be replaced. logger.remove() must not clear every handler, or the temporary sink a running task
# uses to collect WebUI logs is removed too.
_terminal_handler_id: int | None = 0
_terminal_handler_lock = threading.RLock()
# The WebUI filters logs by the task worker thread's id. Helper threads the task starts for parallel downloads, clip encoding, and heartbeats
# have their own thread ids, and their log lines are dropped entirely, so the longest stages
# show nothing in the WebUI. Record a map of "helper thread -> the thread that started it"
# so the log filter can attribute helper-thread records back to their task.
_log_scope_roots: dict[int, int] = {}
_log_scope_lock = threading.Lock()


def log_scope_thread_id(thread_id: int | None = None) -> int:
    """Return the root thread id of this thread's log scope; an unbound thread is its own root."""
    if thread_id is None:
        thread_id = threading.get_ident()
    with _log_scope_lock:
        return _log_scope_roots.get(thread_id, thread_id)


def bind_log_scope(func):
    """
    Wrap a function that will run on another thread so its logs belong to the thread that called this function.

    Call it on the thread that submits the work: the scope is fixed at wrap time, and a thread started by a helper
    still belongs to the original task thread. Unbind when execution ends, because thread ids are reused by the OS
    and a leftover mapping would count a later unrelated thread's logs as part of a task that already finished.
    """
    root_thread_id = log_scope_thread_id()

    @functools.wraps(func)
    def run_in_log_scope(*args, **kwargs):
        thread_id = threading.get_ident()
        if thread_id == root_thread_id:
            return func(*args, **kwargs)

        with _log_scope_lock:
            previous_root = _log_scope_roots.get(thread_id)
            _log_scope_roots[thread_id] = root_thread_id
        try:
            return func(*args, **kwargs)
        finally:
            with _log_scope_lock:
                # The pool reuses threads: restore the owner from before entry instead of always deleting, so
                # a nested wrap does not clear a binding the outer wrap is still using.
                if previous_root is None:
                    _log_scope_roots.pop(thread_id, None)
                else:
                    _log_scope_roots[thread_id] = previous_root

    return run_in_log_scope


def _project_relative_path(file_path):
    """
    Shorten an absolute path to a project-relative path that starts with ``./`` and always uses forward slashes.

    On Windows the project may be started from a mapped network drive or a ``subst`` drive. The call stack then still shows
    ``X:\\Script-to-Reality Renderer Studio\\...``, while ``PROJECT_ROOT`` after ``realpath``
    lands on ``C:\\...``, and ``os.path.relpath`` raises ``ValueError``.
    If the formatter raises, loguru catches it and drops the whole record, so the terminal and the WebUI log panel
    both go empty. Fall back to the original path. The same applies to files outside the project: prefixing
    ``./`` onto a ``..`` relative path only makes the result harder to read.
    """
    try:
        relative_path = os.path.relpath(file_path, PROJECT_ROOT)
    except ValueError:
        return file_path
    if relative_path == os.pardir or relative_path.startswith(os.pardir + os.sep):
        return file_path
    # Windows relpath returns backslash-separated paths, and joining them directly yields mixed separators such as ``./app\\utils``,
    # which does not match logs on other platforms.
    return f"./{relative_path.replace(os.sep, '/')}"


def format_log_record(record):
    """
    Format terminal and WebUI logs the same way.

    Loguru hands the same record to several sinks. The first sink may already have turned an absolute path
    into a project-relative path, so accept both absolute paths and paths already formatted with a ``./`` prefix.
    The WebUI sink turns color off, but time, level, call site, and message stay the same as the terminal.
    """
    file_path = record["file"].path
    if os.path.isabs(file_path):
        record["file"].path = _project_relative_path(file_path)

    # Log messages sometimes contain absolute paths of task files. Shortening them all to project-relative paths
    # keeps the WebUI and the terminal from showing two different forms because they were initialized differently.
    record["message"] = record["message"].replace(PROJECT_ROOT, ".")
    return LOG_RECORD_FORMAT


def configure_terminal_logger(sink, level: str, colorize: bool = True) -> int:
    """
    Safely replace the process-level terminal log handler and keep task-specific handlers.

    Streamlit may rerun log setup on a code reload or a cache miss. Remove only the previously recorded
    handler id's old terminal output, so a background task's WebUI
    log is not interrupted. The lock protects handler-id updates when several browser sessions initialize at once.
    """
    global _terminal_handler_id

    with _terminal_handler_lock:
        if _terminal_handler_id is not None:
            try:
                logger.remove(_terminal_handler_id)
            except ValueError:
                # Tests or an external entry may already have removed this handler. Keep creating the new terminal output;
                # other sinks that are still valid do not need to be affected.
                pass

        _terminal_handler_id = logger.add(
            sink,
            level=level,
            format=format_log_record,
            colorize=colorize,
        )
        return _terminal_handler_id
