import copy
import errno
import os
import shutil
import socket
import tempfile
import threading
from contextlib import contextmanager

import toml
from loguru import logger

from app import __version__

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
config_file = f"{root_dir}/config.toml"


def _load_local_dotenv():
    """Load gitignored .env values without overriding existing environment variables.

    Values are never logged. A missing file is normal; config.toml remains the
    primary configuration file.
    """
    env_path = os.path.join(root_dir, ".env")
    if not os.path.isfile(env_path):
        return
    try:
        with open(env_path, encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.lower().startswith("export "):
                    line = line[7:].strip()
                key, value = line.split("=", 1)
                key = key.strip()
                if not key or key in os.environ:
                    continue
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                    value = value[1:-1]
                if value:
                    os.environ[key] = value
    except OSError:
        logger.warning("could not read .env; continuing with the existing environment")


_load_local_dotenv()
_CONTAINER_CGROUP_MARKERS = ("docker", "containerd", "kubepods", "libpod", "podman")
_DOCKER_HOST_GATEWAY_NAME = "host.docker.internal"
_config_save_lock = threading.RLock()
_pending_config_lock = threading.RLock()
_pending_config_updates = {}
_pending_config_save_requested = False
_pending_config_flush_scheduled = False
_MISSING = object()
_DELETE = object()
_UTF8_BOM = "\ufeff"


class _SynchronizedConfig(dict):
    """Keep dict usage unchanged, while runtime config writes all obey the same lock."""

    def __setitem__(self, key, value):
        # Every full-page Streamlit rerun writes the current widget values back into config. While a video task holds
        # runtime_config_lock, a write that does not change the value has no side effect and
        # must not leave the refreshed page stuck in the middle of the form. Writes that really change config still take the lock below,
        # so Provider, keys, or other global settings cannot switch halfway through a video that is generating.
        current = super().get(key, _MISSING)
        if current is not _MISSING and current == value:
            return
        with _config_save_lock:
            super().__setitem__(key, value)

    def __delitem__(self, key):
        with _config_save_lock:
            super().__delitem__(key)

    def clear(self):
        if not self:
            return
        with _config_save_lock:
            super().clear()

    def pop(self, key, default=_MISSING):
        # ``pop(key, default)`` also leaves config unchanged when the key is missing. The WebUI uses
        # that spelling to mean "use the default policy", and a refresh must be allowed to finish it directly.
        if key not in self:
            if default is _MISSING:
                raise KeyError(key)
            return default
        with _config_save_lock:
            if default is _MISSING:
                return super().pop(key)
            return super().pop(key, default)

    def setdefault(self, key, default=None):
        # As with __setitem__, setdefault on a key that already exists is a read-only operation. Returning early
        # lets a page that only reads default config refresh without waiting on the long-task config lock.
        current = super().get(key, _MISSING)
        if current is not _MISSING:
            return current
        with _config_save_lock:
            return super().setdefault(key, default)

    def update(self, *args, **kwargs):
        changes = dict(*args, **kwargs)
        if all(
            (current := dict.get(self, key, _MISSING)) is not _MISSING
            and current == value
            for key, value in changes.items()
        ):
            return
        with _config_save_lock:
            super().update(changes)


def _pending_update_key(config_section, key):
    """Build the pending-update key for a fixed in-process config section."""
    return id(config_section), key


def update_config_nonblocking(config_section, key, value):
    """
    Update the WebUI's runtime config without blocking.

    Video generation holds ``runtime_config_lock`` so one task does not switch
    Provider, keys, or voice settings halfway through. A Streamlit widget change cannot wait on that long-task lock,
    or the browser looks frozen. If the lock is free, update immediately; if it is busy, keep only the latest value
    of each setting and apply them together when the current task releases the lock.

    Return True if the value is already in effect, or False if it is queued.
    """
    # Every update enters the same queue first, then tries to take the config lock. When several pages change the same
    # setting at once, queue order is the final order, and an earlier thread cannot, after it gets the lock,
    # delete a newer value that another thread has already queued.
    with _pending_config_lock:
        _pending_config_updates[_pending_update_key(config_section, key)] = (
            config_section,
            key,
            copy.deepcopy(value),
        )

    acquired = _config_save_lock.acquire(blocking=False)
    if not acquired:
        # The caller usually asks for a save at the end of this Streamlit rerun, but that step
        # is not guaranteed to run. If the page errors halfway, or the update lands just as the task exits its save phase,
        # a background refresh thread still has to make the queued value take effect.
        _schedule_deferred_config_flush()
        return False

    try:
        _apply_pending_config_updates_locked()
        return config_section.get(key, _MISSING) == value
    finally:
        _config_save_lock.release()


def delete_config_nonblocking(config_section, key):
    """
    Delete a WebUI config item without blocking.

    "Use the default" must actually remove the item, not write an empty string. While a video task holds the config
    lock, a delete replaces any earlier queued update for that item and runs after the task ends.
    """
    with _pending_config_lock:
        _pending_config_updates[_pending_update_key(config_section, key)] = (
            config_section,
            key,
            _DELETE,
        )

    acquired = _config_save_lock.acquire(blocking=False)
    if not acquired:
        _schedule_deferred_config_flush()
        return False

    try:
        _apply_pending_config_updates_locked()
        return key not in config_section
    finally:
        _config_save_lock.release()


def _apply_pending_config_updates_locked():
    """While holding the config write lock, apply the latest values the WebUI has staged."""
    with _pending_config_lock:
        updates = list(_pending_config_updates.values())
        _pending_config_updates.clear()
        # Keep holding the pending-update lock while applying config. A thread reading a "current value + pending value" snapshot
        # therefore sees only the complete state before the apply or after it, never a config set that was updated halfway.
        for config_section, key, value in updates:
            if value is _DELETE:
                config_section.pop(key, None)
            else:
                config_section[key] = value
    return bool(updates)


def snapshot_config_with_pending(config_section):
    """
    Return an effective snapshot of a config section, merged with WebUI updates that are not applied yet.

    Global config cannot be rewritten while a video task holds the lock, but the user can still prepare the next piece. After an LLM request
    uses this snapshot, the Provider, model, and key just chosen in the UI take part in the new request without
    changing the video task that is already running.
    """
    with _pending_config_lock:
        snapshot = dict(config_section)
        section_id = id(config_section)
        for (pending_section_id, key), (_, _, value) in _pending_config_updates.items():
            if pending_section_id != section_id:
                continue
            if value is _DELETE:
                snapshot.pop(key, None)
            else:
                snapshot[key] = copy.deepcopy(value)
    return snapshot


def _flush_pending_config_locked(*, suppress_save_errors):
    """While holding the config write lock, apply and save every pending config change."""
    global _pending_config_save_requested

    updates_applied = _apply_pending_config_updates_locked()
    with _pending_config_lock:
        save_requested = _pending_config_save_requested
        _pending_config_save_requested = False

    if not updates_applied and not save_requested:
        return True

    try:
        save_config()
        return True
    except Exception as exc:
        # Config in memory has already been applied. If the save fails, only the pending-save flag is kept. A video task must not
        # be marked failed because the config file is temporarily unwritable; the next page interaction will try to save again.
        with _pending_config_lock:
            _pending_config_save_requested = True
        if not suppress_save_errors:
            raise
        logger.exception(f"failed to save deferred runtime config: {exc}")
        return False


def _run_deferred_config_flush():
    """Wait for the long task to release the config lock, and reliably flush config updates queued in the meantime."""
    global _pending_config_flush_scheduled

    while True:
        with _config_save_lock:
            flush_succeeded = _flush_pending_config_locked(
                suppress_save_errors=True
            )

        with _pending_config_lock:
            has_pending_work = bool(
                _pending_config_updates or _pending_config_save_requested
            )
            if not flush_succeeded or not has_pending_work:
                _pending_config_flush_scheduled = False
                return


def _schedule_deferred_config_flush():
    """Ensure at most one background thread is waiting to refresh config at a time."""
    global _pending_config_flush_scheduled

    with _pending_config_lock:
        if _pending_config_flush_scheduled:
            return
        _pending_config_flush_scheduled = True

    try:
        threading.Thread(
            target=_run_deferred_config_flush,
            name="mpt-config-flush",
            daemon=True,
        ).start()
    except Exception:
        # No worker owns this reservation when construction/start fails.
        # Leave queued updates and the save request intact so a later call
        # can schedule another flush rather than silently waiting forever.
        with _pending_config_lock:
            _pending_config_flush_scheduled = False
        raise


def try_save_config():
    """
    Save WebUI config without blocking; if the lock is busy, save after the current long task ends.

    Ordinary API, CLI, and maintenance scripts can still call ``save_config`` for the original blocking write;
    only a Streamlit rerun uses this function, so the page does not sit unresponsive while it waits on a video task.
    """
    global _pending_config_save_requested

    with _pending_config_lock:
        _pending_config_save_requested = True

    acquired = _config_save_lock.acquire(blocking=False)
    if not acquired:
        _schedule_deferred_config_flush()
        return False

    try:
        return _flush_pending_config_locked(suppress_save_errors=False)
    finally:
        _config_save_lock.release()


@contextmanager
def runtime_config_lock():
    """
    Stop other WebUI sessions from rewriting config during one full operation that depends on global config.

    This project binds the local loopback address by default, and config is still single-user global config. This light lock mainly
    protects long operations such as generation and preview so another tab cannot switch Provider or keys halfway through.
    """
    with _config_save_lock:
        # If the background refresh thread has not been scheduled yet when the previous short operation releases the lock, the new task must,
        # before reading global config such as Provider and keys, apply the queue first and not run the whole pipeline on the old config.
        _flush_pending_config_locked(suppress_save_errors=True)
        try:
            yield
        finally:
            _flush_pending_config_locked(suppress_save_errors=True)


@contextmanager
def try_runtime_config_lock():
    """
    Try to acquire the runtime config lock and return immediately whether it succeeded.

    A WebUI preview is a short action the user started, and it should not wait minutes while a background video task holds the lock.
    The caller can tell the user to retry soon if the lock was not acquired; once acquired, Provider, keys, and model settings
    still cannot be changed by another session during the preview.
    """
    acquired = _config_save_lock.acquire(blocking=False)
    try:
        if acquired:
            _flush_pending_config_locked(suppress_save_errors=True)
        yield acquired
    finally:
        if acquired:
            _flush_pending_config_locked(suppress_save_errors=True)
            _config_save_lock.release()


def is_running_in_container(
    dockerenv_path: str = "/.dockerenv",
    containerenv_path: str = "/run/.containerenv",
    cgroup_path: str = "/proc/1/cgroup",
) -> bool:
    """
    Return whether the current process is running inside a container.

    This is used mainly to choose Ollama's default address:
    - On an ordinary local run, `localhost` is the user's own machine;
    - Inside a Docker container, `localhost` is the container itself, and reaching Ollama on the host
      usually requires `host.docker.internal`.

    Do not only test whether `/proc/1/cgroup` exists, because ordinary Linux has that file too.
    Return True only when a clear container marker is detected, so non-Docker Linux users are not treated as containers.
    The paths stay injectable so unit tests can cover different runtimes.
    """
    if os.path.isfile(dockerenv_path) or os.path.isfile(containerenv_path):
        return True

    try:
        with open(cgroup_path, mode="r", encoding="utf-8") as fp:
            cgroup_content = fp.read().lower()
    except OSError:
        return False

    return any(marker in cgroup_content for marker in _CONTAINER_CGROUP_MARKERS)


def _can_resolve_hostname(hostname: str) -> bool:
    try:
        socket.gethostbyname(hostname)
    except OSError:
        return False
    return True


def _decode_linux_route_gateway(hex_gateway: str) -> str:
    # The Gateway field in /proc/net/route is little-endian hex, so 010011AC means
    # 172.17.0.1. Parse it separately so that when native Linux Docker has no
    # host.docker.internal DNS record, the host on the container's default gateway can still be tried.
    if len(hex_gateway) != 8:
        raise ValueError("invalid gateway length")

    octets = [
        str(int(hex_gateway[index : index + 2], 16)) for index in range(6, -1, -2)
    ]
    return ".".join(octets)


def get_container_default_gateway_ip(route_path: str = "/proc/net/route") -> str:
    """
    Read the default gateway IP inside a Linux container.

    Docker Desktop usually provides `host.docker.internal`, but native Linux Docker
    does not always provide that DNS name by default. The default gateway can usually serve as a
    fallback for services on the host; if the user's Ollama listens only on 127.0.0.1, the user still has to make
    Ollama listen on a host interface or set `ollama_base_url` manually.
    """
    try:
        with open(route_path, mode="r", encoding="utf-8") as fp:
            route_lines = fp.readlines()
    except OSError:
        return ""

    for line in route_lines[1:]:
        fields = line.strip().split()
        if len(fields) < 3:
            continue

        destination = fields[1]
        gateway = fields[2]
        if destination != "00000000" or gateway == "00000000":
            continue

        try:
            return _decode_linux_route_gateway(gateway)
        except ValueError:
            logger.warning(f"invalid container gateway route entry: {line.strip()}")
            return ""

    return ""


def get_default_ollama_base_url() -> str:
    """
    Return Ollama's default OpenAI-compatible base_url.

    An explicit `ollama_base_url` does not come through here; this only handles the best default
    "when it is not configured". Inside a container the default points at the host; an ordinary local run points at localhost.
    """
    if not is_running_in_container():
        return "http://localhost:11434/v1"

    if _can_resolve_hostname(_DOCKER_HOST_GATEWAY_NAME):
        return f"http://{_DOCKER_HOST_GATEWAY_NAME}:11434/v1"

    gateway_ip = get_container_default_gateway_ip()
    if gateway_ip:
        logger.info(
            "host.docker.internal is not resolvable, fallback to container "
            f"default gateway for Ollama: {gateway_ip}"
        )
        return f"http://{gateway_ip}:11434/v1"

    logger.warning(
        "failed to resolve host.docker.internal and container default gateway; "
        "fallback to host.docker.internal for Ollama"
    )
    return f"http://{_DOCKER_HOST_GATEWAY_NAME}:11434/v1"


def _load_toml_config(config_path: str):
    """
    Load TOML, and tolerate a repeated UTF-8 BOM that a Windows editor may have written.

    ``utf-8-sig`` removes only one BOM at the start of the file. Some Windows editors, or an
    extract-and-save flow, may write a BOM again, so a second invisible character reaches the TOML
    parser and errors on the first line. After the standard parse fails, do one read-only normalization,
    and do not write the original file back, so an API key the user already filled in is not overwritten by accident.
    """
    try:
        return toml.load(config_path)
    except (toml.TomlDecodeError, UnicodeDecodeError) as exc:
        logger.warning(
            "load config failed, retry with UTF-8 BOM compatibility: "
            f"path={config_path}, error={type(exc).__name__}: {exc}"
        )

    try:
        with open(config_path, mode="r", encoding="utf-8-sig") as fp:
            config_content = fp.read()

        normalized_content = config_content.lstrip(_UTF8_BOM)
        removed_bom_count = len(config_content) - len(normalized_content)
        if removed_bom_count:
            logger.warning(
                "removed repeated UTF-8 BOM characters while loading config: "
                f"path={config_path}, count={removed_bom_count}"
            )
        return toml.loads(normalized_content)
    except (toml.TomlDecodeError, UnicodeDecodeError) as exc:
        logger.error(
            "config file is not valid TOML after UTF-8 BOM normalization: "
            f"path={config_path}, error={type(exc).__name__}: {exc}"
        )
        raise



def _initialize_config_from_example(example_file):
    """Publish a complete first-boot config without replacing another writer."""
    fd, temp_path = tempfile.mkstemp(
        prefix=".config-init-", suffix=".toml.tmp",
        dir=os.path.dirname(os.path.abspath(config_file)),
    )
    try:
        with os.fdopen(fd, "wb") as destination:
            with open(example_file, "rb") as source:
                shutil.copyfileobj(source, destination)
            destination.flush()
            os.fsync(destination.fileno())
        try:
            if os.name == "nt":
                # Windows rename refuses an existing destination. POSIX rename
                # overwrites it, so use an exclusive hard-link publication there.
                os.rename(temp_path, config_file)
            else:
                os.link(temp_path, config_file)
        except FileExistsError:
            return False
        except OSError as exc:
            raise OSError(
                exc.errno,
                "cannot safely initialize config.toml; copy config.example.toml "
                "to config.toml before starting the application",
            ) from exc
        return True
    finally:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            # A successful Windows rename consumed the temporary path.
            pass


def load_config():
    # Docker may create an empty directory when a missing config.toml is bind-mounted.
    # Only remove that empty placeholder; a nonempty directory may contain user data.
    if os.path.isdir(config_file):
        try:
            os.rmdir(config_file)
        except FileNotFoundError:
            # Another initializer already removed this empty bind-mount stub.
            pass
        except OSError as exc:
            # Another initializer may already have replaced the empty Docker
            # stub with its complete regular file. Accept only that race, not
            # permission failures or a still-present/nonempty directory.
            if not (
                isinstance(exc, NotADirectoryError) and os.path.isfile(config_file)
            ):
                raise IsADirectoryError(
                    f"{config_file} is a directory and cannot be used as the config file; "
                    "move or rename it, then restart"
                ) from exc

    if not os.path.isfile(config_file):
        example_file = f"{root_dir}/config.example.toml"
        if os.path.isfile(example_file):
            if _initialize_config_from_example(example_file):
                logger.info("copy config.example.toml to config.toml")

    logger.info(f"load config from file: {config_file}")

    return _load_toml_config(config_file)


def save_config():
    """
    Save runtime config atomically.

    Different Streamlit sessions may save config at nearly the same time. Writing config.toml in place
    can let another thread read a TOML file that is only partly written. A process-wide reentrant lock serializes
    the save: write a temp file in the same directory, then atomically replace the target with os.replace.

    A Docker Desktop single-file bind mount makes config.toml itself the mount point,
    and the Linux kernel does not allow rename/replace of a mount point, so it returns EBUSY.
    In that case the only option is to overwrite the file in place while holding the lock; other exceptions are still raised so permission, disk,
    or path errors are not hidden.

    This still keeps the project's existing single-user global config meaning and does not add a complex multi-user config system;
    it mainly avoids corrupting the config file when several tabs or a fast rerun save at once.
    """
    with _config_save_lock:
        config_to_save = dict(_cfg)
        config_to_save["app"] = dict(app)
        config_to_save["azure"] = dict(azure)
        config_to_save["siliconflow"] = dict(siliconflow)
        config_to_save["minimax_tts"] = dict(minimax_tts)
        config_to_save["elevenlabs"] = dict(elevenlabs)
        config_to_save["chatterbox"] = dict(chatterbox)
        config_to_save["kokoro"] = dict(kokoro)
        config_to_save["fish_audio"] = dict(fish_audio)
        config_to_save["voxcpm"] = dict(voxcpm)
        config_to_save["ui"] = dict(ui)
        serialized_config = toml.dumps(config_to_save)

        # A full WebUI rerun calls save when it finishes. If the content is unchanged, return immediately so every
        # click on an ordinary widget does not cause a disk write and an fsync.
        try:
            with open(config_file, mode="r", encoding="utf-8") as f:
                if f.read() == serialized_config:
                    _cfg.clear()
                    _cfg.update(config_to_save)
                    return
        except (OSError, UnicodeError):
            pass

        temp_path = ""
        try:
            fd, temp_path = tempfile.mkstemp(
                prefix=".config-",
                suffix=".toml.tmp",
                dir=root_dir,
            )
            with os.fdopen(fd, mode="w", encoding="utf-8") as f:
                f.write(serialized_config)
                f.flush()
                os.fsync(f.fileno())
            try:
                os.replace(temp_path, config_file)
            except OSError as exc:
                if exc.errno != errno.EBUSY:
                    raise

                logger.warning(
                    "atomic config replacement is unavailable for the mounted "
                    f"file, fallback to in-place write: {config_file}"
                )
                with open(config_file, mode="w", encoding="utf-8") as f:
                    f.write(serialized_config)
                    f.flush()
                    os.fsync(f.fileno())
            _cfg.clear()
            _cfg.update(config_to_save)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)


_cfg = load_config()
app = _SynchronizedConfig(_cfg.get("app", {}))
whisper = _cfg.get("whisper", {})
proxy = _cfg.get("proxy", {})
azure = _SynchronizedConfig(_cfg.get("azure", {}))
siliconflow = _SynchronizedConfig(_cfg.get("siliconflow", {}))
minimax_tts = _SynchronizedConfig(_cfg.get("minimax_tts", {}))
elevenlabs = _SynchronizedConfig(_cfg.get("elevenlabs", {}))
chatterbox = _SynchronizedConfig(_cfg.get("chatterbox", {}))
kokoro = _SynchronizedConfig(_cfg.get("kokoro", {}))
fish_audio = _SynchronizedConfig(_cfg.get("fish_audio", {}))
voxcpm = _SynchronizedConfig(_cfg.get("voxcpm", {}))
ui = _SynchronizedConfig(
    _cfg.get(
        "ui",
        {
            "hide_log": False,
        },
    )
)

hostname = socket.gethostname()

log_level = _cfg.get("log_level", "DEBUG")
listen_host = _cfg.get("listen_host", "0.0.0.0")
listen_port = _cfg.get("listen_port", 8080)
project_name = _cfg.get("project_name", "Script-to-Reality Renderer Studio")
project_description = _cfg.get(
    "project_description",
    "Script-to-Reality Renderer Studio",
)
project_version = _cfg.get("project_version", __version__)
reload_debug = False

app["redis_host"] = os.getenv(
    "MPT_APP_REDIS_HOST",
    os.getenv("REDIS_HOST", app.get("redis_host", "localhost")),
)

ffmpeg_path = app.get("ffmpeg_path", "")
if ffmpeg_path and os.path.isfile(ffmpeg_path):
    os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_path

logger.info(f"{project_name} v{project_version}")
