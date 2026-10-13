"""Check whether a newer stable release of Script-to-Reality Renderer Studio is available."""

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import requests
from loguru import logger
from packaging.version import InvalidVersion, Version


LATEST_RELEASE_API_URL: Final = ""
LATEST_RELEASE_PAGE_URL: Final = ""
# The update check is only a helper, so a network failure must not noticeably slow the local WebUI. Connect and read each have a
# timeout that still lets GitHub respond on a normal network and avoids a long wait when offline.
RELEASE_CHECK_TIMEOUT: Final = (1.0, 2.0)
RELEASE_CHECK_HEADERS: Final = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "User-Agent": "Script-to-Reality-Version-Checker",
}
UPDATE_CHECK_CACHE_TTL_SECONDS: Final = 12 * 60 * 60


def _parse_version(value: str) -> Version:
    """Accept GitHub's usual ``v1.2.3`` tag and convert it to a comparable version."""
    normalized = str(value or "").strip()
    if normalized.lower().startswith("v"):
        normalized = normalized[1:]
    return Version(normalized)


def get_available_update(current_version: str) -> str | None:
    """
    Return the newest stable version newer than the current one; return ``None`` when there is no update or the check fails.

    GitHub's ``releases/latest`` endpoint already excludes drafts and prereleases, so this does not
    filter release state again. The WebUI calls this function in the background through ``AsyncUpdateChecker``;
    network, response-format, or version-tag failures only log and degrade to "show no notice", and do not affect
    core features such as video generation.
    """
    if not LATEST_RELEASE_API_URL:
        return None

    try:
        installed_version = _parse_version(current_version)
    except InvalidVersion:
        logger.warning(
            f"skip update check because current version is invalid: {current_version!r}"
        )
        return None

    try:
        response = requests.get(
            LATEST_RELEASE_API_URL,
            headers=RELEASE_CHECK_HEADERS,
            timeout=RELEASE_CHECK_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        # A failed update check is a recoverable non-core error. Keep the exception type and message to locate proxy,
        # DNS, GitHub rate-limit, or corrupt-response problems without bothering ordinary users in the WebUI.
        logger.debug(
            "GitHub release check failed: "
            f"error_type={type(exc).__name__}, error={exc}"
        )
        return None

    if not isinstance(payload, dict):
        logger.debug(
            "GitHub release check returned an invalid payload: "
            f"payload_type={type(payload).__name__}"
        )
        return None

    tag_name = payload.get("tag_name", "")
    try:
        latest_version = _parse_version(tag_name)
    except InvalidVersion:
        logger.warning(
            f"skip update notification because release tag is invalid: {tag_name!r}"
        )
        return None

    if latest_version <= installed_version:
        return None

    normalized_latest_version = str(latest_version)
    logger.info(
        "Script-to-Reality Renderer Studio update available: "
        f"current={installed_version}, latest={normalized_latest_version}"
    )
    return normalized_latest_version


@dataclass(frozen=True)
class UpdateCheckSnapshot:
    """The current background version-check state, for the WebUI to read without blocking."""

    complete: bool
    available_version: str | None = None


class AsyncUpdateChecker:
    """
    Run the version check on a background thread and cache the latest result.

    Streamlit reruns the page script from the top after any widget interaction. If the title area called
    GitHub directly, the first open or a cache miss would block the whole page. The network request runs on a daemon thread,
    and the page only reads the current snapshot; after the check, a short-lived WebUI fragment refreshes the result once.

    Both "update found" and "no update / network failure" are cached, so an unreachable GitHub
    is not requested again on every rerun. The lock only protects in-memory state, not the network request, so
    other sessions can still read the check state.
    """

    def __init__(
        self,
        check: Callable[[str], str | None] = get_available_update,
        ttl_seconds: float = UPDATE_CHECK_CACHE_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._check = check
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._current_version: str | None = None
        self._available_version: str | None = None
        self._completed_at: float | None = None
        self._checking = False
        self._request_id = 0

    def poll(self, current_version: str) -> UpdateCheckSnapshot:
        """Return the check snapshot immediately; start one new background check when the cache has expired."""
        normalized_current_version = str(current_version or "").strip()
        now = self._clock()

        with self._lock:
            cache_is_fresh = (
                self._current_version == normalized_current_version
                and self._completed_at is not None
                and now - self._completed_at < self._ttl_seconds
            )
            if cache_is_fresh:
                return UpdateCheckSnapshot(
                    complete=True,
                    available_version=self._available_version,
                )

            if (
                self._checking
                and self._current_version == normalized_current_version
            ):
                return UpdateCheckSnapshot(complete=False)

            # When the version changes or the cache expires, the old result must not stay on screen. Clear state before starting
            # a new thread, so the caller gets a clear pending snapshot while the check runs.
            self._current_version = normalized_current_version
            self._available_version = None
            self._completed_at = None
            self._checking = True
            self._request_id += 1

            try:
                worker = threading.Thread(
                    target=self._run_check,
                    args=(normalized_current_version, self._request_id),
                    name="mpt-version-check",
                    daemon=True,
                )
                worker.start()
            except RuntimeError as exc:
                # Thread exhaustion must not leave every subsequent poll pending.
                logger.warning(f"could not start update check thread: {exc}")
                self._checking = False
                self._completed_at = now
                return UpdateCheckSnapshot(complete=True)

        return UpdateCheckSnapshot(complete=False)

    def _run_check(self, current_version: str, request_id: int) -> None:
        try:
            available_version = self._check(current_version)
        except Exception:
            # get_available_update already handles expected network and data errors. This is the background thread's
            # last guard and must log the full stack, so an unexpected exception cannot die silently and stay pending forever.
            logger.exception(
                "unexpected error while checking for a Script-to-Reality Renderer Studio update"
            )
            available_version = None

        with self._lock:
            # A -> B -> A can launch two workers for the same version. Only the
            # latest request may complete the current snapshot.
            if self._request_id != request_id:
                return
            self._available_version = available_version
            self._completed_at = self._clock()
            self._checking = False


_ASYNC_UPDATE_CHECKER = AsyncUpdateChecker()


def poll_available_update(current_version: str) -> UpdateCheckSnapshot:
    """Read the global background checker so different Streamlit sessions do not each request GitHub."""
    return _ASYNC_UPDATE_CHECKER.poll(current_version)
