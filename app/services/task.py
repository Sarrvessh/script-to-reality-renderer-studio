import json
import math
import os
import re
import socket
import tempfile
import threading
import time
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from functools import partial
from os import path
from uuid import uuid4

from loguru import logger

from app.config import config
from app.models import const
from app.models.schema import VideoConcatMode, VideoParams
from app.services import bgm as bgm_service
from app.services import (
    elevenlabs_music,
    llm,
    loomloom,
    material,
    metaso_minimax,
    muapi,
    ofox,
    openrouter_video,
    sonilo,
    subtitle,
    task_artifacts,
    twelvelabs,
    video,
    volcengine_seedance,
    voice,
)
from app.services import upload_post
from app.services import state as sm
from app.utils import file_security, utils
from app.utils.subtitle_writer import staged_subtitle_file


# A publish request can wait several minutes and must not keep occupying a video-generation concurrency slot.
# A fixed-size thread pool caps publish throughput, and lets the video output
# enter the completed state as soon as it is produced.
_cross_post_executor = ThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="mpt-cross-post",
)
_cross_post_max_pending_tasks = max(
    1,
    int(config.app.get("upload_post_max_pending_tasks", 10)),
)
_cross_post_slots = threading.BoundedSemaphore(_cross_post_max_pending_tasks)
_cross_post_registry_lock = threading.RLock()
_cross_post_futures: dict[str, Future] = {}
_cross_post_process_owner = f"{socket.gethostname()}:{os.getpid()}:{uuid4().hex}"
_ACTIVE_CROSS_POST_STATES = {
    const.CROSS_POST_STATE_PENDING,
    const.CROSS_POST_STATE_PROCESSING,
}
_CROSS_POST_STATE_WRITE_ATTEMPTS = 3
_CROSS_POST_STATE_RETRY_DELAY_SECONDS = 0.1
_TASK_FAILURE_STATE_WRITE_ATTEMPTS = 3
_TASK_FAILURE_STATE_RETRY_DELAY_SECONDS = 0.1
_LOOMLOOM_STATE_WRITE_ATTEMPTS = 3
_LOOMLOOM_STATE_RETRY_DELAY_SECONDS = 0.1
_INTERRUPTED_CROSS_POST_ERROR = (
    "cross-posting was interrupted before the process completed"
)
# Map upload-post platform ids to the social platform names llm.py accepts.
_CROSS_POST_SOCIAL_PLATFORMS = {
    "tiktok": "tiktok",
    "instagram": "instagram_reels",
    "facebook": "facebook_reels",
}
# A video-music service only needs to implement ``is_enabled`` and ``generate_bgm``. Provider differences sit in
# the file extension, domain exceptions, and WebUI warning codes; task orchestration, the 0-volume short circuit, and failure fallback
# all share one path so a new provider does not require another copy of the same flow.
_VIDEO_MUSIC_PROVIDERS = {
    "sonilo": {
        "service": sonilo,
        "error_type": sonilo.SoniloError,
        "suffix": ".m4a",
        "warning_code": "sonilo_bgm_failed",
        "display_name": "Sonilo",
    },
    "elevenlabs": {
        "service": elevenlabs_music,
        "error_type": elevenlabs_music.ElevenLabsMusicError,
        "suffix": ".mp3",
        "warning_code": "elevenlabs_bgm_failed",
        "display_name": "ElevenLabs",
    },
}


_SUPPORTED_VIDEO_SOURCES = frozenset({
    "pexels", "pixabay", "coverr", "local", "wavespeed",
    "volcengine_seedance", "ofox", "metaso_minimax", "muapi",
    "loomloom", "openai_image", "openrouter",
})


def _get_video_music_prompt(params: VideoParams) -> str:
    """
    Read the prompt the current video-music provider actually uses.

    New tasks use the provider-agnostic field; old Sonilo CLI arguments and historical tasks may still have only
    ``sonilo_bgm_prompt``, so the legacy field is read only when the Sonilo generic field is empty.
    """
    prompt = str(params.video_music_prompt or "").strip()
    if params.bgm_type == "sonilo" and not prompt:
        prompt = str(params.sonilo_bgm_prompt or "").strip()
    return prompt


def is_task_busy(task: dict | None) -> bool:
    """Return whether the task is still generating or publishing, for every delete entry point."""
    if not task:
        return False

    state = task.get("state")
    try:
        state = int(state)
    except (TypeError, ValueError):
        pass

    # Video generation and cross-posting may both keep reading the task directory. Treating both as busy
    # avoids the API and WebUI each keeping their own rule and ending up with one allowing delete and the other
    # forbidding it.
    return (
        state == const.TASK_STATE_PROCESSING
        or task.get("cross_post_state") in _ACTIVE_CROSS_POST_STATES
    )


def _register_cross_post_future(task_id: str, future: Future) -> None:
    """Register the publish Future held by this process, for startup recovery and tests of real running state."""
    with _cross_post_registry_lock:
        _cross_post_futures[task_id] = future


def _unregister_cross_post_future(task_id: str, future: Future | None = None) -> None:
    """Remove only the matching Future, so an old callback does not drop newer work registered for the same task."""
    with _cross_post_registry_lock:
        current = _cross_post_futures.get(task_id)
        if current is None or (future is not None and current is not future):
            return
        _cross_post_futures.pop(task_id, None)


def _is_cross_post_active_in_process(task_id: str) -> bool:
    """Return whether this process still holds an unfinished publish task."""
    with _cross_post_registry_lock:
        future = _cross_post_futures.get(task_id)
        return future is not None and not future.done()


def _is_windows_process_alive(process_id: int) -> bool:
    """Check process state through a read-only Win32 API so os.kill does not terminate the process by mistake."""
    import ctypes

    process_query_limited_information = 0x1000
    still_active = 259
    error_access_denied = 5
    error_invalid_parameter = 87
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    # ctypes treats an undeclared return value as a 32-bit int. A 64-bit Windows process handle can
    # be truncated because of that, so the Win32 function signatures must be declared before the call.
    kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.GetExitCodeProcess.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_ulong),
    ]
    kernel32.GetExitCodeProcess.restype = ctypes.c_int
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int
    handle = kernel32.OpenProcess(
        process_query_limited_information,
        False,
        process_id,
    )
    if not handle:
        error_code = ctypes.get_last_error()
        if error_code == error_invalid_parameter:
            return False
        if error_code == error_access_denied:
            # If the process exists but the current user cannot query it, treat it as alive. Otherwise
            # a publish task running under another account could be reclaimed by mistake.
            return True
        logger.warning(
            "failed to open cross-post owner process on Windows, "
            f"process_id: {process_id}, error_code: {error_code}"
        )
        return True

    try:
        exit_code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            error_code = ctypes.get_last_error()
            logger.warning(
                "failed to read cross-post owner process state on Windows, "
                f"process_id: {process_id}, error_code: {error_code}"
            )
            return True
        return exit_code.value == still_active
    finally:
        kernel32.CloseHandle(handle)


def _is_cross_post_owner_alive(owner: str | None) -> bool:
    """Return whether the local process for a persisted publish task still exists."""
    if not owner:
        return False

    try:
        hostname, process_id_text, _ = owner.split(":", 2)
        process_id = int(process_id_text)
    except (TypeError, ValueError):
        logger.warning(f"invalid cross-post owner metadata: {owner}")
        return False

    # Processes on other hosts cannot be probed reliably. In a multi-host deployment sharing Redis, conservatively
    # treat them as still running so this node does not delete video files another node is reading.
    if hostname != socket.gethostname():
        return True

    # Whether this process still has real publish work is already known from the Future registry. Reaching
    # here means there is no matching Future. Even if the owner matches this process exactly, treat it
    # as interrupted; that covers a final-state write that keeps failing after the Future has finished.
    if process_id == os.getpid():
        return False

    # On Windows, os.kill(pid, 0) does not match POSIX and may terminate the target process.
    # Use a Win32 API that only requests query access and sends no signal to the target.
    if os.name == "nt":
        return _is_windows_process_alive(process_id)

    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        logger.warning(
            f"failed to inspect cross-post owner process, owner: {owner}, error: {exc}"
        )
        return True
    return True


def _mark_task_failed(
    task_id: str,
    stage: str,
    error: str,
    details: dict | None = None,
) -> dict:
    """Record structured failure details and keep the progress the task had already reached."""
    existing_task = None
    try:
        existing_task = sm.state.get_task(task_id)
    except Exception as exc:
        logger.warning(f"failed to read task state before failure update: {exc}")

    # The concrete service function usually has a more accurate error than the orchestrator. A later empty-result check
    # must not overwrite it with generic copy, or API callers still see only a vague message.
    if (
        existing_task
        and existing_task.get("state") == const.TASK_STATE_FAILED
        and existing_task.get("error")
    ):
        return existing_task

    message = str(error or "unknown task error").strip()
    progress = int((existing_task or {}).get("progress", 0) or 0)
    logger.error(f"task failed, task_id: {task_id}, stage: {stage}, error: {message}")
    failure = {
        "task_id": task_id,
        "state": const.TASK_STATE_FAILED,
        "progress": progress,
        "failed_stage": stage,
        "error": message,
    }
    # Some external tasks have already created a remote ID that can be used to recover or debug. The failed state must keep
    # those non-sensitive fields, but callers must not be allowed to overwrite the shared status, progress, and error shape.
    # Keep completed batch outputs in both the persisted state and the direct
    # return value used by the CLI. These fields are reset before a new attempt.
    failure_details = {
        key: existing_task[key]
        for key in ("videos", "combined_videos", "warnings", "failed_video_index")
        if existing_task and key in existing_task
    }
    failure_details.update({
        key: value for key, value in dict(details or {}).items() if key not in failure
    })
    failure.update(failure_details)
    fields = {key: value for key, value in failure.items() if key != "task_id"}
    for attempt in range(_TASK_FAILURE_STATE_WRITE_ATTEMPTS):
        try:
            if attempt == 0:
                sm.state.update_task(task_id, **fields)
            else:
                # A lost acknowledgement may mean the first write succeeded.
                # Patch only an existing record so retries cannot revive a task
                # that the user has since deleted.
                sm.state.patch_task(task_id, **fields)
            break
        except Exception as exc:
            if attempt + 1 == _TASK_FAILURE_STATE_WRITE_ATTEMPTS:
                logger.exception(
                    f"failed to persist task failure, task_id: {task_id}, error: {exc}"
                )
            else:
                time.sleep(_TASK_FAILURE_STATE_RETRY_DELAY_SECONDS)
    # A state outage must not discard the only completed-file snapshot held by
    # the caller. Return it even when all bounded persistence attempts fail.
    return failure


def _stage_progress_reporter(task_id: str, stage_start: float, stage_end: float):
    """
    Return a callback that turns a stage's 0~1 completion ratio into task progress and writes it.

    The pipeline used to update progress only at stage boundaries, so during long stages such as downloading footage
    and processing clips the bar did not move and users could not tell running from stuck. Progress is display-only:
    an unparseable ratio is ignored, and a failed status-backend write is only logged. It must not stop download or
    compositing.
    """

    def report(fraction) -> None:
        try:
            fraction = max(0.0, min(1.0, float(fraction)))
        except (TypeError, ValueError):
            return
        try:
            sm.state.update_task(
                task_id,
                state=const.TASK_STATE_PROCESSING,
                progress=stage_start + (stage_end - stage_start) * fraction,
            )
        except Exception as exc:
            logger.warning(
                f"failed to update task progress: task_id={task_id}, "
                f"error={type(exc).__name__}, detail={exc}"
            )

    return report


def generate_script(task_id, params):
    logger.info("\n\n## generating video script")
    video_script = params.video_script.strip()
    if not video_script:
        video_script = llm.generate_script(
            video_subject=params.video_subject,
            language=params.video_language,
            paragraph_number=params.paragraph_number,
            video_script_prompt=params.video_script_prompt,
            custom_system_prompt=params.custom_system_prompt,
        )
    else:
        logger.debug(f"video script: \n{video_script}")

    if not video_script:
        _mark_task_failed(task_id, "script", "failed to generate video script")
        return None

    return video_script


def generate_terms(task_id, params, video_script):
    logger.info("\n\n## generating video terms")
    video_terms = params.video_terms
    if not video_terms:
        # Once footage is matched to the copy in order, the keywords themselves must be generated in script order;
        # otherwise later ordered download and ordered concatenation can only reuse one global topic list
        # and cannot fix "pictures for later lines showing up too early".
        video_terms = llm.generate_terms(
            video_subject=params.video_subject,
            video_script=utils.remove_pause_tags(video_script),
            amount=8 if params.match_materials_to_script else 5,
            match_script_order=params.match_materials_to_script,
        )
    else:
        if isinstance(video_terms, str):
            video_terms = [term.strip() for term in re.split(r"[,，]", video_terms)]
        elif isinstance(video_terms, list):
            video_terms = [term.strip() for term in video_terms]
        else:
            raise ValueError("video_terms must be a string or a list of strings.")

        # Delimiter-only input must not reach paid reranking or material search.
        video_terms = [term for term in video_terms if term]

        logger.debug(f"video terms: {utils.to_json(video_terms)}")

    if not video_terms:
        _mark_task_failed(
            task_id,
            "terms",
            "failed to generate video search terms",
        )
        return None

    # Optional TwelveLabs Marengo semantic rerank: when disabled, the original order is returned with no side effects.
    # In ordered-match mode the keyword order is already the script's narrative order, so it must stay as-is and is skipped.
    if not params.match_materials_to_script:
        video_terms = twelvelabs.rerank_terms_by_subject(
            video_subject=params.video_subject,
            search_terms=video_terms,
        )

    return video_terms


def save_script_data(task_id, video_script, video_terms, params):
    script_data = {
        "script": video_script,
        "search_terms": video_terms,
        "params": params,
    }
    task_artifacts.write_script_data(task_id, script_data)


def resolve_custom_audio_file(
    task_id: str,
    custom_audio_file: str | None,
    *,
    allow_server_file_input: bool = False,
) -> str:
    requested_file = (custom_audio_file or "").strip()
    if not requested_file:
        return ""

    task_dir = utils.task_dir(task_id)
    try:
        return file_security.resolve_path_within_directory(
            task_dir,
            requested_file,
        )
    except ValueError as exc:
        task_dir_error = exc

    # A missing path that otherwise stays inside the task directory is safe to
    # report precisely. Paths outside that boundary use the same generic error
    # regardless of whether they exist, so callers cannot probe the host filesystem.
    if str(task_dir_error) == "file does not exist":
        raise task_dir_error

    # HTTP requests and other untrusted callers must never turn a submitted path
    # into a server-side file read. WebUI uploads already live in the task directory;
    # only the local CLI explicitly opts into resolving files elsewhere on the host.
    if not allow_server_file_input:
        raise ValueError(
            "custom audio file must be stored within the current task directory"
        ) from task_dir_error

    server_audio_file = path.realpath(
        requested_file
        if path.isabs(requested_file)
        else path.join(utils.root_dir(), requested_file)
    )
    if not path.isabs(requested_file):
        project_root = path.realpath(utils.root_dir())
        try:
            if path.commonpath([project_root, server_audio_file]) != project_root:
                raise ValueError(
                    "relative custom audio paths must stay within the project directory"
                )
        except ValueError as exc:
            raise ValueError(
                "custom audio file must be task-local or an existing server-side file"
            ) from exc

    if not path.isfile(server_audio_file):
        raise ValueError(
            "custom audio file does not exist or is not a file"
        ) from task_dir_error

    return server_audio_file


def _resolve_reusable_voice_preview(
    task_id: str,
    params,
    video_script: str,
    voice_preview: dict | None,
) -> tuple[str, float, object] | None:
    """
    Validate and resolve the full preview cache submitted by the WebUI.

    This payload is not a public API parameter; it can come only from the WebUI in this process. Even so, the background task
    rechecks the copy and every voice parameter and requires the audio to stay in the current task directory; any mismatch
    falls back to ordinary TTS so a stale preview cannot pollute the final video.
    """
    if not voice_preview:
        return None

    expected_values = {
        "script": str(video_script or "").strip(),
        "voice_name": params.voice_name,
        "voice_rate": float(params.voice_rate),
        "voice_volume": float(params.voice_volume),
    }
    if not math.isclose(float(params.voice_volume), 1.0) or any(
        voice_preview.get(key) != value for key, value in expected_values.items()
    ):
        logger.info(
            f"skip stale voice preview cache, task_id: {task_id}, "
            "reason: voice parameters changed"
        )
        return None

    preview_file = path.realpath(str(voice_preview.get("audio_file") or ""))
    task_root = path.realpath(utils.task_dir(task_id))
    try:
        preview_is_task_local = path.commonpath([task_root, preview_file]) == task_root
    except ValueError:
        preview_is_task_local = False

    duration = voice_preview.get("duration")
    sub_maker = voice_preview.get("sub_maker")
    if (
        not preview_is_task_local
        or not path.isfile(preview_file)
        or not isinstance(duration, (int, float))
        or not math.isfinite(duration)
        or duration <= 0
        or sub_maker is None
    ):
        logger.warning(
            f"skip invalid voice preview cache, task_id: {task_id}, "
            f"audio_file: {preview_file or '<empty>'}"
        )
        return None

    logger.info(
        f"using full voice preview audio, task_id: {task_id}, duration: {duration:.2f}s"
    )
    return preview_file, math.ceil(duration), sub_maker


def generate_audio(
    task_id,
    params,
    video_script,
    voice_preview=None,
    *,
    allow_server_file_input: bool = False,
    voxcpm_reference_audio: bytes | None = None,
    voxcpm_prompt_audio: bytes | None = None,
    voxcpm_prompt_text: str = "",
):
    """
    Generate audio for the video script.
    If a custom audio file is provided, it will be used directly.
    There will be no subtitle maker object returned in this case.
    Otherwise, TTS will be used to generate the audio.
    Returns:
        - audio_file: path to the generated or provided audio file
        - audio_duration: duration of the audio in seconds
        - sub_maker: subtitle maker object if TTS is used, None otherwise
    """
    logger.info("\n\n## generating audio")
    # The /audio and /subtitle request models do not include custom_audio_file.
    # Read it compatibly here so a direct call does not raise an attribute error.
    requested_custom_audio_file = getattr(params, "custom_audio_file", None)
    try:
        custom_audio_file = resolve_custom_audio_file(
            task_id,
            requested_custom_audio_file,
            allow_server_file_input=allow_server_file_input,
        )
    except ValueError as exc:
        _mark_task_failed(
            task_id,
            "audio",
            f"invalid custom audio file: {exc}",
        )
        return None, None, None

    if not custom_audio_file:
        reusable_preview = _resolve_reusable_voice_preview(
            task_id,
            params,
            video_script,
            voice_preview,
        )
        if reusable_preview:
            return reusable_preview

        logger.info("no custom audio file provided, using TTS to generate audio.")
        audio_file = path.join(utils.task_dir(task_id), "audio.mp3")
        tts_kwargs = {
            "text": video_script,
            "voice_name": voice.parse_voice_name(params.voice_name),
            "voice_rate": params.voice_rate,
            "voice_file": audio_file,
        }
        if voxcpm_reference_audio is not None:
            tts_kwargs["voxcpm_reference_audio"] = voxcpm_reference_audio
        if voxcpm_prompt_audio is not None:
            tts_kwargs["voxcpm_prompt_audio"] = voxcpm_prompt_audio
            tts_kwargs["voxcpm_prompt_text"] = voxcpm_prompt_text
        sub_maker = voice.tts(**tts_kwargs)
        if sub_maker is None:
            _mark_task_failed(
                task_id,
                "audio",
                "failed to synthesize audio; verify the selected voice and TTS connectivity",
            )
            return None, None, None
        # Measure the real written audio_file, not sub_maker.cues[-1].end:
        # the latter is the last WORD BOUNDARY, and TTS leaves a fixed tail
        # past it (Edge TTS: ~0.88s at any length - 19% of a 7-word clip but
        # 1.4% of a 153-word one, so short scripts suffer most). The
        # under-count sizes paid generate_bgm() calls, is reported as
        # audio_duration to the API/WebUI, and under-sources
        # download_videos() material, scaled by video_count.
        file_duration = voice.get_audio_duration(audio_file)
        audio_duration = math.ceil(
            file_duration if file_duration > 0 else voice.get_audio_duration(sub_maker)
        )
        if audio_duration == 0:
            _mark_task_failed(task_id, "audio", "generated audio duration is zero")
            return None, None, None
        return audio_file, audio_duration, sub_maker
    else:
        logger.info(f"using custom audio file: {custom_audio_file}")
        audio_duration = voice.get_audio_duration(custom_audio_file)
        if audio_duration == 0:
            _mark_task_failed(
                task_id,
                "audio",
                "custom audio duration is zero",
            )
            return None, None, None
        return custom_audio_file, audio_duration, None


def generate_subtitle(task_id, params, video_script, sub_maker, audio_file):
    """
    Generate subtitle for the video script.
    If subtitle generation is disabled or no subtitle maker is provided, it will return an empty string.
    Otherwise, it will generate the subtitle using the specified provider.
    Returns:
        - subtitle_path: path to the generated subtitle file
    """
    logger.info("\n\n## generating subtitle")
    if not params.subtitle_enabled:
        return ""

    final_subtitle_path = path.join(utils.task_dir(task_id), "subtitle.srt")
    subtitle_provider = config.app.get("subtitle_provider", "edge").strip().lower()
    logger.info(f"\n\n## generating subtitle, provider: {subtitle_provider}")

    if not subtitle_provider:
        logger.info("subtitle provider is empty, skip subtitle generation")
        return ""

    if sub_maker is None and subtitle_provider != "whisper":
        # Custom audio does not go through TTS, so there is no sub_maker timeline
        # from Edge/Azure or other TTS. Only Whisper can transcribe subtitles straight from the audio file;
        # other subtitle providers keep their existing behavior so a wrong empty timeline is not produced.
        logger.warning(
            "subtitle maker is missing, skip subtitle generation for provider: "
            f"{subtitle_provider}"
        )
        return ""

    is_word_level = getattr(params, "subtitle_display_mode", "sentence") == "word_by_word"

    # A failed retry must never reuse captions from an earlier narration.
    with staged_subtitle_file(final_subtitle_path) as subtitle_path:
        if subtitle_provider == "edge":
            voice.create_subtitle(
                text=video_script,
                sub_maker=sub_maker,
                subtitle_file=subtitle_path,
                word_level=is_word_level,
            )
            if not os.path.exists(subtitle_path):
                # Edge subtitles sometimes produce no file because the timeline cannot be matched to the copy. This must not
                # automatically switch to Whisper, or the first failure would download several GB of
                # model without the user knowing. Load the model only when Whisper is configured explicitly. On Edge failure, keep
                # the video without subtitles and record why, so there is no surprise network or disk cost.
                logger.warning(
                    "edge subtitle generation did not produce a subtitle file; "
                    "skip subtitles without falling back to whisper"
                )
                return ""

        if subtitle_provider == "whisper":
            subtitle.create(
                audio_file=audio_file,
                subtitle_file=subtitle_path,
                word_level=is_word_level,
            )
            if not subtitle.file_to_subtitles(subtitle_path):
                logger.warning("whisper produced no usable subtitle cues")
                return ""
            if not is_word_level:
                logger.info("\n\n## correcting subtitle")
                subtitle.correct(subtitle_file=subtitle_path, video_script=video_script)

        subtitle_lines = subtitle.file_to_subtitles(subtitle_path)
        if not subtitle_lines:
            logger.warning(f"subtitle file is invalid: {subtitle_path}")
            return ""

        os.replace(subtitle_path, final_subtitle_path)
        return final_subtitle_path


def get_video_materials(
    task_id,
    params,
    video_terms,
    audio_duration,
    loomloom_video_request: loomloom.LoomLoomConfirmedVideoRequest | None = None,
    subtitle_path: str = "",
):
    if params.video_source == "local":
        logger.info("\n\n## preprocess local materials")
        materials = video.preprocess_video(
            materials=params.video_materials, clip_duration=params.video_clip_duration
        )
        if not materials:
            _mark_task_failed(
                task_id,
                "materials",
                "no valid local video materials were found",
            )
            return None
        return [material_info.url for material_info in materials]
    elif params.video_source == "loomloom":
        if not isinstance(
            loomloom_video_request, loomloom.LoomLoomConfirmedVideoRequest
        ):
            _mark_task_failed(
                task_id,
                "materials",
                "LoomLoom video generation requires a confirmed quote",
            )
            return None

        request = loomloom_video_request
        logger.info(
            "\n\n## generating "
            f"{len(request.batch.input_rows)} video materials with LoomLoom"
        )
        run_id = ""
        try:
            request.validate()
            backend = loomloom.LoomLoomVideoBackend(request.settings)
            execution = backend.execute(
                request.batch,
                client_request_id=request.client_request_id,
                listing_version_id=request.listing_version_id,
                confirm=True,
            )
            run_id = execution.run_id
            # A return from execute means the paid task was accepted remotely. Write the run ID to the
            # process log first. Even if Redis or another status backend is then unavailable, operators can still use the log
            # to find the task on Shengsuanyun. The only identifier must not live only in a local variable.
            logger.info(
                "LoomLoom paid video run created: "
                f"task_id={task_id}, run_id={run_id}, "
                f"listing_version_id={request.listing_version_id}"
            )
            # Record the remote ID as soon as a paid task is created. Even if later polling times out, the log and task
            # state can still help the user or platform support find and recover output that was already generated. A status-backend
            # failure can only reduce observability; it must not stop a remote task that is already billing, or the download of its output.
            _record_loomloom_run_reference(
                task_id=task_id,
                run_id=run_id,
                listing_version_id=request.listing_version_id,
            )
            backend.wait_for_run(run_id)
            return list(
                backend.download_video_results(
                    run_id,
                    utils.task_dir(task_id),
                )
            )
        except (loomloom.LoomLoomError, ValueError) as exc:
            _mark_task_failed(
                task_id,
                "materials",
                str(exc),
                details={
                    "loomloom_run_id": run_id,
                    "loomloom_listing_version_id": request.listing_version_id,
                },
            )
            return None
    elif params.video_source == "openrouter":
        # Original AI clips only. Stock search is not a fallback.
        logger.info("\n\n## generating original scenes with OpenRouter")
        try:
            from app.services import autonomous
            return autonomous.generate_for_task(
                task_id, params, audio_duration=audio_duration, subtitle_path=subtitle_path,
            )
        except openrouter_video.OpenRouterVideoError as exc:
            remote_job_id = str(getattr(exc, "job_id", "") or "").strip()
            details = {"openrouter_video_job_id": remote_job_id} if remote_job_id else None
            _mark_task_failed(task_id, "materials", str(exc), details=details)
            return None
    else:
        logger.info(f"\n\n## downloading videos from {params.video_source}")
        # Ordered matching applies only when the user turns it on explicitly. Force footage downloads to rotate through keywords
        # in order so one early keyword does not download too much footage and push later script topics off the final timeline.
        try:
            downloaded_videos = material.download_videos(
                task_id=task_id,
                search_terms=video_terms,
                source=params.video_source,
                video_aspect=params.video_aspect,
                video_concat_mode=(
                    VideoConcatMode.sequential
                    if params.match_materials_to_script
                    else params.video_concat_mode
                ),
                audio_duration=audio_duration * params.video_count,
                max_clip_duration=params.video_clip_duration,
                match_script_order=params.match_materials_to_script,
                # The footage stage occupies 40%~50%: advance once per finished file so on a slow network
                # the progress bar does not sit at 40% for a long time.
                progress_callback=_stage_progress_reporter(task_id, 40, 50),
            )
        except volcengine_seedance.VolcEngineSeedanceError as exc:
            # Both an unconfirmed state and a generated-but-download-failed state map to a remote
            # task that can be recovered in the Ark console. Write the failure from the exception's task_id in one place so branches
            # do not each keep their own recovery info and drop it again when the code grows.
            remote_task_id = str(getattr(exc, "task_id", "") or "").strip()
            details = (
                {"volcengine_seedance_task_id": remote_task_id}
                if remote_task_id
                else None
            )
            _mark_task_failed(
                task_id,
                "materials",
                str(exc),
                details=details,
            )
            return None
        except (
            material.WaveSpeedUnconfirmedTaskError,
            material.WaveSpeedDownloadError,
        ) as exc:
            # Once a paid prediction exists, a polling or download failure must
            # leave its ID in task state. Earlier successful clips cannot turn
            # this incomplete run into an apparently completed video task.
            prediction_id = str(getattr(exc, "prediction_id", "") or "").strip()
            details = {"wavespeed_prediction_id": prediction_id} if prediction_id else None
            _mark_task_failed(task_id, "materials", str(exc), details=details)
            return None
        except material.OpenAIImagePaidResultError as exc:
            # A paid request with an uncertain response, failed result download,
            # or failed local render must stop before another image order.
            _mark_task_failed(task_id, "materials", str(exc))
            return None
        except ofox.OFoxError as exc:
            # Same recovery meaning as Ark: both an unconfirmed state and a generated-but-download-failed state map to a remote
            # task that can be recovered in the OFox console. Write the failure from the exception's task_id in one place.
            remote_task_id = str(getattr(exc, "task_id", "") or "").strip()
            details = (
                {"ofox_task_id": remote_task_id} if remote_task_id else None
            )
            _mark_task_failed(
                task_id,
                "materials",
                str(exc),
                details=details,
            )
            return None
        except metaso_minimax.MetasoMiniMaxError as exc:
            # Metaso tasks and Ark tasks use different recovery entry points and field names, so they cannot be merged into one
            # vague remote_task_id. Keep a clear Provider prefix so the API, WebUI,
            # and ops logs can point straight at the matching platform.
            remote_task_id = str(getattr(exc, "task_id", "") or "").strip()
            details = (
                {"metaso_minimax_task_id": remote_task_id} if remote_task_id else None
            )
            _mark_task_failed(
                task_id,
                "materials",
                str(exc),
                details=details,
            )
            return None
        except muapi.MuAPIError as exc:
            # MuAPI tasks are billable after acceptance.  Preserve the remote
            # request ID on any ambiguous or post-completion failure so the
            # user can recover it from the provider dashboard.
            remote_task_id = str(getattr(exc, "task_id", "") or "").strip()
            details = {"muapi_task_id": remote_task_id} if remote_task_id else None
            _mark_task_failed(
                task_id,
                "materials",
                str(exc),
                details=details,
            )
            return None
        if not downloaded_videos:
            _mark_task_failed(
                task_id,
                "materials",
                f"failed to download video materials from {params.video_source}",
            )
            return None
        return downloaded_videos


def _record_loomloom_run_reference(
    *, task_id: str, run_id: str, listing_version_id: str
) -> bool | None:
    """
    Do everything possible to save an already created paid LoomLoom Run so a status failure does not stop the remote task.

    True means the save succeeded, False means the task record is already gone, and None means the status backend
    is still unavailable after limited retries. Whatever the result, the caller should keep polling and downloading, because
    execute already caused an external paid side effect and stopping the local flow only makes the output harder to recover.
    """
    fields = {
        "loomloom_run_id": run_id,
        "loomloom_listing_version_id": listing_version_id,
    }
    for attempt in range(1, _LOOMLOOM_STATE_WRITE_ATTEMPTS + 1):
        try:
            updated = sm.state.patch_task(task_id, **fields)
        except Exception as exc:
            if attempt >= _LOOMLOOM_STATE_WRITE_ATTEMPTS:
                logger.exception(
                    "failed to persist LoomLoom paid run after retries: "
                    f"task_id={task_id}, run_id={run_id}, attempts={attempt}, "
                    f"error={exc}"
                )
                return None
            logger.warning(
                "retry LoomLoom paid run state update: "
                f"task_id={task_id}, run_id={run_id}, attempt={attempt}, "
                f"error={exc}"
            )
            time.sleep(_LOOMLOOM_STATE_RETRY_DELAY_SECONDS)
            continue

        if updated is False:
            logger.warning(
                "could not persist LoomLoom paid run because task is missing: "
                f"task_id={task_id}, run_id={run_id}"
            )
        return updated

    return None


def _get_material_source_groups(task_id: str, video_paths: list[str]) -> dict[str, str]:
    """Recover keyword groups from the downloaded material manifest."""
    try:
        with open(path.join(utils.task_dir(task_id), "script.json"), encoding="utf-8") as file:
            payload = json.load(file)
        groups = {
            record["local_file"]: record["search_term"]
            for record in payload.get("material_sources", [])
            if isinstance(record, dict)
            and isinstance(record.get("local_file"), str)
            and isinstance(record.get("search_term"), str)
            and record["search_term"]
        }
        return {
            file: groups[path.basename(file)]
            for file in video_paths
            if path.basename(file) in groups
        }
    except (OSError, ValueError, AttributeError, TypeError) as exc:
        logger.warning(f"Cannot read material keyword groups: task_id={task_id}, error={exc}")
        return {}


def generate_final_videos(
    task_id, params, downloaded_videos, audio_file, subtitle_path, audio_duration,
    *, partial_results: dict | None = None,
):
    """Record completed outputs even if a later clip raises before returning."""
    final_video_paths = []
    combined_video_paths = []
    warnings = []
    if partial_results is not None:
        # The caller owns this snapshot so it can report completed files even
        # when the state backend fails during a progress/checkpoint write.
        partial_results.update(
            videos=final_video_paths, combined_videos=combined_video_paths,
            warnings=warnings,
        )
    allocate_batch_materials = params.video_count > 1 and params.video_source in {
        "pexels", "pixabay", "coverr", "local"
    }
    source_usage = {}
    material_selections = []
    source_groups = (
        _get_material_source_groups(task_id, downloaded_videos)
        if (allocate_batch_materials and params.match_materials_to_script
            and params.video_source != "local")
        else {}
    )
    video_music_provider = _VIDEO_MUSIC_PROVIDERS.get(params.bgm_type)
    video_music_requested = (
        video_music_provider is not None
        and bgm_service.should_use_bgm(params.bgm_type, params.bgm_volume)
    )
    # Matching preserves keyword order; batch allocation varies each keyword's candidates.
    if params.match_materials_to_script:
        video_concat_mode = VideoConcatMode.sequential
    elif params.video_count == 1:
        video_concat_mode = params.video_concat_mode
    else:
        video_concat_mode = VideoConcatMode.random
    video_transition_mode = params.video_transition_mode

    _progress = 50
    for i in range(params.video_count):
        index = i + 1
        if partial_results is not None:
            partial_results["failed_video_index"] = index
        combined_video_path = path.join(
            utils.task_dir(task_id), f"combined-{index}.mp4"
        )
        logger.info(f"\n\n## combining video: {index} => {combined_video_path}")
        used_video_paths = []
        batch_options = (
            {
                "source_usage": source_usage,
                "source_groups": source_groups,
                "used_video_paths": used_video_paths,
            }
            if allocate_batch_materials else {}
        )
        # Each video takes one share of the 50% progress, and compositing and final encoding each take half of that. Clip handling is
        # the slowest part of compositing, so advance that half clip by clip.
        combine_share = 50 / params.video_count / 2
        video.combine_videos(
            combined_video_path=combined_video_path,
            video_paths=downloaded_videos,
            audio_file=audio_file,
            video_aspect=params.video_aspect,
            video_fit_mode=params.video_fit_mode,
            video_concat_mode=video_concat_mode,
            video_transition_mode=video_transition_mode,
            max_clip_duration=params.video_clip_duration,
            threads=params.n_threads,
            clip_speed=params.video_clip_speed,
            progress_callback=_stage_progress_reporter(
                task_id, _progress, _progress + combine_share
            ),
            allow_loop=params.video_source != "openrouter",
            **batch_options,
        )
        if allocate_batch_materials:
            selected_sources = list(dict.fromkeys(used_video_paths))
            reused_sources = [file for file in selected_sources if source_usage.get(file, 0)]
            for file in selected_sources:
                source_usage[file] = source_usage.get(file, 0) + 1
            material_selections.append({
                "video_index": index,
                "local_files": [path.basename(file) for file in used_video_paths],
                "reused_files": [path.basename(file) for file in reused_sources],
            })
            task_artifacts.patch_script_data(task_id, material_selections=material_selections)
            logger.info(
                f"Batch material allocation: video_index={index}, "
                f"sources={len(selected_sources)}, reused_sources={len(reused_sources)}"
            )
            if reused_sources:
                warnings.append({
                    "code": "batch_materials_reused",
                    "video_index": index,
                    "count": len(reused_sources),
                })

        _progress += 50 / params.video_count / 2
        sm.state.update_task(task_id, progress=_progress)

        final_video_path = path.join(utils.task_dir(task_id), f"final-{index}.mp4")

        # Video-music mode first disables default BGM resolution explicitly so a leftover bgm_file from an old task is not
        # reused by mistake. Build a proxy and call the paid API only when volume is greater than 0; volume 0 always skips.
        bgm_file_override = "" if video_music_provider else None
        if video_music_requested:
            service = video_music_provider["service"]
            display_name = video_music_provider["display_name"]
            warning_code = video_music_provider["warning_code"]
            generated_bgm_path = path.join(
                utils.task_dir(task_id),
                (f"{params.bgm_type}-bgm-{index}{video_music_provider['suffix']}"),
            )
            try:
                service.generate_bgm(
                    video_path=combined_video_path,
                    output_path=generated_bgm_path,
                    video_duration=audio_duration,
                    prompt=_get_video_music_prompt(params),
                )
                bgm_file_override = generated_bgm_path
            except video_music_provider["error_type"] as exc:
                # If the video, narration, and subtitles are already done, a temporary third-party music failure should not waste the whole
                # task. This video explicitly disables BGM and returns the degraded result to the WebUI so the user is told.
                logger.warning(
                    f"{display_name} BGM generation failed: task_id={task_id}, "
                    f"video_index={index}, error={exc}"
                )
                bgm_file_override = ""
                warnings.append({"code": warning_code, "video_index": index})

        logger.info(f"\n\n## generating video: {index} => {final_video_path}")
        bgm_mix_succeeded = video.generate_video(
            video_path=combined_video_path,
            audio_path=audio_file,
            subtitle_path=subtitle_path,
            output_file=final_video_path,
            params=params,
            bgm_file_override=bgm_file_override,
        )
        if (
            video_music_provider is not None
            and bgm_file_override
            and not bgm_mix_succeeded
        ):
            # The third party already returned audio that passed FFmpeg checks, but MoviePy's final mix can still
            # fail because of the runtime. The video service keeps a cut without BGM; when API generation fails,
            # override is empty, so the warning is not appended twice.
            warnings.append(
                {
                    "code": video_music_provider["warning_code"],
                    "video_index": index,
                }
            )

        final_video_paths.append(final_video_path)
        combined_video_paths.append(combined_video_path)
        _progress += 50 / params.video_count / 2
        sm.state.update_task(
            task_id, progress=_progress, videos=list(final_video_paths),
            combined_videos=list(combined_video_paths), warnings=list(warnings) or None,
        )

    return final_video_paths, combined_video_paths, warnings


def _patch_cross_post_state(task_id: str, **kwargs) -> bool | None:
    """Safely update publish fields, with limited retries on a brief status-backend failure."""
    for attempt in range(1, _CROSS_POST_STATE_WRITE_ATTEMPTS + 1):
        try:
            return sm.state.patch_task(task_id, **kwargs)
        except Exception as exc:
            # A brief Redis disconnect must not leave the task stuck in pending/processing forever. Publish-state
            # writes are rare, so a fixed number of short waits covers a transient fault and
            # avoids blocking a background thread forever. The last failure keeps the full stack for diagnosis.
            if attempt >= _CROSS_POST_STATE_WRITE_ATTEMPTS:
                logger.exception(
                    f"failed to update cross-post state after retries, "
                    f"task_id: {task_id}, fields: {', '.join(kwargs)}, "
                    f"attempts: {attempt}, error: {exc}"
                )
                return None

            logger.warning(
                f"retry cross-post state update, task_id: {task_id}, "
                f"fields: {', '.join(kwargs)}, attempt: {attempt}, error: {exc}"
            )
            time.sleep(_CROSS_POST_STATE_RETRY_DELAY_SECONDS)

    return None


def _record_cross_post_failure(
    task_id: str,
    error: Exception,
    results: list[dict] | None = None,
) -> None:
    """Do everything possible to save a publish failure; if the status backend is down, the log keeps the diagnostics."""
    updated = _patch_cross_post_state(
        task_id,
        cross_post_state=const.CROSS_POST_STATE_FAILED,
        cross_post_results=results or None,
        cross_post_error=str(error),
        cross_post_owner=None,
    )
    if updated is False:
        logger.warning(f"discard cross-post failure for missing task: {task_id}")


def _ensure_cross_post_terminal_state(task_id: str) -> None:
    """After a Future finishes, converge any task still in an active state to failed."""
    try:
        task = sm.state.get_task(task_id)
    except Exception as exc:
        # This is already the Future's final callback; no later synchronous caller can handle the exception.
        # After the status backend recovers, the next process start still handles leftover state through recovery.
        logger.exception(
            f"failed to verify final cross-post state, task_id: {task_id}, error: {exc}"
        )
        return

    if not task or task.get("cross_post_state") not in _ACTIVE_CROSS_POST_STATES:
        return

    logger.warning(
        f"cross-post worker ended without terminal state, task_id: {task_id}, "
        f"state: {task.get('cross_post_state')}"
    )
    _record_cross_post_failure(
        task_id,
        RuntimeError("cross-post worker ended without persisting a terminal state"),
        task.get("cross_post_results"),
    )


def recover_interrupted_cross_posts(page_size: int = 100) -> int | None:
    """
    Mark publish tasks that cannot be resumed after a process restart as failed.

    Cross-posting uses an in-process thread pool, not a persistent task queue. On startup,
    leftover pending/processing rows in Redis do not resume on their own; if they stay marked as
    running, the user can never delete the task. Collect task IDs once, then read the current state;
    do not advance by page across repeated Redis SCANs, or a changed scan order will skip leftover tasks.
    Only active records with no Future in this process are handled, and already generated video results are kept.
    """
    recovered = 0
    try:
        task_ids = sm.state.list_task_ids(scan_count=page_size)
        for task_id in task_ids:
            # After the scan the task may have been deleted or moved to a terminal state; decide recovery from the latest state.
            # Do not overwrite an already finished publish result with the old snapshot from scan time.
            task = sm.state.get_task(task_id)
            if (
                not task
                or task.get("cross_post_state") not in _ACTIVE_CROSS_POST_STATES
                or _is_cross_post_active_in_process(task_id)
                or _is_cross_post_owner_alive(task.get("cross_post_owner"))
            ):
                continue

            updated = _patch_cross_post_state(
                task_id,
                cross_post_state=const.CROSS_POST_STATE_FAILED,
                cross_post_error=_INTERRUPTED_CROSS_POST_ERROR,
                cross_post_owner=None,
            )
            if updated is True:
                recovered += 1
    except Exception as exc:
        logger.exception(f"failed to recover interrupted cross-post tasks: {exc}")
        return None

    if recovered:
        logger.warning(f"recovered interrupted cross-post tasks: {recovered}")
    return recovered


def _run_cross_post(
    task_id: str,
    video_paths: tuple[str, ...],
    video_subject: str,
    video_script: str,
    video_language: str,
    platforms: tuple[str, ...],
    youtube_privacy_status: str,
    youtube_made_for_kids: bool = False,
    upload_account: dict | None = None,
) -> None:
    """Run cross-posting in the background and fill in only the publish-related task fields."""
    results = []
    try:
        state_updated = _patch_cross_post_state(
            task_id,
            cross_post_state=const.CROSS_POST_STATE_PROCESSING,
            cross_post_error=None,
            cross_post_owner=_cross_post_process_owner,
        )
        if state_updated is not True:
            # False means the task was deleted, and None means the status backend is temporarily unavailable. In both cases
            # do not keep calling the third-party API, or the user cannot query or control this publish.
            if state_updated is False:
                logger.warning(f"skip cross-post for missing task: {task_id}")
            else:
                _record_cross_post_failure(
                    task_id,
                    RuntimeError("failed to persist cross-post processing state"),
                )
            return

        logger.info(
            f"cross-post started, task_id: {task_id}, platforms: {', '.join(platforms)}"
        )
        youtube_extra = None
        post_title = video_subject or "Check out this video! #shorts #viral"
        if platforms:
            has_youtube = any(platform.startswith("youtube") for platform in platforms)
            social_platform = "youtube_shorts"
            if not has_youtube:
                first = (platforms[0] or "").strip().lower()
                # llm.py resolves unknown ids to its default platform.
                social_platform = _CROSS_POST_SOCIAL_PLATFORMS.get(first, first)
            metadata = llm.generate_social_metadata(
                video_subject=video_subject,
                video_script=video_script,
                language=video_language or "",
                platform=social_platform,
            )
            if has_youtube:
                youtube_extra = {
                    "youtube_title": metadata.get("title", video_subject),
                    "youtube_description": metadata.get("caption", ""),
                    "tags": metadata.get("hashtags", []),
                    "privacyStatus": youtube_privacy_status,
                    "selfDeclaredMadeForKids": youtube_made_for_kids,
                    "containsSyntheticMedia": True,
                }
            post_title = (
                metadata.get("caption")
                or metadata.get("title")
                or video_subject
                or "Check out this video! #shorts #viral"
            )

        for video_path in video_paths:
            pending_result_index = None

            def record_background_request(request_id: str) -> None:
                nonlocal pending_result_index
                # Persist the remote handle before polling. If this process
                # exits mid-upload, recovery can expose the ID to the user.
                pending_result_index = len(results)
                results.append(
                    {
                        "video_index": pending_result_index + 1,
                        "request_id": request_id,
                        "status": "processing",
                    }
                )
                if _patch_cross_post_state(
                    task_id, cross_post_results=list(results)
                ) is not True:
                    logger.warning(
                        "could not persist background upload request ID: "
                        f"task_id={task_id}, request_id={request_id}"
                    )

            upload_kwargs = {}
            if upload_account is not None:
                upload_kwargs["account"] = upload_account
            result = upload_post.cross_post_video(
                video_path=video_path,
                title=post_title,
                platforms=list(platforms),
                youtube_extra=youtube_extra,
                on_background_start=record_background_request,
                **upload_kwargs,
            )
            if not isinstance(result, dict):
                result = {
                    "success": False,
                    "error": "Upload-Post returned an invalid response",
                }
            if pending_result_index is None:
                results.append(result)
            else:
                results[pending_result_index] = result

        failures = [result for result in results if not result.get("success")]
        if failures:
            error_messages = [
                str(
                    result.get("error")
                    or result.get("message")
                    or "unknown upload error"
                )
                for result in failures
            ]
            cross_post_state = const.CROSS_POST_STATE_FAILED
            cross_post_error = "; ".join(error_messages)
            logger.warning(
                f"cross-post completed with failures, task_id: {task_id}, "
                f"failed: {len(failures)}, total: {len(results)}"
            )
        else:
            cross_post_state = const.CROSS_POST_STATE_COMPLETE
            cross_post_error = None
            logger.success(
                f"cross-post completed, task_id: {task_id}, videos: {len(results)}"
            )

        state_updated = _patch_cross_post_state(
            task_id,
            cross_post_state=cross_post_state,
            cross_post_results=results,
            cross_post_error=cross_post_error,
            cross_post_owner=None,
        )
        if state_updated is False:
            logger.warning(f"discard cross-post result for missing task: {task_id}")
        elif state_updated is None:
            # If the upload has finished but the result was not persisted, do not leave the task in processing.
            # Writing the failed state goes through limited retries again so the caller at least gets a clear terminal state.
            _record_cross_post_failure(
                task_id,
                RuntimeError("failed to persist final cross-post result"),
                results,
            )
    except Exception as exc:
        # A publish failure affects only publish state and must not overwrite a video task that already completed.
        # The original exception is stored on the task so API callers can diagnose it without server logs.
        logger.exception(f"cross-post failed, task_id: {task_id}, error: {exc}")
        _record_cross_post_failure(task_id, exc, results)


def _run_cross_post_with_slot(*args) -> None:
    """Run the publish task and always return queue capacity on success, failure, or exception."""
    try:
        _run_cross_post(*args)
    except Exception as exc:
        # _run_cross_post already handles expected exceptions; this is the last guard so an exception from future
        # logic is not left only inside a Future that nobody reads.
        task_id = str(args[0]) if args else "unknown"
        logger.exception(f"cross-post worker crashed, task_id: {task_id}, error: {exc}")
        if args:
            _record_cross_post_failure(task_id, exc)
    finally:
        _cross_post_slots.release()


def _finalize_cross_post_future(task_id: str, future: Future) -> None:
    """Clear the Future registration and converge cancellation, exceptions, and failed status writes."""
    _unregister_cross_post_future(task_id, future)

    try:
        error = future.exception()
    except CancelledError:
        logger.warning(f"cross-post future was cancelled, task_id: {task_id}")
        # If the Future is cancelled before it starts, the worker's finally does not run, so the callback must
        # return queue capacity and set the persisted state to failed.
        _cross_post_slots.release()
        _record_cross_post_failure(
            task_id,
            RuntimeError("cross-post job was cancelled before execution"),
        )
        return
    except Exception as exc:
        logger.exception(
            f"failed to inspect cross-post future, task_id: {task_id}, error: {exc}"
        )
        _ensure_cross_post_terminal_state(task_id)
        return

    if error is not None:
        logger.error(
            f"cross-post future failed, task_id: {task_id}, "
            f"error: {type(error).__name__}: {error}"
        )

    _ensure_cross_post_terminal_state(task_id)


def _schedule_cross_post(
    task_id: str,
    video_paths: list[str],
    params: VideoParams,
    video_script: str,
    platforms: list[str],
    youtube_privacy_status: str,
    youtube_made_for_kids: bool = False,
) -> str | None:
    """Submit a background publish task; return None on success, or a queryable error reason if scheduling fails."""
    if not _cross_post_slots.acquire(blocking=False):
        error = "cross-post queue is full; publishing was skipped"
        logger.warning(
            f"skip cross-post because queue is full, task_id: {task_id}, "
            f"capacity: {_cross_post_max_pending_tasks}"
        )
        _patch_cross_post_state(
            task_id,
            cross_post_state=const.CROSS_POST_STATE_FAILED,
            cross_post_error=error,
            cross_post_owner=None,
        )
        return error

    try:
        future = _cross_post_executor.submit(
            _run_cross_post_with_slot,
            task_id,
            tuple(video_paths),
            params.video_subject or "",
            video_script,
            params.video_language or "",
            tuple(platforms),
            youtube_privacy_status,
            youtube_made_for_kids,
            upload_post.upload_post_service.snapshot_account(),
        )
        _register_cross_post_future(task_id, future)
        future.add_done_callback(partial(_finalize_cross_post_future, task_id))
    except RuntimeError as exc:
        _unregister_cross_post_future(task_id)
        _cross_post_slots.release()
        logger.exception(
            f"failed to schedule cross-post, task_id: {task_id}, error: {exc}"
        )
        _patch_cross_post_state(
            task_id,
            cross_post_state=const.CROSS_POST_STATE_FAILED,
            cross_post_error=f"failed to schedule cross-post: {exc}",
            cross_post_owner=None,
        )
        return f"failed to schedule cross-post: {exc}"

    return None


def _run_pipeline(
    task_id,
    params: VideoParams,
    stop_at: str = "video",
    voice_preview: dict | None = None,
    loomloom_video_request: loomloom.LoomLoomConfirmedVideoRequest | None = None,
    allow_server_file_input: bool = False,
    voxcpm_reference_audio: bytes | None = None,
    voxcpm_prompt_audio: bytes | None = None,
    voxcpm_prompt_text: str = "",
):
    logger.info(f"start task: {task_id}, stop_at: {stop_at}")
    sm.state.update_task(
        task_id, state=const.TASK_STATE_PROCESSING, progress=5,
        failed_stage=None, error=None,
        videos=[], combined_videos=[], warnings=None, failed_video_index=None,
    )

    if stop_at == "video" and params.subtitle_enabled:
        try:
            video.validate_subtitle_colors(params)
        except ValueError as exc:
            return _mark_task_failed(task_id, "preflight", str(exc))

    if stop_at in {"materials", "video"} and params.video_source not in _SUPPORTED_VIDEO_SOURCES:
        return _mark_task_failed(
            task_id, "preflight", f"unsupported video source: {params.video_source!r}"
        )

    if (
        stop_at in {"materials", "video"}
        and params.video_source == "volcengine_seedance"
        and not volcengine_seedance.is_enabled()
    ):
        return _mark_task_failed(
            task_id,
            "preflight",
            "Volcano Engine Seedance requires an Ark API key",
        )

    if (
        stop_at in {"materials", "video"}
        and params.video_source == "ofox"
        and not ofox.is_enabled()
    ):
        return _mark_task_failed(
            task_id,
            "preflight",
            "OFox video generation requires an OFox API key",
        )

    if (
        stop_at in {"materials", "video"}
        and params.video_source == "metaso_minimax"
        and not metaso_minimax.is_enabled()
    ):
        return _mark_task_failed(
            task_id,
            "preflight",
            "Metaso MiniMax requires an API key",
        )

    if (
        stop_at in {"materials", "video"}
        and params.video_source == "muapi"
        and not muapi.is_enabled()
    ):
        return _mark_task_failed(
            task_id,
            "preflight",
            "MuAPI video generation requires a MuAPI API key",
        )

    if (
        stop_at in {"materials", "video"}
        and params.video_source == "openrouter"
        and not openrouter_video.is_configured()
    ):
        return _mark_task_failed(
            task_id,
            "preflight",
            "OpenRouter video uses the OpenRouter key already saved for the language model",
        )

    if (
        stop_at in {"materials", "video"}
        and params.video_source == "openai_image"
        and not material.is_openai_image_enabled(
            config.snapshot_config_with_pending(config.app)
        )
    ):
        return _mark_task_failed(
            task_id,
            "preflight",
            "OpenAI image source requires openai_image_base_url and "
            "openai_image_model in config.toml (openai_image_api_keys is "
            "optional for local gateways that need no auth)",
        )

    # Only the full finished-video flow needs a video-music provider. Stop a full task that lacks a key early so it does not
    # spend LLM, TTS, and footage quota first; intermediate-artifact endpoints can still be used on their own.
    video_music_provider = _VIDEO_MUSIC_PROVIDERS.get(params.bgm_type)
    video_music_enabled = (
        stop_at == "video"
        and video_music_provider is not None
        and bgm_service.should_use_bgm(params.bgm_type, params.bgm_volume)
    )
    if video_music_enabled:
        service = video_music_provider["service"]
        display_name = video_music_provider["display_name"]
        if not service.is_enabled():
            return _mark_task_failed(
                task_id,
                "preflight",
                f"{display_name} background music requires an API key",
            )

        # The WebUI limits input length, but the API, CLI, and historical tasks can bypass the form controls.
        # Check the provider limit again before generating the script, voice, and footage so the third party does not reject the request
        # only after the full video is composited. The service layer keeps the same check as a last line of defense for direct calls.
        music_prompt = _get_video_music_prompt(params)
        max_prompt_length = int(getattr(service, "MAX_PROMPT_LENGTH", 0) or 0)
        if max_prompt_length and len(music_prompt) > max_prompt_length:
            return _mark_task_failed(
                task_id,
                "preflight",
                (f"{display_name} music prompt exceeds {max_prompt_length} characters"),
            )

        # A provider may offer an account preflight that does not bill. The check function should raise only deterministic
        # errors; if the network is flaky or the permission scope cannot be confirmed, the service logs a warning and continues generation.
        validate_access = getattr(service, "validate_generation_access", None)
        if callable(validate_access):
            try:
                validate_access()
            except video_music_provider["error_type"] as exc:
                return _mark_task_failed(task_id, "preflight", str(exc))

    # Only script/terms intermediate artifacts do not need FFmpeg (they produce no audio or video). The API,
    # CLI, and WebUI all run tasks through this shared entry, so probe once here instead of
    # repeating the check at each entry, which keeps the three paths consistent. It sits after the music-key check
    # so those checks keep their original "fail first" order and error text.
    if stop_at not in ("script", "terms") and not utils.check_ffmpeg_ready():
        return _mark_task_failed(
            task_id,
            "preflight",
            "ffmpeg is not available; install ffmpeg or set app.ffmpeg_path "
            "in config.toml to a working ffmpeg executable",
        )

    # 1. Generate script
    video_script = generate_script(task_id, params)
    # The LLM adapter uses a leading "Error: " as its failure sentinel. A
    # user-provided script may legitimately quote that text anywhere, including
    # at the beginning, so only interpret it for an LLM-generated script.
    is_provider_error = (
        not (params.video_script or "").strip()
        and isinstance(video_script, str)
        and video_script.startswith("Error: ")
    )
    if not video_script or is_provider_error:
        error = (
            video_script.removeprefix("Error: ").strip()
            if is_provider_error
            else "failed to generate video script"
        )
        return _mark_task_failed(task_id, "script", error)

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=10)

    if stop_at == "script":
        sm.state.update_task(
            task_id, state=const.TASK_STATE_COMPLETE, progress=100, script=video_script
        )
        return {"script": video_script}

    # 2. Generate terms
    video_terms = ""
    if stop_at in {"terms", "materials", "video"} and params.video_source != "local":
        video_terms = generate_terms(task_id, params, video_script)
        if not video_terms:
            return _mark_task_failed(
                task_id,
                "terms",
                "failed to generate video search terms",
            )

    save_script_data(task_id, video_script, video_terms, params)

    if stop_at == "terms":
        sm.state.update_task(
            task_id, state=const.TASK_STATE_COMPLETE, progress=100, terms=video_terms
        )
        return {"script": video_script, "terms": video_terms}

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=20)

    # 3. Generate audio
    generate_audio_kwargs = {
        "voice_preview": voice_preview,
        "allow_server_file_input": allow_server_file_input,
    }
    if voxcpm_reference_audio is not None:
        generate_audio_kwargs["voxcpm_reference_audio"] = voxcpm_reference_audio
    if voxcpm_prompt_audio is not None:
        generate_audio_kwargs["voxcpm_prompt_audio"] = voxcpm_prompt_audio
        generate_audio_kwargs["voxcpm_prompt_text"] = voxcpm_prompt_text
    audio_file, audio_duration, sub_maker = generate_audio(
        task_id,
        params,
        video_script,
        **generate_audio_kwargs,
    )
    if not audio_file:
        return _mark_task_failed(
            task_id,
            "audio",
            "failed to prepare narration audio",
        )

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=30)

    if stop_at == "audio":
        # Full video jobs apply gain in the final mixer. Audio-only exports have
        # no mixer, so apply it here to an owned output (never a custom input).
        audio_volume = 1.0 if params.voice_volume is None else float(params.voice_volume)
        if audio_volume != 1.0:
            descriptor, output_file = tempfile.mkstemp(
                prefix="audio-export-", suffix=".mp3", dir=utils.task_dir(task_id)
            )
            os.close(descriptor)
            export_ready = False
            try:
                export_ready = voice.apply_audio_volume(audio_file, output_file, audio_volume)
                if not export_ready:
                    return _mark_task_failed(task_id, "audio", "failed to apply audio export volume")
            finally:
                if not export_ready:
                    try:
                        os.remove(output_file)
                    except OSError as exc:
                        logger.warning(f"failed to remove incomplete audio export: {exc}")
            audio_file = output_file
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_COMPLETE,
            progress=100,
            audio_file=audio_file,
        )
        return {"audio_file": audio_file, "audio_duration": audio_duration}

    # 4. Generate subtitle
    subtitle_path = generate_subtitle(
        task_id, params, video_script, sub_maker, audio_file
    )

    if stop_at == "subtitle":
        if not subtitle_path:
            return _mark_task_failed(
                task_id,
                "subtitle",
                "failed to generate subtitles; verify the subtitle provider and timeline",
            )
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_COMPLETE,
            progress=100,
            subtitle_path=subtitle_path,
        )
        return {"subtitle_path": subtitle_path}

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=40)

    # 5. Get video materials
    downloaded_videos = get_video_materials(
        task_id,
        params,
        video_terms,
        audio_duration,
        loomloom_video_request=loomloom_video_request,
        subtitle_path=subtitle_path,
    )
    if not downloaded_videos:
        return _mark_task_failed(
            task_id,
            "materials",
            "failed to prepare video materials",
        )

    if stop_at == "materials":
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_COMPLETE,
            progress=100,
            materials=downloaded_videos,
        )
        return {"materials": downloaded_videos}

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=50)

    # Only the full video-generation flow needs to handle the video-concatenation mode;
    # that avoids /subtitle and /audio requests touching fields they do not have.
    if type(params.video_concat_mode) is str:
        params.video_concat_mode = VideoConcatMode(params.video_concat_mode)

    # 6. Generate final videos
    partial_results = {}
    try:
        final_video_paths, combined_video_paths, generation_warnings = generate_final_videos(
            task_id,
            params,
            downloaded_videos,
            audio_file,
            subtitle_path,
            audio_duration,
            partial_results=partial_results,
        )
    except Exception as exc:
        logger.exception(f"video batch failed, task_id: {task_id}, error: {exc}")
        # Warnings for the failing clip do not describe any usable output.
        completed_count = len(partial_results.get("videos", []))
        partial_results["warnings"] = [
            warning for warning in partial_results.get("warnings", [])
            if warning.get("video_index", 0) <= completed_count
        ] or None
        return _mark_task_failed(
            task_id, "video", f"{type(exc).__name__}: {exc}", details=partial_results,
        )

    if not final_video_paths:
        return _mark_task_failed(
            task_id,
            "video",
            "failed to generate final video",
        )

    if params.video_source == "openrouter":
        from app.services import cv_engine
        for path in final_video_paths:
            check = cv_engine.validate_playable(path)
            if not check["playable"]:
                return _mark_task_failed(
                    task_id,
                    "video",
                    "The final MP4 failed validation.",
                    details={"validation": check},
                )

    logger.success(
        f"task {task_id} finished, generated {len(final_video_paths)} videos."
    )

    # 7. Finish the video-generation task first, then submit cross-posting if needed. A third-party upload can take
    # several minutes and must not block returning the video result or affect a finished cut.
    cross_post_enabled = (
        upload_post.upload_post_service.is_configured()
        and upload_post.upload_post_service.auto_upload
    )
    platforms = (
        list(upload_post.upload_post_service.platforms) if cross_post_enabled else []
    )
    should_cross_post = cross_post_enabled and bool(platforms)
    if cross_post_enabled and not platforms:
        logger.warning(
            f"skip cross-post because no platforms are configured, task_id: {task_id}"
        )
    cross_post_state = const.CROSS_POST_STATE_PENDING if should_cross_post else None

    kwargs = {
        "videos": final_video_paths,
        "combined_videos": combined_video_paths,
        "script": video_script,
        "terms": video_terms,
        "audio_file": audio_file,
        "audio_duration": audio_duration,
        "subtitle_path": subtitle_path,
        "materials": downloaded_videos,
        "cross_post_state": cross_post_state,
        "cross_post_results": None,
        "cross_post_error": None,
        "cross_post_owner": _cross_post_process_owner if should_cross_post else None,
        "warnings": generation_warnings or None,
    }
    sm.state.update_task(
        task_id, state=const.TASK_STATE_COMPLETE, progress=100, **kwargs
    )

    if should_cross_post:
        scheduling_error = _schedule_cross_post(
            task_id=task_id,
            video_paths=final_video_paths,
            params=params,
            video_script=video_script,
            platforms=platforms,
            youtube_privacy_status=(
                upload_post.upload_post_service.youtube_privacy_status
            ),
            # Freeze the audience choice at queue time. Later WebUI edits must not change the declaration of a video already queued.
            youtube_made_for_kids=(
                upload_post.upload_post_service.youtube_made_for_kids
            ),
        )
        # A full queue or a shut-down thread pool is a scheduling failure known synchronously. The scheduler has already
        # updated task state, so fix the returned snapshot here and do not hand the caller a pending that later queries disagree with.
        if scheduling_error:
            kwargs["cross_post_state"] = const.CROSS_POST_STATE_FAILED
            kwargs["cross_post_error"] = scheduling_error
            kwargs["cross_post_owner"] = None

    return kwargs


def start(
    task_id,
    params: VideoParams,
    stop_at: str = "video",
    voice_preview: dict | None = None,
    loomloom_video_request: loomloom.LoomLoomConfirmedVideoRequest | None = None,
    allow_server_file_input: bool = False,
    voxcpm_reference_audio: bytes | None = None,
    voxcpm_prompt_audio: bytes | None = None,
    voxcpm_prompt_text: str = "",
):
    """
    Run the task pipeline and turn unexpected exceptions into a queryable failed state.

    ``allow_server_file_input`` is only for the local CLI. The HTTP API and WebUI must keep
    the default so custom audio stays constrained to the current task directory.
    """
    try:
        return _run_pipeline(
            task_id,
            params,
            stop_at=stop_at,
            voice_preview=voice_preview,
            loomloom_video_request=loomloom_video_request,
            allow_server_file_input=allow_server_file_input,
            voxcpm_reference_audio=voxcpm_reference_audio,
            voxcpm_prompt_audio=voxcpm_prompt_audio,
            voxcpm_prompt_text=voxcpm_prompt_text,
        )
    except Exception as exc:
        logger.exception(
            f"unexpected task pipeline failure, task_id: {task_id}, error: {exc}"
        )
        return _mark_task_failed(
            task_id,
            "pipeline",
            f"{type(exc).__name__}: {exc}",
        )


if __name__ == "__main__":
    task_id = "task_id"
    params = VideoParams(
        video_subject="金钱的作用",
        voice_name="zh-CN-XiaoyiNeural-Female",
        voice_rate=1.0,
    )
    start(task_id, params, stop_at="video")
