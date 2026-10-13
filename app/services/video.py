import itertools
import hashlib
from concurrent.futures import ThreadPoolExecutor
import io
import math
import os
import random
import gc
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
from contextlib import ExitStack, contextmanager, redirect_stdout
from functools import lru_cache
from typing import Callable, List
from loguru import logger
import numpy as np
from moviepy import (
    AudioFileClip,
    ColorClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
    VideoFileClip,
    afx,
)
from moviepy.video.tools.subtitles import SubtitlesClip
from PIL import Image, ImageColor, ImageDraw, ImageFont, ImageOps

from app.config import config
from app.models import const
from app.models.schema import (
    MaterialInfo,
    VideoAspect,
    VideoConcatMode,
    VideoFitMode,
    VideoParams,
    VideoTransitionMode,
)
from app.services import bgm as bgm_service
from app.services.utils import video_effects
from app.utils import file_security, logging_utils, utils
from app.utils.subtitle_reader import moviepy_subtitles

class SubClippedVideoClip:
    def __init__(
        self,
        file_path,
        start_time=None,
        end_time=None,
        width=None,
        height=None,
        duration=None,
        source_file_path=None,
    ):
        self.file_path = file_path
        self.start_time = start_time
        self.end_time = end_time
        self.width = width
        self.height = height
        self.source_file_path = source_file_path or file_path
        if duration is None:
            self.duration = end_time - start_time
        else:
            self.duration = duration

    def __str__(self):
        return f"SubClippedVideoClip(file_path={self.file_path}, start_time={self.start_time}, end_time={self.end_time}, duration={self.duration}, width={self.width}, height={self.height})"


audio_codec = "aac"
# The ffmpeg/AAC combination in Docker is more likely to fluctuate in audio quality with the default settings,
# so the audio bitrate is raised explicitly to avoid obvious distortion in the finished video from a too-low default.
audio_bitrate = "192k"
fps = 30
# When FFmpeg concatenates or transcodes at a given frame rate, the final duration can be tens of milliseconds shorter than the theoretical duration MoviePy reads.
# Leave a very small safety margin on the video material so frame rounding at the end of the audio does not cause a black frame,
# a freeze, or a last bit of narration with no picture.
_VIDEO_DURATION_SAFETY_MARGIN = 0.1
_MIN_MATERIAL_DIMENSION = 480
# Messaging apps and some encoders round the frame size down. WhatsApp, for example, squeezes 9:16
# footage to 478x850, two pixels under 480. A hard cutoff at 480 would drop all of that footage
# and fail the whole job with "no valid materials found". A small tolerance here
# lets through footage that is only slightly under the threshold because of rounding, while still rejecting footage that is truly low resolution.
_MIN_DIMENSION_TOLERANCE = 10
# Process segments serially by default so CPU and memory use do not jump for users upgrading from an older version.
# Raise the concurrency yourself if you need speed, but each worker starts its own encode.
_CLIP_PROCESSING_CONCURRENCY = 1


def _get_clip_processing_concurrency() -> int:
    try:
        concurrency = int(config.app.get("video_clip_concurrency", _CLIP_PROCESSING_CONCURRENCY))
    except (TypeError, ValueError):
        concurrency = _CLIP_PROCESSING_CONCURRENCY
    return max(1, min(8, concurrency))
_DEFAULT_VIDEO_CODEC = "libx264"
# ffmpeg logs nothing while it concatenates segments, and `subprocess.run` blocks until the process exits, so a long concat
# looks like "no output" in the logs. Record a heartbeat at an interval so encoding can be told apart from a hang.
_FFMPEG_CONCAT_HEARTBEAT_SECONDS = 30.0
# MoviePy also logs nothing while writing the final video. A one-minute portrait video can take several
# minutes to encode on CPU. Use the same interval heartbeat as the concat stage.
_STAGE_HEARTBEAT_SECONDS = 30.0
_DEFAULT_FFMPEG_CONCAT_TIMEOUT_SECONDS = 3600
_SUBTITLE_SPRING_DURATION_SECONDS = 0.18
_MIN_SUBTITLE_SPRING_SCALE = 0.05
_MAX_SUBTITLE_SPRING_SCALE = 1.35
_SUPPORTED_VIDEO_CODECS = (
    "libx264",
    "h264_nvenc",
    "h264_amf",
    "h264_qsv",
    "h264_mf",
    "h264_videotoolbox",
)
_runtime_disabled_video_codecs = set()
# MoviePy pipes sRGB frames, and ffmpeg's default RGB→YUV matrix is BT.601 with no color tags,
# while players and YouTube decode untagged HD as BT.709 and shift the colors. Convert with the
# BT.709 matrix and tag the stream; setparams writes the tags the -color_* flags alone do not.
# The pixel format stays the encoder's choice: MoviePy leaves odd-sized frames to libx264.
_BT709_VIDEO_FILTER = (
    "scale=out_color_matrix=bt709:out_range=tv,"
    "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv"
)
_BT709_FFMPEG_PARAMS = ["-vf", _BT709_VIDEO_FILTER]


def _get_subtitle_spring_scale(time_seconds: float, duration_seconds: float) -> float:
    """Return the scale used by the subtitle bounce animation at a given time."""
    if duration_seconds <= 0 or time_seconds >= duration_seconds:
        return 1.0

    progress = max(0.0, min(time_seconds / duration_seconds, 1.0))
    scale = 1.0 - math.exp(-6.0 * progress) * math.cos(2.5 * math.pi * progress)
    return max(
        _MIN_SUBTITLE_SPRING_SCALE,
        min(scale, _MAX_SUBTITLE_SPRING_SCALE),
    )


def _scale_subtitle_frame_on_canvas(frame: np.ndarray, scale: float) -> np.ndarray:
    """
    Scale the subtitle image or its transparent mask around the center without changing the canvas size.

    MoviePy stores the subtitle color frame and the transparent mask separately. The bounce animation must scale and
    crop both in exactly the same way, or the first frame of the animation composites the transparent area as a black text outline
    onto the video. A 2D array is a mask with values from 0 to 1, and a 3D array is an RGB/RGBA color frame.
    """
    if frame.ndim not in (2, 3):
        raise ValueError("subtitle frame must be a 2D mask or 3D color frame")

    height, width = frame.shape[:2]
    scaled_width = max(1, int(round(width * scale)))
    scaled_height = max(1, int(round(height * scale)))
    offset = ((width - scaled_width) // 2, (height - scaled_height) // 2)

    if frame.ndim == 2:
        # MoviePy masks use floats from 0 to 1, and Pillow's L mode uses 0 to 255. After converting,
        # restore the original type and range so CompositeVideoClip's transparency meaning stays the same.
        mask_image = Image.fromarray(
            np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        )
        resized_mask = mask_image.resize(
            (scaled_width, scaled_height),
            Image.Resampling.BILINEAR,
        )
        mask_canvas = Image.new("L", (width, height), 0)
        mask_canvas.paste(resized_mask, offset)
        return (np.asarray(mask_canvas) / 255.0).astype(frame.dtype, copy=False)

    if frame.shape[2] not in (3, 4):
        raise ValueError("subtitle color frame must use RGB or RGBA channels")
    color_image = Image.fromarray(frame)
    resized_color = color_image.resize(
        (scaled_width, scaled_height),
        Image.Resampling.BILINEAR,
    )
    background = (0, 0, 0, 0) if frame.shape[2] == 4 else (0, 0, 0)
    color_canvas = Image.new(color_image.mode, (width, height), background)
    color_canvas.paste(resized_color, offset)
    return np.asarray(color_canvas).astype(frame.dtype, copy=False)


def _apply_subtitle_spring_animation(clip, subtitle_duration: float):
    """Scale the subtitle color frame and mask together so the bounce animation does not show a black first frame."""
    animation_duration = min(
        _SUBTITLE_SPRING_DURATION_SECONDS,
        max(0.0, subtitle_duration),
    )
    if animation_duration <= 0:
        return clip

    def transform_frame(get_frame, time_seconds):
        frame = get_frame(time_seconds)
        scale = _get_subtitle_spring_scale(time_seconds, animation_duration)
        if scale == 1.0:
            return frame
        return _scale_subtitle_frame_on_canvas(frame, scale)

    # apply_to=["mask"] is the important part of the fix: MoviePy processes only the color frame by default, so the old code
    # briefly kept the original-size mask when each subtitle appeared and showed a black text outline.
    return clip.transform(transform_frame, apply_to=["mask"])


def _get_required_video_duration(audio_duration: float) -> float:
    """
    Return the target duration for concatenating video materials.

    Use case: when compositing, the material duration needs to cover the narration audio. Stopping at "exactly equal"
    to the audio duration can leave the final video slightly short because of FFmpeg frame-rate rounding, so a small
    margin is always added. The function is separate so the margin can be tested and adjusted from real feedback later.
    """
    return max(0.0, float(audio_duration) + _VIDEO_DURATION_SAFETY_MARGIN)


def is_material_resolution_acceptable(width: int, height: int) -> bool:
    """
    Decide whether a material's resolution is high enough to composite.

    The nominal minimum is 480x480, but it may be `_MIN_DIMENSION_TOLERANCE` pixels smaller,
    to allow sizes rounded down by an encoder or a messaging app (for example WhatsApp's 478x850).
    """
    min_dimension = _MIN_MATERIAL_DIMENSION - _MIN_DIMENSION_TOLERANCE
    return width >= min_dimension and height >= min_dimension


def _prioritize_unique_source_clips(
    subclipped_items: List[SubClippedVideoClip],
    concat_mode: VideoConcatMode,
    source_usage: dict[str, int] | None = None,
    source_groups: dict[str, str] | None = None,
) -> List[SubClippedVideoClip]:
    """
    Prefer to show each source material only once, so the same material is less likely to repeat in the finished video.

    Online material often arrives as "one long video cut into several short segments". The old logic, in
    random mode, shuffled every short segment, so several cuts of the same source video could land
    at the beginning and in the middle, and users read that as repeated footage. This function only reorders segments:
    put the longest segment from each source file first, and keep the rest as backup. When the total material is too short,
    later segments may still fill out the audio length, so video generation can still succeed. Preferring the longest
    segment avoids randomly picking a tiny tail fragment and reusing footage early when there is already enough material.
    """
    if not subclipped_items:
        return []

    concat_mode_value = getattr(concat_mode, "value", concat_mode)
    if concat_mode_value != VideoConcatMode.random.value:
        if source_usage is None:
            return subclipped_items
        if not source_groups:
            return sorted(
                subclipped_items,
                key=lambda item: source_usage.get(item.source_file_path, 0),
            )
        # Keep keyword rounds in order while rotating candidates within each keyword.
        groups = {}
        for item in subclipped_items:
            key = source_groups.get(item.source_file_path, item.source_file_path)
            groups.setdefault(key, []).append(item)
        for items in groups.values():
            items.sort(key=lambda item: source_usage.get(item.source_file_path, 0))
        return [
            item
            for row in itertools.zip_longest(*groups.values())
            for item in row
            if item is not None
        ]

    grouped_items: dict[str, list[SubClippedVideoClip]] = {}
    for item in subclipped_items:
        grouped_items.setdefault(item.source_file_path, []).append(item)

    primary_items = []
    overflow_items = []
    for items in grouped_items.values():
        primary_item = max(items, key=lambda item: item.duration)
        primary_items.append(primary_item)
        overflow_items.extend(item for item in items if item is not primary_item)

    random.shuffle(primary_items)
    random.shuffle(overflow_items)
    if source_usage is not None:
        # Stable sorting retains randomness among equally used sources.
        primary_items.sort(key=lambda item: source_usage.get(item.source_file_path, 0))
        overflow_items.sort(key=lambda item: source_usage.get(item.source_file_path, 0))
    logger.info(
        "prioritized unique video materials, "
        f"sources: {len(grouped_items)}, "
        f"primary clips: {len(primary_items)}, "
        f"fallback clips: {len(overflow_items)}"
    )
    return primary_items + overflow_items


def get_ffmpeg_binary():
    """
    Stay compatible with callers that historically read the FFmpeg path from the video service.

    The real lookup now lives in `app.utils.utils.get_ffmpeg_binary()`. Video, voice,
    and later paths should share that priority order. This thin wrapper remains so external scripts or
    old tests that import `app.services.video.get_ffmpeg_binary` directly do not get an AttributeError.
    """
    return utils.get_ffmpeg_binary()


def _get_configured_video_codec() -> str:
    """
    Read the video encoder configured by the user.

    This setting is for advanced users who want to try hardware encoders such as NVENC/AMF/QSV/VideoToolbox.
    Only a fixed allowlist is accepted, so an open-ended FFmpeg parameter cannot be filled in incorrectly,
    produce an uncontrollable output format, or fail the job only in a later stage.
    """
    configured_codec = str(
        config.app.get("video_codec", _DEFAULT_VIDEO_CODEC) or _DEFAULT_VIDEO_CODEC
    ).strip()
    if configured_codec not in _SUPPORTED_VIDEO_CODECS:
        logger.warning(
            f"unsupported video codec configured: {configured_codec}, "
            f"fallback to {_DEFAULT_VIDEO_CODEC}"
        )
        return _DEFAULT_VIDEO_CODEC
    return configured_codec


@lru_cache(maxsize=16)
def _ffmpeg_encoder_exists(ffmpeg_binary: str, codec: str) -> bool:
    """
    Check whether the current FFmpeg build claims to support the given encoder.

    This only proves the encoder was compiled into FFmpeg. It does not prove the current machine's hardware and drivers
    can actually use it. A real encode failure still falls back to libx264.
    """
    try:
        result = subprocess.run(
            [ffmpeg_binary, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning(
            "failed to inspect ffmpeg encoders, "
            f"fallback to {_DEFAULT_VIDEO_CODEC}: {str(exc)}"
        )
        return False

    if result.returncode != 0:
        logger.warning(
            "failed to inspect ffmpeg encoders, "
            f"fallback to {_DEFAULT_VIDEO_CODEC}: {(result.stderr or result.stdout or '').strip()}"
        )
        return False
    return codec in result.stdout


def _get_effective_video_codec(preferred_codec: str | None = None) -> str:
    """
    Return the video encoder actually used for this run.

    When the user picks a hardware encoder, check FFmpeg's encoder list first. If this process has already
    failed a real encode with it, fall back immediately so every segment in the job does not fail again.
    """
    selected_codec = preferred_codec or _get_configured_video_codec()
    if selected_codec == _DEFAULT_VIDEO_CODEC:
        return _DEFAULT_VIDEO_CODEC

    if selected_codec in _runtime_disabled_video_codecs:
        logger.warning(
            f"video codec {selected_codec} was disabled after a runtime failure, "
            f"fallback to {_DEFAULT_VIDEO_CODEC}"
        )
        return _DEFAULT_VIDEO_CODEC

    ffmpeg_binary = utils.get_ffmpeg_binary()
    if not _ffmpeg_encoder_exists(ffmpeg_binary, selected_codec):
        logger.warning(
            f"ffmpeg encoder {selected_codec} is not available, "
            f"fallback to {_DEFAULT_VIDEO_CODEC}"
        )
        return _DEFAULT_VIDEO_CODEC

    return selected_codec


def _disable_runtime_video_codec(codec: str, reason: str):
    if codec == _DEFAULT_VIDEO_CODEC:
        return
    _runtime_disabled_video_codecs.add(codec)
    logger.warning(
        f"video codec {codec} failed, fallback to {_DEFAULT_VIDEO_CODEC}. "
        f"reason: {reason}"
    )


def _get_temp_audio_dir(output_dir: str) -> str:
    """
    Return the directory to use for MoviePy's temporary audio file.

    On Windows, Windows Defender can lock files written to the task output
    directory while scanning them, causing MoviePy to fail with a
    PermissionError (WinError 32) on the TEMP_MPY_wvf_snd temp file and
    leaving the final MP4 at 0 bytes.  Using the system temp directory
    sidesteps the scan without changing behaviour on other platforms.

    On Linux/macOS/Docker the output directory is returned unchanged so
    existing behaviour is preserved.
    """
    if sys.platform == "win32":
        return tempfile.gettempdir()
    return output_dir


def _fallback_write_videofile(clip, output_file: str, failed_codec: str, reason: str, **kwargs):
    """
    After a hardware encode fails, retry with libx264, and disable that hardware encoder only if the retry succeeds.

    FFmpeg failures on Windows have messy causes: the GPU or driver may not support the encoder, or the output
    file may be locked, the directory may be unwritable, or antivirus software may block it. Only when libx264 can write the file
    is the original failure likely the hardware encoder itself, so later jobs are not disabled by mistake.
    """
    clip.write_videofile(output_file, codec=_DEFAULT_VIDEO_CODEC, **kwargs)
    _disable_runtime_video_codec(failed_codec, reason)
    return _DEFAULT_VIDEO_CODEC


def _write_videofile_with_codec_fallback(
    clip, output_file: str, codec: str, atomic_output: bool = False, **kwargs
):
    """
    Write the video with the chosen encoder, and automatically retry once with libx264 on failure.

    A hardware encoder depends on FFmpeg and also on the GPU, the driver, and the current runtime.
    A job must not fail entirely because an advanced encoder is unavailable, so the fallback is handled in one place.
    """
    kwargs.setdefault("ffmpeg_params", _BT709_FFMPEG_PARAMS)
    if atomic_output:
        # Final videos can be downloaded by path while they are being rendered.
        # Keep both failed encodes and in-progress writes away from that path.
        output_dir = os.path.dirname(os.path.abspath(output_file))
        descriptor, temp_output = tempfile.mkstemp(
            prefix=f".{os.path.basename(output_file)}.",
            suffix=os.path.splitext(output_file)[1] or ".mp4",
            dir=output_dir,
        )
        os.close(descriptor)
        os.unlink(temp_output)
        try:
            used_codec = _write_videofile_with_codec_fallback(
                clip, temp_output, codec, **kwargs
            )
            os.replace(temp_output, output_file)
            return used_codec
        finally:
            try:
                os.unlink(temp_output)
            except FileNotFoundError:
                pass
            except OSError as exc:
                logger.warning(
                    f"failed to remove temporary final video: {temp_output}, "
                    f"error: {exc}"
                )

    effective_codec = _get_effective_video_codec(codec)
    try:
        clip.write_videofile(output_file, codec=effective_codec, **kwargs)
        return effective_codec
    except Exception as exc:
        if effective_codec == _DEFAULT_VIDEO_CODEC:
            raise
        return _fallback_write_videofile(
            clip,
            output_file,
            failed_codec=effective_codec,
            reason=str(exc),
            **kwargs,
        )


def _escape_ffmpeg_concat_path(file_path: str) -> str:
    # The concat demuxer wraps paths in single quotes, so a single quote in the path has to be escaped first.
    return file_path.replace("'", "'\\''")


def _format_ffmpeg_concat_path(file_path: str) -> str:
    """
    Build a path for a concat demuxer file list.

    The official FFmpeg docs require special characters and spaces in a concat list to be escaped. Backslashes in a Windows
    absolute path are also easy to parse as escape characters. Convert them to forward slashes here,
    turn `C:\\Users\\...` into `C:/Users/...`, then handle single quotes, for macOS/Linux compatibility.
    """
    absolute_path = os.path.abspath(file_path)
    # Normalize the native separator only. A backslash can be a literal
    # component of a valid POSIX path and must still identify the same file.
    return _escape_ffmpeg_concat_path(absolute_path.replace(os.sep, "/"))


def _describe_concat_output_progress(output_file: str) -> str:
    """Return a readable description of the output file's current size, for concat heartbeat logs."""
    try:
        size = os.path.getsize(output_file)
    except OSError:
        # If the output file has not been created yet, degrade safely and do not affect the concat itself.
        return "output size not available"
    return f"output size: {size / (1024 * 1024):.2f} MB"


@contextmanager
def _stage_heartbeat(description: str):
    """
    Log a heartbeat at an interval during a long stage that has no logs of its own.

    The heartbeat thread is bound to the thread that entered the stage, so the log shows up on that task's WebUI panel.
    When the stage ends or raises, stop and wait for the heartbeat thread to exit so it does not keep
    writing "still running" after the stage is already over.
    """
    started_at = time.monotonic()
    stop_event = threading.Event()

    def log_heartbeat() -> None:
        while not stop_event.wait(_STAGE_HEARTBEAT_SECONDS):
            logger.info(
                f"{description} still running: "
                f"elapsed={time.monotonic() - started_at:.0f}s"
            )

    reporter = threading.Thread(
        target=logging_utils.bind_log_scope(log_heartbeat), daemon=True
    )
    reporter.start()
    try:
        yield
    finally:
        stop_event.set()
        reporter.join(timeout=1)


def _report_clip_progress(
    progress_callback: Callable[[float], None] | None,
    covered_duration: float,
    required_duration: float,
) -> None:
    """Convert covered finished-video duration into a 0-1 ratio and notify the caller."""
    if progress_callback is None:
        return
    fraction = 1.0
    if required_duration > 0:
        fraction = min(1.0, covered_duration / required_duration)
    try:
        progress_callback(fraction)
    except Exception as exc:
        # Progress is only for display. A failed callback (for example a status backend that is briefly unavailable) must not discard
        # segments that were already processed.
        logger.warning(
            "failed to report clip processing progress: "
            f"error={type(exc).__name__}, detail={exc}"
        )


def _run_concat_with_heartbeat(command: list[str], output_file: str):
    """
    Block until ffmpeg finishes, and log a heartbeat at an interval while waiting.

    ffmpeg logs nothing while concatenating segments, and `subprocess.run` buffers output until the process exits, so a long concat
    looks like "no output" in the logs. Record how long the wait has been and the output file size, so encoding can be told apart from a hang.
    """
    started_at = time.monotonic()
    stop_event = threading.Event()

    def log_heartbeat() -> None:
        while not stop_event.wait(_FFMPEG_CONCAT_HEARTBEAT_SECONDS):
            logger.info(
                "ffmpeg concat still running: "
                f"elapsed={time.monotonic() - started_at:.0f}s, "
                f"{_describe_concat_output_progress(output_file)}"
            )

    reporter = threading.Thread(
        target=logging_utils.bind_log_scope(log_heartbeat), daemon=True
    )
    reporter.start()
    try:
        configured_timeout = config.app.get(
            "ffmpeg_concat_timeout_seconds", _DEFAULT_FFMPEG_CONCAT_TIMEOUT_SECONDS
        )
        try:
            timeout_seconds = float(configured_timeout)
        except (TypeError, ValueError) as exc:
            raise ValueError("ffmpeg_concat_timeout_seconds must be positive") from exc
        if (
            isinstance(configured_timeout, bool)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("ffmpeg_concat_timeout_seconds must be positive")

        try:
            # subprocess.run kills and waits for FFmpeg on timeout, so a stalled
            # encoder cannot leave an orphaned child or a permanently active task.
            return subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"ffmpeg concat exceeded {timeout_seconds:g} seconds"
            ) from exc
    finally:
        stop_event.set()


def concat_video_clips_with_ffmpeg(
    clip_files: List[str],
    output_file: str,
    threads: int,
    output_dir: str,
    max_duration: float | None = None,
):
    # Separate renders may share a directory. Each FFmpeg process must keep its
    # own manifest until all codec attempts finish, without overwriting or
    # deleting another render's list.
    concat_list_file = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix="ffmpeg-concat-", suffix=".txt",
            dir=output_dir, delete=False,
        ) as fp:
            concat_list_file = fp.name
            for clip_file in clip_files:
                fp.write(f"file '{_format_ffmpeg_concat_path(clip_file)}'\n")
    except Exception:
        if concat_list_file:
            delete_files(concat_list_file)
        raise

    staged_output = None

    def build_command(codec: str) -> list[str]:
        command = [
            utils.get_ffmpeg_binary(),
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_list_file,
            "-c:v",
            codec,
            "-threads",
            str(threads or 2),
        ]
        if codec == "libx264":
            command.extend(["-preset", "veryfast"])
        command.extend([
            "-vf",
            _BT709_VIDEO_FILTER,
            "-pix_fmt",
            "yuv420p",
        ])
        if max_duration is not None and max_duration > 0:
            command.extend(["-t", f"{max_duration:.3f}"])
        command.append(staged_output)
        return command

    def run_concat(codec: str):
        command = build_command(codec)
        # Concatenate and encode with ffmpeg only once, instead of re-encoding on every MoviePy segment merge,
        # which reduces quality loss and color shift. The heartbeat log shows the task is still running during the blocking wait.
        result = _run_concat_with_heartbeat(command, staged_output)
        if result.returncode != 0:
            error_message = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(error_message or "ffmpeg concat failed")
        return codec

    try:
        descriptor, staged_output = tempfile.mkstemp(
            prefix=".ffmpeg-concat-",
            suffix=os.path.splitext(output_file)[1] or ".mp4",
            dir=os.path.dirname(os.path.abspath(output_file)),
        )
        os.close(descriptor)
        effective_codec = _get_effective_video_codec()
        try:
            result_codec = run_concat(effective_codec)
        except TimeoutError:
            # A hung encoder is not evidence that another codec will work. Do
            # not spend a second timeout period retrying the same input.
            raise
        except Exception as exc:
            if effective_codec == _DEFAULT_VIDEO_CODEC:
                raise
            result_codec = run_concat(_DEFAULT_VIDEO_CODEC)
            _disable_runtime_video_codec(effective_codec, str(exc))
        # Failed attempts and in-progress output stay private until FFmpeg has
        # finished. Publication errors must not trigger another codec attempt.
        if os.path.getsize(staged_output) == 0:
            raise RuntimeError("ffmpeg concat produced no output")
        os.replace(staged_output, output_file)
        return result_codec
    finally:
        delete_files([concat_list_file, staged_output])


def _sanitize_image_file(image_path: str) -> str:
    # Some local images open in Pillow but throw inside ImageClip because of broken EXIF/eXIf metadata.
    # Re-export a "clean image" here and strip the bad metadata.
    # Bound the intermediate basename while retaining complete source identity.
    # Different extensions must not overwrite pixels, and valid long filenames
    # must not exceed the filesystem's component limit after adding a suffix.
    source_identity = hashlib.sha256(os.fsencode(os.path.abspath(image_path))).hexdigest()
    sanitized_path = os.path.join(os.path.dirname(image_path), f"{source_identity}.sanitized.png")

    temp_path = ""
    try:
        with Image.open(image_path) as image:
            with ImageOps.exif_transpose(image) as upright:
                upright.load()
                # Strip metadata after applying the camera's orientation.
                # Palette transparency belongs to pixels, not removable metadata.
                mode = "RGBA" if "A" in upright.getbands() or "transparency" in upright.info else "RGB"
                cleaned_image = upright.convert(mode)
                cleaned_image.info.clear()
                descriptor, temp_path = tempfile.mkstemp(
                    prefix=".image-sanitize-", suffix=".png",
                    dir=os.path.dirname(os.path.abspath(sanitized_path)),
                )
                os.close(descriptor)
                cleaned_image.save(temp_path)
                cleaned_image.close()
        os.replace(temp_path, sanitized_path)
        temp_path = ""
    finally:
        if temp_path:
            delete_files(temp_path)

    return sanitized_path


def _open_image_clip_with_fallback(image_path: str):
    # MoviePy does not apply camera EXIF orientation while decoding an image.
    # CMYK JPEG channels also must become RGB rather than an apparent alpha mask.
    # Ordinary RGB inputs retain the direct path.
    try:
        with Image.open(image_path) as image:
            orientation = image.getexif().get(274, 1)
            image_mode = image.mode
    except Exception:
        orientation = 1
        image_mode = None
    if orientation in range(2, 9) or image_mode == "CMYK":
        sanitized_path = _sanitize_image_file(image_path)
        return ImageClip(sanitized_path), sanitized_path

    # Try to open the original image first. If broken metadata makes that fail, try a metadata-free copy.
    try:
        return ImageClip(image_path), image_path
    except Exception as exc:
        logger.warning(
            f"failed to open image directly, trying sanitized copy: {image_path}, error: {str(exc)}"
        )
        sanitized_path = _sanitize_image_file(image_path)
        return ImageClip(sanitized_path), sanitized_path


_moviepy_reader_open_lock = threading.Lock()


def _open_video_clip_quietly(video_path: str, audio: bool = False) -> VideoFileClip:
    """
    Open a video file quietly, so MoviePy 2.1.x does not print ffmpeg probe info straight to stdout.

    Background:
    This dependency version of `FFMPEG_VideoReader` contains `print(self.infos)` and
    `print(ffmpeg command)`. Reading an intermediate video with no audio track prints
    `audio_found: False`. That is only input-material metadata. It does not mean the finished video has no audio,
    but it can make WebUI or terminal users think generation failed.

    Implementation:
    1. Redirect stdout only for the short window while VideoFileClip is opened;
    2. Default `audio=False`, because the material stage does not need the original soundtrack,
       and the final audio is attached later in `generate_video()`;
    3. If the dependency does print something, downgrade it to a debug log so it can be inspected when needed.
    """
    captured_stdout = io.StringIO()
    # redirect_stdout changes process-wide state. Overlapping reader opens can
    # restore each other's capture buffers instead of the original stdout.
    # Serialize this short construction window; clip processing stays parallel.
    with _moviepy_reader_open_lock:
        with redirect_stdout(captured_stdout):
            clip = VideoFileClip(video_path, audio=audio)

    moviepy_stdout = captured_stdout.getvalue().strip()
    if moviepy_stdout:
        logger.debug(
            "suppressed MoviePy video reader stdout for "
            f"{video_path}, chars: {len(moviepy_stdout)}"
        )

    return clip


def close_clip(clip):
    if clip is None:
        return
        
    try:
        # close main resources
        if hasattr(clip, 'reader') and clip.reader is not None:
            clip.reader.close()
            
        # close audio resources
        if hasattr(clip, 'audio') and clip.audio is not None:
            if hasattr(clip.audio, 'reader') and clip.audio.reader is not None:
                clip.audio.reader.close()
            del clip.audio
            
        # close mask resources
        if hasattr(clip, 'mask') and clip.mask is not None:
            if hasattr(clip.mask, 'reader') and clip.mask.reader is not None:
                clip.mask.reader.close()
            del clip.mask
            
        # handle child clips in composite clips
        if hasattr(clip, 'clips') and clip.clips:
            for child_clip in clip.clips:
                if child_clip is not clip:  # avoid possible circular references
                    close_clip(child_clip)
            
        # clear clip list
        if hasattr(clip, 'clips'):
            clip.clips = []
            
    except Exception as e:
        logger.error(f"failed to close clip: {str(e)}")
    
    del clip
    gc.collect()

def delete_files(files: List[str] | str):
    if isinstance(files, str):
        files = [files]

    # When a video is looped to fill the duration, the same temporary segment path appears more than once in the FFmpeg concat list.
    # The concat must keep the duplicates, but cleanup can delete the file only once. Deduplicate in the original order so every
    # caller is idempotent, and so a successful first delete is not followed by a FileNotFoundError.
    unique_files = dict.fromkeys(file for file in files if file)
    for file in unique_files:
        try:
            os.remove(file)
        except FileNotFoundError:
            # Cleanup may find the file already gone, for example after an FFmpeg failure path or a concurrent cleanup
            # already removed it. That is not something the user needs to handle, and it should not clutter the generation log.
            continue
        except OSError as e:
            # Permissions, a read-only filesystem, or a disk error can leave a real temp file. Keep the warning
            # so the path and the system error can be used to locate the environment problem.
            logger.warning(f"failed to delete temporary file {file}: {str(e)}")


def get_bgm_file(bgm_type: str = "random", bgm_file: str = ""):
    if not bgm_type:
        return ""

    if bgm_file:
        try:
            resolved_bgm_file = bgm_service.resolve_bgm_file(bgm_file)
        except ValueError as exc:
            # bgm_file in an API request comes from user input. Resolve it only inside the user BGM directory or the built-in
            # song directory, so MoviePy cannot read arbitrary server files such as config or secrets.
            logger.warning(
                f"reject unsafe bgm file: {bgm_file}, error: {str(exc)}"
            )
            return ""
        return resolved_bgm_file

    if bgm_type == "random":
        files = bgm_service.list_bgm_files()
        # When the background-music directory is empty, fall back to "no BGM" instead of letting random.choice([]) raise.
        if not files:
            logger.warning("no background music files found")
            return ""
        return random.choice(files)

    return ""


def _fit_clip_to_canvas(
    clip,
    *,
    target_width: int,
    target_height: int,
    fit_mode: VideoFitMode | str = VideoFitMode.cover,
):
    """Resize a clip to an exact canvas using cover/crop or contain/letterbox."""
    source_width, source_height = (int(value) for value in clip.size)
    target_width = int(target_width)
    target_height = int(target_height)
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError(
            "video dimensions must be positive: "
            f"source={source_width}x{source_height}, "
            f"target={target_width}x{target_height}"
        )

    mode = VideoFitMode(fit_mode)
    if (source_width, source_height) == (target_width, target_height):
        return clip

    # Exact aspect-ratio matches do not need either a crop or a background.
    if source_width * target_height == source_height * target_width:
        return clip.resized(new_size=(target_width, target_height))

    width_scale = target_width / source_width
    height_scale = target_height / source_height

    if mode == VideoFitMode.cover:
        # ceil guarantees the resized clip covers the complete canvas despite
        # floating-point rounding. Any excess is removed symmetrically.
        scale_factor = max(width_scale, height_scale)
        resized_width = max(target_width, math.ceil(source_width * scale_factor))
        resized_height = max(target_height, math.ceil(source_height * scale_factor))
        resized_clip = clip.resized(new_size=(resized_width, resized_height))
        crop_x = max(0, (resized_width - target_width) // 2)
        crop_y = max(0, (resized_height - target_height) // 2)
        return resized_clip.cropped(
            x1=crop_x,
            y1=crop_y,
            width=target_width,
            height=target_height,
        )

    # contain preserves the legacy behavior: show the complete source frame,
    # centered over a black canvas when the aspect ratios differ.
    scale_factor = min(width_scale, height_scale)
    resized_width = max(1, min(target_width, int(source_width * scale_factor)))
    resized_height = max(1, min(target_height, int(source_height * scale_factor)))
    background = ColorClip(
        size=(target_width, target_height), color=(0, 0, 0)
    ).with_duration(clip.duration)
    resized_clip = clip.resized(
        new_size=(resized_width, resized_height)
    ).with_position("center")
    return CompositeVideoClip(
        [background, resized_clip], size=(target_width, target_height)
    ).with_duration(clip.duration)


def combine_videos(
    combined_video_path: str,
    video_paths: List[str],
    audio_file: str,
    video_aspect: VideoAspect = VideoAspect.portrait,
    video_concat_mode: VideoConcatMode = VideoConcatMode.random,
    video_transition_mode: VideoTransitionMode = None,
    max_clip_duration: int = 5,
    threads: int = 2,
    clip_speed: float = 1.0,
    video_fit_mode: VideoFitMode = VideoFitMode.cover,
    source_usage: dict[str, int] | None = None,
    source_groups: dict[str, str] | None = None,
    used_video_paths: List[str] | None = None,
    progress_callback: Callable[[float], None] | None = None,
    allow_loop: bool = True,
) -> str:
    # Pydantic leaves this optional default as the string "random" unless
    # explicitly provided. Normalize it once for all downstream enum access.
    video_concat_mode = VideoConcatMode(video_concat_mode or VideoConcatMode.random)
    audio_clip = AudioFileClip(audio_file)
    try:
        # Only the narration duration is needed here, to decide how long to concatenate material video. audio_clip is not used again
        # after that. Close it immediately so an early return or an error path does not leak the file handle.
        audio_duration = audio_clip.duration
    finally:
        close_clip(audio_clip)
    logger.info(f"audio duration: {audio_duration} seconds")
    logger.info(f"maximum clip duration: {max_clip_duration} seconds")
    required_video_duration = _get_required_video_duration(audio_duration)
    logger.info(
        f"required video duration: {required_video_duration:.2f} seconds "
        f"(audio duration + {_VIDEO_DURATION_SAFETY_MARGIN:.2f}s safety margin)"
    )

    # Stay compatible with a direct API call that omitted the transition mode, so a later .value access does not crash.
    transition_value = getattr(video_transition_mode, "value", video_transition_mode)
    normalized_clip_speed = utils.normalize_clip_speed(clip_speed)
    if normalized_clip_speed != 1.0:
        # Log the effective value only once. That makes it easier to see when an out-of-range API parameter was normalized,
        # and it avoids repeating the same log on the per-segment hot path.
        logger.info(f"clip playback speed: {normalized_clip_speed:.2f}x")
    # max_clip_duration limits the final playback duration in the finished video, not how long the source is read.
    # Playing 1.5 seconds of source at 0.5x in MoviePy yields a 3-second clip, and playing
    # 6 seconds of source at 2x also yields a 3-second clip. Before slicing, convert back to source duration from the speed. If
    # the code still reads a fixed 3 seconds, slows it down, and trims it, the next segment starts at source second 3 and skips the middle
    # 1.5 seconds. This calculation also keeps the source timeline continuous and non-overlapping at every speed.
    source_clip_duration = max_clip_duration * normalized_clip_speed
    output_dir = os.path.dirname(combined_video_path)

    aspect = VideoAspect(video_aspect)
    fit_mode = VideoFitMode(video_fit_mode)
    video_width, video_height = aspect.to_resolution()

    processed_clips = []
    subclipped_items = []
    video_duration = 0
    for video_path in video_paths:
        clip = None
        try:
            clip = _open_video_clip_quietly(video_path)
            clip_duration = float(clip.duration)
            clip_w, clip_h = clip.size
            if (
                not math.isfinite(clip_duration)
                or clip_duration <= 0
                or not all(
                    math.isfinite(float(dimension)) and float(dimension) > 0
                    for dimension in (clip_w, clip_h)
                )
            ):
                raise ValueError("invalid video duration or dimensions")
        except Exception as exc:
            logger.warning(
                f"skipping unreadable video source: path={video_path}, error={exc}"
            )
            continue
        finally:
            close_clip(clip)
        
        start_time = 0

        while start_time < clip_duration:
            end_time = min(start_time + source_clip_duration, clip_duration)

            # Keep every valid segment.
            # That neither drops material whose whole video is already shorter than max_clip_duration
            # nor swallows the short tail left at the end of a long video.
            if end_time > start_time:
                subclipped_items.append(
                    SubClippedVideoClip(
                        file_path=video_path,
                        start_time=start_time,
                        end_time=end_time,
                        width=clip_w,
                        height=clip_h,
                        source_file_path=video_path,
                    )
                )

            start_time = end_time
            if video_concat_mode.value == VideoConcatMode.sequential.value:
                break

    subclipped_items = _prioritize_unique_source_clips(
        subclipped_items=subclipped_items,
        concat_mode=video_concat_mode,
        **({"source_usage": source_usage, "source_groups": source_groups}
           if source_usage is not None else {}),
    )
        
    logger.debug(f"total subclipped items: {len(subclipped_items)}")
    
    # Add downloaded clips over and over until the duration of the audio (max_duration) has been reached
    def process_one_clip(indexed_item):
        """Trim one source segment, change its speed, apply transitions, and write it out. Return None on failure."""
        index, subclipped_item = indexed_item
        source_clip = None
        clip = None
        clip_file = None
        try:
            logger.debug(
                f"processing clip {index + 1}: {subclipped_item.width}x{subclipped_item.height}, "
                f"source: {os.path.basename(subclipped_item.source_file_path)}"
            )
            source_clip = _open_video_clip_quietly(subclipped_item.file_path)
            clip = source_clip.subclipped(
                subclipped_item.start_time, subclipped_item.end_time
            )
            # Playback speed is a property of the material and should be applied before transitions. A one-second Fade/Slide
            # then does not become 0.5 or 2 seconds just because the material speed changed. The later max-duration trim remains a
            # safety net for floating-point error or an unusual material duration, so the final segment stays within the configured limit.
            if normalized_clip_speed != 1.0:
                clip = clip.with_speed_scaled(normalized_clip_speed)
            # Normalize every source clip before transitions are applied. In cover mode
            # the clip fills the canvas and the excess edges are cropped; contain keeps
            # the complete source frame and uses black bars for the unused area.
            clip_w, clip_h = clip.size
            if clip_w != video_width or clip_h != video_height:
                clip_ratio = clip.w / clip.h
                video_ratio = video_width / video_height
                logger.debug(
                    "resizing clip, "
                    f"source: {clip_w}x{clip_h}, ratio: {clip_ratio:.2f}, "
                    f"target: {video_width}x{video_height}, ratio: {video_ratio:.2f}, "
                    f"fit_mode: {fit_mode.value}"
                )
                clip = _fit_clip_to_canvas(
                    clip,
                    target_width=video_width,
                    target_height=video_height,
                    fit_mode=fit_mode,
                )

            shuffle_side = random.choice(["left", "right", "top", "bottom"])
            # Short footage and faster playback can leave less than one second.
            # Keep the chosen transition inside the rendered clip's timeline.
            transition_duration = min(1.0, clip.duration)
            if transition_value == VideoTransitionMode.fade_in.value:
                clip = video_effects.fadein_transition(clip, transition_duration)
            elif transition_value == VideoTransitionMode.fade_out.value:
                clip = video_effects.fadeout_transition(clip, transition_duration)
            elif transition_value == VideoTransitionMode.slide_in.value:
                clip = video_effects.slidein_transition(clip, transition_duration, shuffle_side)
            elif transition_value == VideoTransitionMode.slide_out.value:
                clip = video_effects.slideout_transition(clip, transition_duration, shuffle_side)
            elif transition_value == VideoTransitionMode.zoom_in.value:
                clip = video_effects.zoomin_transition(clip, 1)
            elif transition_value == VideoTransitionMode.zoom_out.value:
                clip = video_effects.zoomout_transition(clip, 1)
            elif transition_value == VideoTransitionMode.shuffle.value:
                transition_funcs = [
                    lambda c: video_effects.fadein_transition(c, transition_duration),
                    lambda c: video_effects.fadeout_transition(c, transition_duration),
                    lambda c: video_effects.slidein_transition(c, transition_duration, shuffle_side),
                    lambda c: video_effects.slideout_transition(c, transition_duration, shuffle_side),
                    lambda c: video_effects.zoomin_transition(c, 1),
                    lambda c: video_effects.zoomout_transition(c, 1),
                ]
                shuffle_transition = random.choice(transition_funcs)
                clip = shuffle_transition(clip)

            if clip.duration > max_clip_duration:
                clip = clip.subclipped(0, max_clip_duration)

            # Write each candidate clip to a unique temporary file. Threads must not
            # share the same output path.
            # Distinct combinations can share a task/output directory. Reserve
            # an owned path so their encoders and cleanup never share a clip.
            with tempfile.NamedTemporaryFile(
                dir=output_dir or ".", prefix="temp-clip-", suffix=".mp4", delete=False
            ) as temporary_clip:
                clip_file = temporary_clip.name
            _write_videofile_with_codec_fallback(
                clip,
                clip_file,
                codec=_get_configured_video_codec(),
                logger=None,
                fps=fps,
            )

            clip_duration_saved = clip.duration
            processed_clip = SubClippedVideoClip(
                file_path=clip_file,
                duration=clip_duration_saved,
                width=clip_w,
                height=clip_h,
                source_file_path=subclipped_item.source_file_path,
            )
            clip_file = None
            return processed_clip
        except Exception as exc:
            logger.error(f"failed to process clip: {str(exc)}")
            return None
        finally:
            # The derived clip shares its FFmpeg reader with the source. If
            # subclipping itself failed, close the original source instead.
            close_clip(clip if clip is not None else source_clip)
            # MoviePy may leave a truncated MP4 even when encoding raises. It
            # was never returned, so concat cleanup cannot see it.
            # Close the reader first so Windows can remove the partial file.
            if clip_file:
                delete_files(clip_file)

    # Segments are always processed in a thread pool. Per-segment logs are the only progress for this stage. Bind them to
    # the thread that started compositing so the WebUI task log can collect them.
    process_clip_in_task_scope = logging_utils.bind_log_scope(process_one_clip)
    clip_processing_workers = 1
    if len(subclipped_items) >= 2:
        clip_processing_workers = min(_get_clip_processing_concurrency(), len(subclipped_items))
    with ThreadPoolExecutor(
        max_workers=clip_processing_workers,
        thread_name_prefix="clip-process",
    ) as executor:
        next_candidate_index = 0
        while (
            next_candidate_index < len(subclipped_items)
            and video_duration < required_video_duration
        ):
            remaining_duration = required_video_duration - video_duration
            batch = []
            batch_duration = 0.0
            candidate_index = next_candidate_index
            while candidate_index < len(subclipped_items) and batch_duration < remaining_duration:
                subclipped_item = subclipped_items[candidate_index]
                source_duration = subclipped_item.end_time - subclipped_item.start_time
                output_duration = min(
                    max_clip_duration,
                    source_duration / normalized_clip_speed,
                )
                leftover = remaining_duration - batch_duration
                if not allow_loop and output_duration > leftover:
                    trimmed_source = max(leftover, 0.05) * normalized_clip_speed
                    subclipped_item = SubClippedVideoClip(
                        file_path=subclipped_item.file_path,
                        start_time=subclipped_item.start_time,
                        end_time=min(subclipped_item.end_time, subclipped_item.start_time + trimmed_source),
                        width=subclipped_item.width,
                        height=subclipped_item.height,
                        source_file_path=subclipped_item.source_file_path,
                    )
                    output_duration = leftover
                batch.append((candidate_index, subclipped_item))
                batch_duration += output_duration
                candidate_index += 1
                if not allow_loop and batch_duration >= remaining_duration - 1e-3:
                    break
            if not batch:
                break
            for processed_clip in executor.map(process_clip_in_task_scope, batch):
                if processed_clip is None:
                    continue
                processed_clips.append(processed_clip)
                video_duration += processed_clip.duration
                # Each 4K segment takes more than ten seconds. Report covered duration per segment so the log and the progress bar
                # show that this stage is still moving.
                logger.info(
                    f"processed clip {len(processed_clips)}: "
                    f"{video_duration:.1f} of {required_video_duration:.1f}s covered"
                )
                _report_clip_progress(
                    progress_callback, video_duration, required_video_duration
                )
            next_candidate_index = candidate_index
    
    # Stock footage may repeat to fill the narration. AI footage must not.
    if video_duration < required_video_duration - 0.05 and not allow_loop:
        raise RuntimeError(
            f"AI footage covers {video_duration:.2f}s of the required {required_video_duration:.2f}s. "
            "Clips were not repeated."
        )
    if allow_loop and video_duration < required_video_duration:
        logger.warning(
            f"video duration ({video_duration:.2f}s) is shorter than required duration "
            f"({required_video_duration:.2f}s), looping clips to match audio length."
        )
        base_clips = processed_clips.copy()
        for clip in itertools.cycle(base_clips):
            if video_duration >= required_video_duration:
                break
            processed_clips.append(clip)
            video_duration += clip.duration
        logger.info(
            f"video duration: {video_duration:.2f}s, audio duration: {audio_duration:.2f}s, "
            f"required duration: {required_video_duration:.2f}s, "
            f"looped {len(processed_clips)-len(base_clips)} clips"
        )
     
    # merge video clips progressively, avoid loading all videos at once to avoid memory overflow
    logger.info("starting clip merging process")
    if not processed_clips:
        if video_paths:
            raise RuntimeError("no readable video clips available for merging")
        logger.warning("no clips available for merging")
        return combined_video_path
    
    clip_files = [clip.file_path for clip in processed_clips]
    logger.info(f"concatenating {len(clip_files)} clips with ffmpeg")
    try:
        concat_video_clips_with_ffmpeg(
            clip_files=clip_files,
            output_file=combined_video_path,
            threads=threads,
            output_dir=output_dir,
            max_duration=audio_duration,
        )
        if used_video_paths is not None:
            # Exclude safety-margin clips that FFmpeg trims entirely from the output.
            elapsed = 0.0
            for clip in processed_clips:
                if elapsed >= audio_duration:
                    break
                used_video_paths.append(clip.source_file_path)
                elapsed += clip.duration
    finally:
        # FFmpeg failures and timeouts must not strand one encoded MP4 per clip.
        # Repeated clips share a path; delete_files already deduplicates them.
        delete_files(clip_files)
            
    logger.info("video combining completed")
    return combined_video_path


def wrap_text(text, max_width, font="Arial", fontsize=60):
    # Subtitle wrapping must happen before the TextClip is created, or MoviePy measures the render area from the raw text only.
    # Measure the width with PIL using the current font and size so each line stays
    # inside the usable video width, and a large size or a long Chinese sentence does not overflow the frame.
    if "\n" in text:
        # Hard breaks in SRT text are separate layout lines. Measuring a token
        # across a newline makes Pillow count both lines as one wide string,
        # then character wrapping can split an otherwise fitting word.
        wrapped_lines = [
            wrap_text(line, max_width, font=font, fontsize=fontsize)
            for line in text.split("\n")
        ]
        return "\n".join(line for line, _ in wrapped_lines), sum(
            height for _, height in wrapped_lines
        )

    font = ImageFont.truetype(font, fontsize)
    max_width = int(max_width)

    # getbbox() returns "the visible ink height of the current glyphs", not the font line height. English that contains only
    # characters without descenders, such as A, m, and n, is missing the descent, and that error accumulates on every line.
    # The last line of the TextClip is then clipped by the canvas. ascent + descent comes from the font itself,
    # does not depend on the language or the character combination, and matches MoviePy's baseline drawing model.
    ascent, descent = font.getmetrics()
    line_height = int(ascent + descent)
    if line_height <= 0:
        # A normal TrueType/OpenType font does not reach this branch. Keep a diagnostic log and a font-size fallback
        # so a broken or unusual font with bad metrics does not produce a zero-height subtitle.
        logger.warning(
            "invalid subtitle font metrics, fallback to font size: "
            f"ascent={ascent}, descent={descent}, fontsize={fontsize}"
        )
        line_height = max(1, int(fontsize))

    def get_text_size(inner_text):
        inner_text = inner_text.strip()
        if not inner_text:
            return 0, line_height
        left, top, right, bottom = font.getbbox(inner_text)
        # bbox is still the right way to measure the real width used for wrapping. Height must always use the stable font line height.
        return right - left, line_height

    width, height = get_text_size(text)
    if width <= max_width:
        # An SRT entry may contain manual line breaks. Even when the whole text does not need to wrap again for width,
        # the canvas height must still be based on the existing line count, or the second line and later lines are clipped.
        return text, (text.count("\n") + 1) * line_height

    def split_long_token(token):
        # When a token is itself too wide (common for a long Chinese sentence with no spaces, or a very long English word),
        # fall back to splitting by character. The important part: when candidate is too wide, commit the previous
        # still-valid current first, then put the current character on the next line. Do not push the too-wide character back onto the previous line.
        lines = []
        current = ""
        for char in token:
            candidate = f"{current}{char}"
            candidate_width, _ = get_text_size(candidate)
            if candidate_width <= max_width or not current:
                current = candidate
                continue
            lines.append(current)
            current = char
        if current:
            lines.append(current)
        return lines

    lines = []
    current = ""
    words = text.split(" ")
    for word in words:
        candidate = f"{current} {word}".strip() if current else word
        candidate_width, _ = get_text_size(candidate)
        if candidate_width <= max_width:
            current = candidate
            continue

        if current:
            lines.append(current)

        word_width, _ = get_text_size(word)
        if word_width <= max_width:
            current = word
        else:
            lines.extend(split_long_token(word))
            current = ""

    if current:
        lines.append(current)

    line_start_punctuation = "，。！？；：、,.!?;:)]}）】》」』”’"
    for index in range(1, len(lines)):
        # When a long Chinese sentence is split by character, a closing mark such as a final period or comma can land alone
        # on the next line. That stretches the subtitle background and looks like a small dot sitting
        # under the text. Without redesigning the wrap algorithm, move the last character of the previous line
        # in front of the punctuation line so the mark stays with the text. This covers common closing punctuation in Chinese and English.
        if not lines[index] or lines[index][0] not in line_start_punctuation:
            continue
        if len(lines[index - 1]) <= 1:
            continue

        candidate = f"{lines[index - 1][-1]}{lines[index]}"
        candidate_width, _ = get_text_size(candidate)
        if candidate_width <= max_width:
            lines[index] = candidate
            lines[index - 1] = lines[index - 1][:-1]

    result = "\n".join(line.strip() for line in lines if line.strip()).strip()
    # Height follows the final result. An explicit line break in the original text may stay inside a token,
    # and then the temporary lines list is not the same length as the number of lines MoviePy actually renders.
    height = (result.count("\n") + 1) * line_height
    return result, height


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    # Match the Pillow color syntax accepted by subtitle preflight.
    # The background clip sets opacity separately from the color value.
    if isinstance(color, str):
        try:
            return ImageColor.getcolor(color, "RGB")
        except (ValueError, TypeError):
            pass
    return (0, 0, 0)


def _rounded_subtitle_background_clip(
    width: int,
    height: int,
    color: str,
    alpha: int = 140,
    radius: int = 16,
) -> ImageClip:
    # The new subtitle background is used only when the user turns it on: draw a rounded translucent plate as an RGBA image,
    # then give it to MoviePy as a transparent ImageClip. The default path stays the same,
    # and a softer subtitle look can be tried at low cost.
    rgb = _hex_to_rgb(color)
    safe_alpha = max(0, min(255, int(alpha)))
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle(
        [0, 0, max(0, width - 1), max(0, height - 1)],
        radius=max(0, int(radius)),
        fill=(rgb[0], rgb[1], rgb[2], safe_alpha),
    )
    return ImageClip(np.array(img), transparent=True)


def _get_visible_center_position(
    text_clip: TextClip,
    container_width: int,
    container_height: int,
) -> tuple[int, int]:
    """
    Place the TextClip in the center of the background container using the text's real visible pixels.

    MoviePy's TextClip builds a transparent canvas from the font line height and the baseline. For many fonts the
    visible glyphs are not at the geometric center of that canvas, so `with_position("center")`
    centers the whole transparent canvas and the subtitle looks too high or too low. Read the TextClip's
    transparent mask and offset from the bbox of pixels that are actually present, so the text the user sees
    is visually centered in the subtitle background.
    """
    x = int(round((container_width - text_clip.w) / 2))
    y = int(round((container_height - text_clip.h) / 2))

    try:
        if text_clip.mask is None:
            return x, y

        mask_frame = text_clip.mask.get_frame(0)
        ys, _ = np.where(mask_frame > 0.01)
        if len(ys) == 0:
            return x, y

        visible_top = int(ys.min())
        visible_bottom = int(ys.max())
        visible_height = visible_bottom - visible_top + 1
        y = int(round((container_height - visible_height) / 2 - visible_top))
    except Exception as exc:
        logger.debug(f"failed to center subtitle text by visible mask: {str(exc)}")

    return x, y


def validate_subtitle_colors(params: VideoParams) -> None:
    """Reject colors Pillow cannot render, before generation services run."""
    if not params.subtitle_enabled:
        return
    colors = [("text_fore_color", params.text_fore_color)]
    # Rendering truncates the width to an integer; a zero-width stroke is unused.
    if int(params.stroke_width) > 0:
        colors.append(("stroke_color", params.stroke_color))
    if isinstance(params.text_background_color, str):
        colors.append(("text_background_color", params.text_background_color))
    for field, value in colors:
        if value is None:
            continue
        try:
            ImageColor.getcolor(value, "RGBA")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"invalid subtitle color: {field}") from exc


def subtitle_colors_are_indistinguishable(params: VideoParams) -> bool:
    """Check whether the subtitle text and background are the same color, and warn that the subtitles may be hard to read."""
    if not params.subtitle_enabled or not params.text_background_color:
        return False

    def normalize_color(value):
        if isinstance(value, bool):
            return "#000000" if value else ""
        return str(value or "").strip().lower()

    text_color = normalize_color(params.text_fore_color)
    background_color = normalize_color(params.text_background_color)
    return bool(text_color and text_color == background_color)


@lru_cache(maxsize=64)
def _subtitle_font_supports_sample(font_path: str, sample: str) -> bool:
    """Check whether the font contains the glyphs the sample text needs, and cache repeated checks."""
    try:
        font = ImageFont.truetype(font_path, 30)
        missing_mask = font.getmask("\U0010ffff")
        missing_signature = (
            missing_mask.size,
            missing_mask.getbbox(),
            bytes(missing_mask),
        )
        for char in sample:
            char_mask = font.getmask(char)
            char_signature = (
                char_mask.size,
                char_mask.getbbox(),
                bytes(char_mask),
            )
            if char_mask.getbbox() is None or char_signature == missing_signature:
                return False
        return True
    except Exception as e:
        # A failed font probe must not stop generation. Keep the log for environment compatibility debugging.
        logger.warning(f"failed to inspect subtitle font glyphs: {font_path}, {e}")
        return True


def subtitle_font_supports_text(font_path: str, text: str) -> bool:
    """Check whether the font can draw letters and digits in the text, ignoring whitespace and punctuation."""
    sample = "".join(
        dict.fromkeys(
            char
            for char in str(text or "")
            if unicodedata.category(char)[0] in {"L", "N"}
        )
    )[:64]
    if not sample:
        return True
    return _subtitle_font_supports_sample(font_path, sample)


def generate_video(
    video_path: str,
    audio_path: str,
    subtitle_path: str,
    output_file: str,
    params: VideoParams,
    bgm_file_override: str | None = None,
) -> bool:
    """
    Composite the final video and return whether background-music handling succeeded this time.

    The return value describes only BGM handling: True when BGM was not requested or was mixed successfully; False when BGM was
    requested but loading, effects, or mixing failed. Even if BGM fails, a video with only
    narration is still written, and the task layer decides whether to show a degraded-mode warning.
    """
    aspect = VideoAspect(params.video_aspect)
    video_width, video_height = aspect.to_resolution()

    logger.info(f"generating video: {video_width} x {video_height}")
    logger.info(f"  ① video: {video_path}")
    logger.info(f"  ② audio: {audio_path}")
    logger.info(f"  ③ subtitle: {subtitle_path}")
    logger.info(f"  ④ output: {output_file}")

    # 
    # PermissionError: [WinError 32] The process cannot access the file because it is being used by another process: 'final-1.mp4.tempTEMP_MPY_wvf_snd.mp3'
    # write into the same directory as the output file
    output_dir = os.path.dirname(output_file)

    font_path = ""
    if params.subtitle_enabled:
        if not params.font_name:
            params.font_name = "STHeitiMedium.ttc"
        # The API entry prechecks this, but the WebUI, CLI, and internal calls can still reach the render layer directly.
        # Always check the real path so the font stays inside resource/fonts, and block absolute paths,
        # ../ traversal, and symlinks that point outside the directory, before handing the file to PIL/MoviePy.
        font_path = file_security.resolve_path_within_directory(
            utils.font_dir(), params.font_name
        )
        if os.name == "nt":
            font_path = font_path.replace("\\", "/")

        logger.info(f"  ⑤ font: {font_path}")

    def resolve_subtitle_background_color():
        # Historical parameter: in the API, `text_background_color` may be a boolean
        # or an actual color string. Normalize it here so True/False is not
        # passed straight to TextClip and rendered in an unexpected way.
        if isinstance(params.text_background_color, bool):
            return "#000000" if params.text_background_color else None
        return params.text_background_color

    def create_text_clip(subtitle_item):
        params.font_size = int(params.font_size)
        params.stroke_width = int(params.stroke_width)
        phrase = subtitle_item[1]
        max_width = video_width * 0.9
        bg_color = resolve_subtitle_background_color()
        rounded_bg_enabled = bool(
            getattr(params, "rounded_subtitle_background", False) and bg_color
        )
        has_subtitle_background = bool(bg_color)
        # A rounded background is sized to the real text width, so the left and right padding should be tighter. The old rectangular background keeps
        # a larger safety margin so long subtitles from older configs do not sit on the edge or get clipped.
        padding_ratio = 0.4 if rounded_bg_enabled else 0.6
        pad_x = int(params.font_size * padding_ratio) if has_subtitle_background else 0
        # A subtitle background needs clear horizontal padding around the text. Subtract
        # the padding from the usable width before wrapping, so long English or a large font that fills 90% of the video width
        # does not sit on the background edge and look clipped. Both the plain rectangle and the rounded background
        # use this path. Subtitles with no background keep the original maximum width.
        text_max_width = max(1, int(max_width) - 2 * pad_x)
        wrapped_txt, txt_height = wrap_text(
            phrase,
            max_width=text_max_width,
            font=font_path,
            fontsize=params.font_size,
        )
        interline = int(params.font_size * 0.25)
        line_count = wrapped_txt.count("\n") + 1
        vertical_padding = int(params.font_size * 0.35)
        # Pillow/MoviePy expand the stroke above and below the glyphs and include that in each line's
        # advance height. Adding stroke padding only once around the whole subtitle block still lets a thick stroke on multiple lines
        # accumulate error line by line. Count stroke space on both sides for the actual line count. A thin default stroke only
        # adds a little height, and "small size + thick stroke + multiple lines" still fits.
        stroke_padding = int(params.stroke_width * 2 * line_count)
        text_clip_margin_y = max(
            int(params.font_size * 0.3), int(params.stroke_width * 2)
        )
        # With `method=label`, MoviePy shrinks the text box height automatically. With multiple subtitle lines,
        # a stroke, or a background color, the lower half of the last line is easy to clip. Pass
        # a more conservative height here and include line spacing plus extra vertical padding, so the subtitle
        # background box and the text itself both render in full.
        clip_h = int(
            txt_height
            + vertical_padding
            + (interline * line_count)
            + stroke_padding
        )

        if rounded_bg_enabled:
            # A rounded background should fit the text width instead of using 90% of the video width. First measure
            # the longest line with PIL, then add horizontal padding, so a short subtitle does not get an overly wide plate.
            try:
                font = ImageFont.truetype(font_path, params.font_size)
                text_w = max(
                    int(font.getbbox(line)[2] - font.getbbox(line)[0])
                    for line in wrapped_txt.split("\n")
                )
            except Exception as exc:
                logger.warning(
                    f"failed to measure subtitle text width, fallback to max width: {str(exc)}"
                )
                text_w = int(max_width)

            box_w = max(1, min(int(max_width), text_w + 2 * pad_x))
            radius = max(8, int(params.font_size * 0.4))
            text_clip = TextClip(
                text=wrapped_txt,
                font=font_path,
                font_size=params.font_size,
                color=params.text_fore_color,
                bg_color=None,
                stroke_color=params.stroke_color,
                stroke_width=params.stroke_width,
                interline=interline,
                size=(box_w, None),
                text_align="center",
                margin=(0, text_clip_margin_y),
            )
            clip_h = max(clip_h, text_clip.h)
            bg_clip = _rounded_subtitle_background_clip(
                width=box_w,
                height=clip_h,
                color=bg_color,
                alpha=140,
                radius=radius,
            )
            text_position = _get_visible_center_position(text_clip, box_w, clip_h)
            _clip = CompositeVideoClip(
                [bg_clip, text_clip.with_position(text_position)],
                size=(box_w, clip_h),
            )
        elif bg_color:
            size = (
                int(max_width),
                clip_h,
            )
            text_clip = TextClip(
                text=wrapped_txt,
                font=font_path,
                font_size=params.font_size,
                color=params.text_fore_color,
                bg_color=None,
                stroke_color=params.stroke_color,
                stroke_width=params.stroke_width,
                interline=interline,
                size=(int(max_width), None),
                text_align="center",
                margin=(0, text_clip_margin_y),
            )
            size = (size[0], max(size[1], text_clip.h))
            bg_clip = _rounded_subtitle_background_clip(
                width=size[0],
                height=size[1],
                color=bg_color,
                alpha=255,
                radius=0,
            )
            text_position = _get_visible_center_position(text_clip, size[0], size[1])
            _clip = CompositeVideoClip(
                [bg_clip, text_clip.with_position(text_position)],
                size=size,
            )
        else:
            size = (
                int(max_width),
                clip_h,
            )
            _clip = TextClip(
                text=wrapped_txt,
                font=font_path,
                font_size=params.font_size,
                color=params.text_fore_color,
                bg_color=None,
                stroke_color=params.stroke_color,
                stroke_width=params.stroke_width,
                interline=interline,
                size=size,
                text_align="center",
            )
        duration = subtitle_item[0][1] - subtitle_item[0][0]
        _clip = _clip.with_start(subtitle_item[0][0])
        _clip = _clip.with_end(subtitle_item[0][1])
        _clip = _clip.with_duration(duration)

        # The bounce animation is enabled only when the user explicitly selects it. The default none keeps the original subtitle render path.
        anim_type = getattr(params, "subtitle_animation", "none")
        if anim_type in ("pop_spring", "spring", "pop"):
            _clip = _apply_subtitle_spring_animation(_clip, duration)

        if params.subtitle_position == "bottom":
            _clip = _clip.with_position(("center", video_height * 0.95 - _clip.h))
        elif params.subtitle_position == "top":
            _clip = _clip.with_position(("center", video_height * 0.05))
        elif params.subtitle_position in ("two_thirds_bottom", "two_thirds", "2/3_bottom"):
            # 2/3 from the bottom = 1/3 from the top: y = (video_height - _clip.h) * (1/3)
            y_two_thirds = (video_height - _clip.h) / 3.0
            _clip = _clip.with_position(("center", y_two_thirds))
        elif params.subtitle_position == "custom":
            # Ensure the subtitle is fully within the screen bounds
            margin = 10  # Additional margin, in pixels
            max_y = video_height - _clip.h - margin
            min_y = margin
            custom_y = (video_height - _clip.h) * (params.custom_position / 100)
            custom_y = max(
                min_y, min(custom_y, max_y)
            )  # Constrain the y value within the valid range
            _clip = _clip.with_position(("center", custom_y))
        else:  # center
            _clip = _clip.with_position(("center", "center"))
        return _clip

    # MoviePy's CompositeAudioClip.close() does not close child AudioFileClip objects. An
    # ExitStack holds every original file reader explicitly, so success, a subtitle error, a mix failure, and
    # a video-write failure all release the FFmpeg subprocess, especially so Windows does not keep the file locked.
    with ExitStack() as clip_stack:
        source_video_clip = clip_stack.enter_context(
            _open_video_clip_quietly(video_path)
        )
        voice_source_clip = clip_stack.enter_context(AudioFileClip(audio_path))
        video_clip = source_video_clip
        audio_clip = voice_source_clip.with_effects(
            [afx.MultiplyVolume(1.0 if params.voice_volume is None else params.voice_volume)]
        )

        def make_textclip(text):
            return TextClip(
                text=text,
                font=font_path,
                font_size=params.font_size,
            )

        if params.subtitle_enabled and subtitle_path and os.path.exists(subtitle_path):
            sub = clip_stack.enter_context(
                SubtitlesClip(
                    subtitles=moviepy_subtitles(subtitle_path),
                    encoding="utf-8",
                    make_textclip=make_textclip,
                )
            )
            text_clips = []
            for item in sub.subtitles:
                clip = create_text_clip(subtitle_item=item)
                text_clips.append(clip)
            video_clip = CompositeVideoClip([video_clip, *text_clips])
            clip_stack.callback(video_clip.close)

        bgm_enabled = bgm_service.should_use_bgm(
            params.bgm_type, params.bgm_volume
        )
        if not bgm_enabled and params.bgm_type:
            # Every BGM source shares this short-circuit. When volume is not greater than 0, do not resolve a random or
            # custom file, and do not load a file returned by a provider. That avoids pointless IO and mixing.
            logger.info(
                f"skipping background music because volume is not positive: "
                f"type={params.bgm_type}, volume={params.bgm_volume}"
            )

        # Provider music can be passed in directly by the task layer as the matching file. None means keep the random/custom
        # BGM lookup. An empty string explicitly disables this BGM. Every source still has to pass the shared volume rule first.
        bgm_file = ""
        if bgm_enabled:
            bgm_file = (
                bgm_file_override
                if bgm_file_override is not None
                else get_bgm_file(
                    bgm_type=params.bgm_type,
                    bgm_file=params.bgm_file,
                )
            )
        bgm_mix_succeeded = True
        if bgm_file:
            try:
                bgm_effects = [afx.MultiplyVolume(params.bgm_volume)]
                # Random or custom music resolved inside the service may be shorter than the finished video and needs to loop. A file passed
                # in through override means the provider already matched the duration. Decide whether to loop from
                # the file source, so each new provider does not require another name allowlist change.
                if bgm_file_override is None:
                    bgm_effects.append(afx.AudioLoop(duration=video_clip.duration))
                # Fade the complete playback timeline, not each source loop.
                bgm_effects.append(afx.AudioFadeOut(3))
                bgm_source_clip = clip_stack.enter_context(AudioFileClip(bgm_file))
                bgm_clip = bgm_source_clip.with_effects(bgm_effects)
                audio_clip = CompositeAudioClip([audio_clip, bgm_clip])
            except Exception:
                bgm_mix_succeeded = False
                # Log the full stack and stable context so file decoding, MoviePy effects, and
                # CompositeAudioClip failures can be told apart. File contents and API keys are not logged.
                logger.exception(
                    f"failed to mix background music: type={params.bgm_type}, "
                    f"file={bgm_file}"
                )

        final_video_clip = video_clip.with_audio(audio_clip)
        clip_stack.callback(final_video_clip.close)
        # Keep the input audio sample rate explicitly. If it cannot be read, fall back to MoviePy's default of 44100Hz.
        # That reduces another resample, and the quality swing it causes, across environments and especially in Docker.
        output_audio_fps = int(getattr(audio_clip, "fps", 0) or 44100)
        with _stage_heartbeat("final video render"):
            _write_videofile_with_codec_fallback(
                final_video_clip,
                output_file=output_file,
                codec=_get_configured_video_codec(),
                atomic_output=True,
                audio_codec=audio_codec,
                audio_fps=output_audio_fps,
                audio_bitrate=audio_bitrate,
                temp_audiofile_path=_get_temp_audio_dir(output_dir),
                threads=params.n_threads or 2,
                logger=None,
                fps=fps,
            )
        return bgm_mix_succeeded


def render_image_zoom_video(image_path: str, clip_duration: int = 5) -> str:
    """
    Render one local image as an mp4 segment with a slow zoom, and return the output file path.

    Local material preprocessing and OpenAI-compatible text-to-image materials share this "image -> clip" render
    path: the ImageClip plays for a fixed clip_duration and adds about 3% zoom
    per second, so a static frame does not look flat in the finished video. The caller handles render errors
    according to that material source's own failure contract.
    """
    clip, _ = _open_image_clip_with_fallback(image_path)
    clip = clip.with_duration(clip_duration).with_position("center")
    temp_path = ""
    try:
        # Apply a zoom effect using the resize method.
        # A lambda function is used to make the zoom effect dynamic over time.
        # The zoom effect starts from the original size and gradually scales up to 120%.
        # t represents the current time, and clip.duration is the total duration of the clip.
        # Note: 1 represents 100% size, so 1.2 represents 120%.
        zoom_clip = clip.resized(
            lambda t: 1 + (clip_duration * 0.03) * (t / clip.duration)
        )

        # Optionally, create a composite video clip containing the zoomed clip.
        # This is useful if you want to add other elements to the video.
        final_clip = CompositeVideoClip([zoom_clip])
        try:
            # The duration changes the rendered content, so it must be part of
            # the output identity. Different tasks may render the same image
            # concurrently; only publish a complete MP4 after MoviePy closes it.
            source_identity = hashlib.sha256(os.fsencode(os.path.abspath(image_path))).hexdigest()
            duration_identity = hashlib.sha256(str(clip_duration).encode()).hexdigest()[:16]
            video_file = os.path.join(
                os.path.dirname(image_path), f"{source_identity}.zoom-{duration_identity}.mp4"
            )
            descriptor, temp_path = tempfile.mkstemp(
                prefix=".image-zoom-",
                suffix=".mp4",
                dir=os.path.dirname(os.path.abspath(video_file)),
            )
            os.close(descriptor)
            final_clip.write_videofile(
                temp_path, fps=30, logger=None, ffmpeg_params=_BT709_FFMPEG_PARAMS
            )
        finally:
            close_clip(final_clip)
        os.replace(temp_path, video_file)
        temp_path = ""
        return video_file
    finally:
        close_clip(clip)
        if temp_path:
            delete_files(temp_path)


def preprocess_video(materials: List[MaterialInfo], clip_duration=4):
    # In some regenerate flows the WebUI may pass an empty material list. Return an empty result here instead of raising a NoneType error.
    if not materials:
        return []

    # Return only materials that passed preprocessing, so low-resolution images do not continue into video compositing.
    valid_materials = []
    local_videos_dir = utils.storage_dir("local_videos", create=True)

    for material in materials:
        if not material.url:
            continue

        try:
            material_source_path = file_security.resolve_path_within_directory(
                local_videos_dir, material.url
            )
        except ValueError as exc:
            # Material paths for a local video_source come from API parameters and must stay inside the dedicated material directory.
            # A file name is allowed, and a historically returned absolute path is allowed, but the path must not escape into other
            # system directories. That blocks arbitrary file reads and MoviePy probing of local sensitive files.
            logger.warning(
                f"skip unsafe local material: {material.url}, "
                f"local_videos_dir: {local_videos_dir}, error: {str(exc)}"
            )
            continue

        ext = utils.parse_extension(material_source_path)
        is_image = ext in const.FILE_TYPE_IMAGES
        try:
            # Read image materials as images directly, so a VideoFileClip misdetect does not trigger an unstable fallback.
            if is_image:
                clip, material_source_path = _open_image_clip_with_fallback(
                    material_source_path
                )
            else:
                clip = _open_video_clip_quietly(material_source_path)
        except Exception:
            # For a nonstandard extension, or when probing fails, fall back to image mode, which matches older calls that passed a local image path directly.
            try:
                clip, material_source_path = _open_image_clip_with_fallback(
                    material_source_path
                )
                # The successful decoder determines the material kind, even
                # when the uploaded filename has a video or unknown suffix.
                is_image = True
            except Exception as exc:
                logger.warning(
                    f"skip unreadable local material: {material.url}, error: {str(exc)}"
                )
                continue
        try:
            width = clip.size[0]
            height = clip.size[1]
            if not is_material_resolution_acceptable(width, height):
                logger.warning(
                    f"low resolution material: {width}x{height}, minimum "
                    f"{_MIN_MATERIAL_DIMENSION}x{_MIN_MATERIAL_DIMENSION} required "
                    f"(tolerance {_MIN_DIMENSION_TOLERANCE}px)"
                )
                # After a low-resolution material is detected, close it immediately and do not return it to the later pipeline.
                close_clip(clip)
                continue

            if is_image:
                logger.info(f"processing image: {material_source_path}")
                # The probe already opened the material once to read its size. Release that handle before rendering
                # the image segment used for export.
                close_clip(clip)
                video_file = render_image_zoom_video(
                    material_source_path, clip_duration
                )
                material.url = video_file
                logger.success(f"image processed: {video_file}")
            else:
                # A normal video material only needs its size checked. Release the handle as soon as that check is done.
                close_clip(clip)
                # Update url to the resolved absolute path so that downstream
                # stages (combine_videos) can open the file without re-resolving.
                material.url = material_source_path
        except Exception:
            close_clip(clip)
            raise

        valid_materials.append(material)

    return valid_materials
