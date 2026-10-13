import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import requests
from loguru import logger

from app.config import config
from app.services import bgm as bgm_service
from app.utils import utils


DEFAULT_BASE_URL = "https://api.elevenlabs.io"
VIDEO_TO_MUSIC_PATH = "/v1/music/video-to-music"
SUBSCRIPTION_PATH = "/v1/user/subscription"
DEFAULT_MODEL_ID = "music_v2"
SUPPORTED_MODEL_IDS = frozenset({"music_v1", "music_v2"})
MAX_VIDEO_DURATION_SECONDS = 600
MAX_PROMPT_LENGTH = 1000
MAX_PROXY_BYTES = 200 * 1024 * 1024
MAX_GENERATED_AUDIO_BYTES = 50 * 1024 * 1024
MAX_ERROR_BODY_BYTES = 500


class ElevenLabsMusicError(RuntimeError):
    """An ElevenLabs music request, proxy generation, or returned-audio validation failed."""


class ElevenLabsPaidPlanRequiredError(ElevenLabsMusicError):
    """The key is valid, but the current account plan does not include the ElevenLabs Music API."""


class ElevenLabsAuthenticationError(ElevenLabsMusicError):
    """The ElevenLabs API key is missing or was rejected by the server."""


def get_api_key() -> str:
    """
    Read the shared ElevenLabs API key.

    Music and the existing ElevenLabs TTS use the same account setting, so the user does not maintain
    two keys in the WebUI; the environment variable is only a fallback when the local setting is empty.
    """
    configured_key = str(config.elevenlabs.get("api_key", "") or "").strip()
    return configured_key or os.getenv("ELEVENLABS_API_KEY", "").strip()


def is_enabled() -> bool:
    return bool(get_api_key())


def _base_url() -> str:
    return str(
        config.elevenlabs.get("music_base_url", DEFAULT_BASE_URL)
        or DEFAULT_BASE_URL
    ).rstrip("/")


def _model_id() -> str:
    """Allow only the models currently published for official Video-to-Music, and fall back safely on a bad setting."""
    model_id = str(
        config.elevenlabs.get("music_model_id", DEFAULT_MODEL_ID)
        or DEFAULT_MODEL_ID
    ).strip()
    return model_id if model_id in SUPPORTED_MODEL_IDS else DEFAULT_MODEL_ID


def _request_timeout() -> tuple[int, int]:
    """Cap the music read timeout so long video generation can finish and a bad setting can still recover."""
    raw_timeout = config.elevenlabs.get("music_timeout", 600)
    try:
        read_timeout = float(raw_timeout)
    except (TypeError, ValueError):
        read_timeout = 600
    if not math.isfinite(read_timeout) or read_timeout <= 0:
        read_timeout = 600
    return 15, max(1, math.ceil(min(read_timeout, 1800)))


def _safe_response_error(response: requests.Response) -> str:
    """Read only a limited third-party error body so a bad response cannot exhaust memory or pollute task logs."""
    try:
        body_bytes = next(
            response.iter_content(chunk_size=MAX_ERROR_BODY_BYTES),
            b"",
        )
    except requests.RequestException:
        body_bytes = b""
    if isinstance(body_bytes, bytes):
        try:
            body = body_bytes.decode(
                response.encoding or "utf-8",
                errors="replace",
            )
        except LookupError:
            # response.encoding comes straight from the upstream charset, and an unknown value (for example
            # charset=unknown-charset) makes codecs raise LookupError, while errors
            # only affects UnicodeDecodeError. The old response.text fell back internally to
            # UTF-8; this keeps that behavior so the exception still hits the caller's music fallback path.
            body = body_bytes.decode("utf-8", errors="replace")
    else:
        body = str(body_bytes)
    body = body.strip().replace("\n", " ")[:MAX_ERROR_BODY_BYTES]
    return body or response.reason or "request failed"


def test_connection() -> dict[str, Any]:
    """
    Check the API key and account plan with the subscription endpoint, which does not spend music credits.

    That endpoint can only confirm the key can read subscription info and the account is not on the free plan. It cannot prove the key
    has Music endpoint permission. ElevenLabs can restrict a
    key by endpoint, quota, and IP, so the UI success message must keep that limit; real permission is confirmed by the generation request.
    Billing and usage details in the response are not written to logs, so account privacy stays out of the log.
    """
    api_key = get_api_key()
    if not api_key:
        raise ElevenLabsAuthenticationError("ElevenLabs API key is required")
    try:
        with requests.get(
            f"{_base_url()}{SUBSCRIPTION_PATH}",
            headers={"xi-api-key": api_key},
            timeout=(15, 30),
            stream=True,
            allow_redirects=False,
        ) as response:
            if 300 <= response.status_code < 400:
                raise ElevenLabsMusicError(
                    "ElevenLabs account check returned a redirect; "
                    "the API key was not forwarded"
                )
            if response.status_code == 401:
                raise ElevenLabsAuthenticationError(
                    "ElevenLabs API key was rejected (401): "
                    f"{_safe_response_error(response)}"
                )
            if not response.ok:
                raise ElevenLabsMusicError(
                    "ElevenLabs account check failed "
                    f"({response.status_code}): "
                    f"{_safe_response_error(response)}"
                )
            try:
                payload = response.json()
            except ValueError as exc:
                raise ElevenLabsMusicError(
                    "ElevenLabs returned an invalid subscription response"
                ) from exc
    except requests.RequestException as exc:
        raise ElevenLabsMusicError(
            f"failed to connect to ElevenLabs: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ElevenLabsMusicError(
            "ElevenLabs returned an unexpected subscription response"
        )
    tier = str(payload.get("tier") or "").strip().lower()
    if not tier:
        raise ElevenLabsMusicError(
            "ElevenLabs subscription response does not include an account tier"
        )
    if tier == "free":
        raise ElevenLabsPaidPlanRequiredError(
            "ElevenLabs Music API requires a paid plan; "
            "the current account is on the free tier"
        )
    logger.info(f"ElevenLabs account and plan check succeeded: tier={tier}")
    return payload


def validate_generation_access() -> None:
    """
    Reject accounts that definitely cannot generate music before the expensive video pipeline starts.

    A free plan and an invalid key are deterministic errors and must stop immediately, so LLM, TTS,
    and material credits are not spent first. The subscription endpoint can also be unreachable because of a Music-only endpoint scope, an IP restriction, or
    a temporary network problem; those results do not prove the Music API is unavailable, so they only log a warning
    and let the real generation request decide, which avoids blocking a restricted but usable key.
    """
    try:
        test_connection()
    except (ElevenLabsPaidPlanRequiredError, ElevenLabsAuthenticationError):
        raise
    except ElevenLabsMusicError as exc:
        logger.warning(
            "ElevenLabs account preflight was inconclusive; "
            f"generation will verify Music API access: error={exc}"
        )


def _remove_file(file_path: str) -> None:
    """Best-effort cleanup of ElevenLabs intermediate files, without hiding the original exception the caller is handling."""
    if not file_path or not os.path.exists(file_path):
        return
    try:
        os.remove(file_path)
    except OSError as exc:
        logger.warning(
            "failed to remove ElevenLabs temporary file: "
            f"path={file_path}, error={exc}"
        )


def _create_video_proxy(video_path: str) -> str:
    """
    Build an H.264 proxy video with no audio and a longest edge of 1280 pixels.

    Video-to-Music only analyzes the picture, so uploading the original HD file does not improve the music and only adds bandwidth
    and wait time. The proxy stays strictly within the official 200 MB limit and is deleted after the request.
    """
    descriptor, proxy_path = tempfile.mkstemp(
        prefix=".elevenlabs-music-proxy-",
        suffix=".mp4",
        dir=os.path.dirname(os.path.abspath(video_path)),
    )
    os.close(descriptor)
    command = [
        utils.get_ffmpeg_binary(),
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-i",
        video_path,
        "-vf",
        (
            "scale=w=1280:h=1280:force_original_aspect_ratio=decrease:"
            "force_divisible_by=2"
        ),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "30",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-fs",
        str(MAX_PROXY_BYTES),
        proxy_path,
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        _remove_file(proxy_path)
        raise ElevenLabsMusicError(
            "ElevenLabs video proxy generation timed out"
        ) from exc
    except OSError as exc:
        _remove_file(proxy_path)
        raise ElevenLabsMusicError(
            "failed to run FFmpeg for ElevenLabs video proxy"
        ) from exc
    if result.returncode != 0:
        _remove_file(proxy_path)
        detail = (result.stderr or "").strip().replace("\n", " ")[-500:]
        raise ElevenLabsMusicError(
            f"failed to generate ElevenLabs video proxy: {detail}"
        )
    proxy_size = os.path.getsize(proxy_path) if os.path.isfile(proxy_path) else 0
    if proxy_size <= 0 or proxy_size > MAX_PROXY_BYTES:
        _remove_file(proxy_path)
        raise ElevenLabsMusicError(
            "ElevenLabs video proxy is empty or exceeds the 200 MB limit"
        )
    logger.info(
        "ElevenLabs video proxy prepared: "
        f"source={video_path}, size={proxy_size} bytes"
    )
    return proxy_path


def _stream_audio(response: requests.Response, temp_audio_path: str) -> int:
    """Save the audio in chunks with a size cap so a bad response cannot fill the local disk."""
    total_bytes = 0
    with open(temp_audio_path, "wb") as output:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            total_bytes += len(chunk)
            if total_bytes > MAX_GENERATED_AUDIO_BYTES:
                raise ElevenLabsMusicError(
                    "ElevenLabs audio exceeds the 50 MB limit"
                )
            output.write(chunk)
        output.flush()
        os.fsync(output.fileno())
    if total_bytes <= 0:
        raise ElevenLabsMusicError("ElevenLabs returned no audio data")
    return total_bytes


def _request_bgm(video_path: str, output_path: str, prompt: str) -> str:
    """Request ElevenLabs music, download it fully, validate it with FFmpeg, then publish it atomically."""
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    descriptor, temp_audio_path = tempfile.mkstemp(
        prefix=".elevenlabs-music-",
        suffix=Path(output_path).suffix or ".mp3",
        dir=output_dir,
    )
    os.close(descriptor)
    try:
        model_id = _model_id()
        logger.info(
            "requesting ElevenLabs background music: "
            f"video={video_path}, model={model_id}, "
            f"prompt_provided={bool(prompt)}"
        )
        request_data = {"model_id": model_id}
        if prompt:
            request_data["description"] = prompt
        try:
            with open(video_path, "rb") as video_file:
                response = requests.post(
                    f"{_base_url()}{VIDEO_TO_MUSIC_PATH}",
                    headers={"xi-api-key": get_api_key()},
                    params={"output_format": "mp3_44100_128"},
                    files=[
                        (
                            # Official docs show the form array as ``videos[]``, but on 2026-07-18
                            # the production API returned 422 for that field; the real Starlette parameter name is
                            # ``videos``. requests can keep adding the same field name for repeated uploads.
                            "videos",
                            (Path(video_path).name, video_file, "video/mp4"),
                        )
                    ],
                    data=request_data,
                    stream=True,
                    timeout=_request_timeout(),
                    allow_redirects=False,
                )
                with response:
                    if 300 <= response.status_code < 400:
                        raise ElevenLabsMusicError(
                            "ElevenLabs generation returned a redirect; "
                            "the video and API key were not forwarded"
                        )
                    if not response.ok:
                        raise ElevenLabsMusicError(
                            "ElevenLabs generation failed "
                            f"({response.status_code}): "
                            f"{_safe_response_error(response)}"
                        )
                    total_bytes = _stream_audio(response, temp_audio_path)
        except requests.RequestException as exc:
            # A disconnect during download is also a request failure and must enter task fallback, so it does not leave a partial
            # audio file or fail the already generated video because of a third-party network blip.
            raise ElevenLabsMusicError(
                f"failed to request ElevenLabs music: {exc}"
            ) from exc

        try:
            bgm_service.validate_audio_file(temp_audio_path, timeout_seconds=120)
        except (bgm_service.BgmUploadError, bgm_service.BgmServiceError) as exc:
            raise ElevenLabsMusicError(
                "ElevenLabs returned audio that FFmpeg cannot decode"
            ) from exc
        os.replace(temp_audio_path, output_path)
        temp_audio_path = ""
        logger.info(
            "ElevenLabs background music generated: "
            f"output={output_path}, size={total_bytes} bytes"
        )
        return output_path
    finally:
        _remove_file(temp_audio_path)


def generate_bgm(
    video_path: str,
    output_path: str,
    video_duration: float,
    prompt: str = "",
) -> str:
    """Generate ElevenLabs background music matched to the duration and picture of one concatenated video."""
    if not get_api_key():
        raise ElevenLabsMusicError("ElevenLabs API key is required")
    if not os.path.isfile(video_path):
        raise ElevenLabsMusicError("ElevenLabs input video does not exist")
    try:
        duration = float(video_duration)
    except (TypeError, ValueError) as exc:
        raise ElevenLabsMusicError(
            "ElevenLabs video duration is invalid"
        ) from exc
    if not math.isfinite(duration) or duration <= 0:
        raise ElevenLabsMusicError("ElevenLabs video duration is invalid")
    if duration > MAX_VIDEO_DURATION_SECONDS:
        raise ElevenLabsMusicError(
            "ElevenLabs supports videos up to 600 seconds"
        )
    prompt = str(prompt or "").strip()
    if len(prompt) > MAX_PROMPT_LENGTH:
        raise ElevenLabsMusicError(
            "ElevenLabs music prompt exceeds 1000 characters"
        )

    proxy_path = ""
    try:
        proxy_path = _create_video_proxy(video_path)
        return _request_bgm(proxy_path, output_path, prompt)
    except ElevenLabsMusicError:
        raise
    except OSError as exc:
        raise ElevenLabsMusicError(
            f"ElevenLabs local file operation failed: {exc}"
        ) from exc
    finally:
        _remove_file(proxy_path)
