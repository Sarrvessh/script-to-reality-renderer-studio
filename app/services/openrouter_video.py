"""OpenRouter's asynchronous video API, kept separate from the language-model client.

Quote and model discovery only call GET /videos/models. A paid clip is submitted
once, and only after the caller has approved a cost that fits the budget.
A failed or uncertain submission is never sent again.
"""

from __future__ import annotations

import logging
import os
import time
from decimal import Decimal, InvalidOperation
from typing import Any, Callable
from urllib.parse import urlparse

import requests

from app.config import config
from app.services import state as sm
from app.utils import utils

logger = logging.getLogger(__name__)

API_ROOT = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "heygen/heygen-video-1"
DEFAULT_RESOLUTION = "480p"
DEFAULT_DURATION = 5
DEFAULT_ASPECT = "9:16"
DEFAULT_BUDGET = Decimal("0.10")
BUDGET_CEILING = Decimal("0.10")
_POLL_INTERVAL_SECONDS = 10
_POLL_DEADLINE_SECONDS = 600
_JOB_ID = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")


class OpenRouterVideoError(RuntimeError):
    def __init__(self, message: str, job_id: str = "") -> None:
        super().__init__(message)
        self.job_id = job_id


class OpenRouterVideoApprovalRequired(OpenRouterVideoError):
    pass


class OpenRouterVideoBudgetError(OpenRouterVideoError):
    pass


def resolve_api_key() -> str:
    """Reuse the key already saved for OpenRouter text. Never log the value."""
    stored = str(config.app.get("openrouter_api_key", "") or "").strip()
    if stored:
        return stored
    return os.getenv("OPENROUTER_API_KEY", "").strip()


def is_configured() -> bool:
    return bool(resolve_api_key())


def _headers() -> dict[str, str]:
    key = resolve_api_key()
    if not key:
        raise OpenRouterVideoError("Save an OpenRouter key in Settings before generating video.")
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _money(value: Any) -> Decimal:
    try:
        amount = Decimal(str(value).strip())
    except (InvalidOperation, AttributeError) as exc:
        raise OpenRouterVideoError("OpenRouter did not return a usable video price.") from exc
    if not amount.is_finite() or amount < 0:
        raise OpenRouterVideoError("OpenRouter did not return a usable video price.")
    return amount


def _rate_for(pricing: dict[str, Any], resolution: str) -> tuple[Decimal, str]:
    """Price one second of text-to-video. Reference SKUs are for image-guided jobs."""
    lowered = {str(key).lower(): value for key, value in pricing.items()}
    token = "".join(ch for ch in resolution.lower() if ch.isalnum())
    for key in (
        f"duration_seconds_{token}",
        f"per-video-second-{resolution.lower()}",
        "duration_seconds",
        "per-video-second",
    ):
        if key in lowered:
            return _money(lowered[key]), "second"
    for key in ("per-video", "video"):
        if key in lowered:
            return _money(lowered[key]), "video"
    raise OpenRouterVideoError(
        f"OpenRouter has no text-to-video price for {resolution}. The clip was not submitted."
    )


def _format_money(amount: Decimal) -> str:
    quantized = amount.quantize(Decimal("0.0001"))
    text = format(quantized, "f").rstrip("0").rstrip(".")
    if "." not in text or len(text.split(".", 1)[1]) < 2:
        return f"{quantized:.2f}"
    return text


def estimate_cost(model: dict[str, Any], duration: int, resolution: str) -> dict[str, Any]:
    rate, unit = _rate_for(model.get("pricing_skus") or {}, resolution)
    clips = 1
    total = rate if unit == "video" else rate * Decimal(duration)
    return {
        "rate_usd": _format_money(rate),
        "rate_unit": unit,
        "clips": clips,
        "estimated_usd": _format_money(total),
    }


def list_models(session: requests.Session | None = None) -> list[dict[str, Any]]:
    client = session or requests.Session()
    try:
        response = client.get(f"{API_ROOT}/videos/models", headers=_headers(), timeout=30)
    except requests.RequestException as exc:
        raise OpenRouterVideoError("OpenRouter video models could not be loaded.") from exc
    if response.status_code != 200:
        raise OpenRouterVideoError("OpenRouter video models could not be loaded.")
    try:
        payload = response.json()
    except ValueError as exc:
        raise OpenRouterVideoError("OpenRouter video models could not be loaded.") from exc
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise OpenRouterVideoError("OpenRouter video models could not be loaded.")
    return [row for row in rows if isinstance(row, dict) and str(row.get("id") or "").strip()]


def public_models() -> list[dict[str, Any]]:
    models = []
    for row in list_models():
        models.append({
            "id": str(row.get("id") or ""),
            "name": str(row.get("name") or row.get("id") or ""),
            "durations": list(row.get("supported_durations") or []),
            "resolutions": list(row.get("supported_resolutions") or []),
            "aspects": list(row.get("supported_aspect_ratios") or []),
        })
    return models


def _find_model(models: list[dict[str, Any]], model_id: str) -> dict[str, Any]:
    for row in models:
        if str(row.get("id") or "") == model_id:
            return row
    raise OpenRouterVideoError(f"{model_id} is not available for OpenRouter video. The clip was not submitted.")


def _require_choice(model: dict[str, Any], field: str, value: Any, label: str) -> None:
    allowed = model.get(field) or []
    if allowed and value not in allowed and str(value) not in {str(item) for item in allowed}:
        supported = ", ".join(str(item) for item in allowed)
        raise OpenRouterVideoError(
            f"{label} {value} is not supported by this model. Supported: {supported}. The clip was not submitted."
        )


def _authorized_budget(value: Any) -> Decimal:
    """The approved ceiling is $0.10. A higher request cannot raise it."""
    if value in (None, ""):
        return BUDGET_CEILING
    requested = _money(value)
    if requested > BUDGET_CEILING:
        return BUDGET_CEILING
    return requested


def project_budget(value: Any) -> Decimal:
    """A preconfigured project budget. This is separate from the single-clip ceiling."""
    if value in (None, ""):
        return Decimal("1.00")
    requested = _money(value)
    if requested > Decimal("100"):
        return Decimal("100")
    return requested


def quote(body: dict[str, Any] | None = None, session: requests.Session | None = None) -> dict[str, Any]:
    """Price one clip. This never submits a generation job."""
    payload = body if isinstance(body, dict) else {}
    model_id = str(payload.get("model") or _configured_model()).strip() or DEFAULT_MODEL
    resolution = str(payload.get("resolution") or DEFAULT_RESOLUTION).strip() or DEFAULT_RESOLUTION
    aspect = str(payload.get("aspect_ratio") or DEFAULT_ASPECT).strip() or DEFAULT_ASPECT
    try:
        duration = int(payload.get("duration") or DEFAULT_DURATION)
    except (TypeError, ValueError) as exc:
        raise OpenRouterVideoError("Duration must be a whole number of seconds.") from exc
    project = payload.get("project_budget") is True
    budget = project_budget(payload.get("budget_usd")) if project else _authorized_budget(payload.get("budget_usd"))
    models = list_models(session)
    model = _find_model(models, model_id)
    _require_choice(model, "supported_durations", duration, "Duration")
    _require_choice(model, "supported_resolutions", resolution, "Resolution")
    _require_choice(model, "supported_aspect_ratios", aspect, "Aspect ratio")
    priced = estimate_cost(model, duration, resolution)
    estimated = _money(priced["estimated_usd"])
    affordable = int(budget / estimated) if estimated > 0 else 0
    return {
        "model": model_id,
        "name": str(model.get("name") or model_id),
        "duration": duration,
        "resolution": resolution,
        "aspect_ratio": aspect,
        "clips": priced["clips"],
        "rate_usd": priced["rate_usd"],
        "rate_unit": priced["rate_unit"],
        "estimated_usd": priced["estimated_usd"],
        "budget_usd": _format_money(budget),
        "within_budget": estimated <= budget,
        "max_scenes": min(6, affordable) if project else 1,
        "requires_approval": not project,
        "submitted": False,
    }


def _configured_model() -> str:
    saved = str(config.app.get("openrouter_video_model", "") or "").strip()
    return saved or DEFAULT_MODEL


def _job_id(value: Any) -> str:
    job_id = str(value or "").strip()
    if not job_id or len(job_id) > 128 or any(ch not in _JOB_ID for ch in job_id):
        raise OpenRouterVideoError("OpenRouter accepted a video job but did not return a usable job id.")
    return job_id


def _content_url(job_id: str, advertised: Any) -> str:
    fallback = f"{API_ROOT}/videos/{job_id}/content?index=0"
    if isinstance(advertised, list) and advertised:
        candidate = str(advertised[0] or "")
        parsed = urlparse(candidate)
        if parsed.scheme == "https" and parsed.netloc == "openrouter.ai" and parsed.path.startswith("/api/v1/videos/"):
            return candidate
    return fallback


def _remember_job(task_id: str, job_id: str, model_id: str) -> None:
    try:
        sm.state.patch_task(task_id, openrouter_video_job_id=job_id, openrouter_video_model=model_id)
    except Exception:
        # The paid job already exists. Losing the local note must not hide the id from the log.
        pass


def _submit_once(client: requests.Session, payload: dict[str, Any]) -> str:
    try:
        response = client.post(f"{API_ROOT}/videos", headers=_headers(), json=payload, timeout=60)
    except requests.RequestException as exc:
        raise OpenRouterVideoError(
            "The video request did not return a job id, so it was not sent again."
        ) from exc
    if response.status_code not in {200, 202}:
        raise OpenRouterVideoError("OpenRouter rejected the video request before a job was created. It was not sent again.")
    try:
        body = response.json()
    except ValueError as exc:
        raise OpenRouterVideoError(
            "OpenRouter did not return a readable job id, so the request was not sent again."
        ) from exc
    if not isinstance(body, dict):
        raise OpenRouterVideoError("OpenRouter did not return a job id, so the request was not sent again.")
    return _job_id(body.get("id"))


def _poll(client: requests.Session, job_id: str, sleep: Callable[[float], None], clock: Callable[[], float]) -> dict[str, Any]:
    deadline = clock() + _POLL_DEADLINE_SECONDS
    url = f"{API_ROOT}/videos/{job_id}"
    while True:
        try:
            response = client.get(url, headers=_headers(), timeout=30)
            body = response.json()
        except (requests.RequestException, ValueError) as exc:
            if clock() >= deadline:
                raise OpenRouterVideoError(
                    "The video job could not be checked before the time limit. It was not submitted again.",
                    job_id=job_id,
                ) from exc
            sleep(_POLL_INTERVAL_SECONDS)
            continue
        if not isinstance(body, dict):
            raise OpenRouterVideoError("OpenRouter returned an unreadable job status. It was not submitted again.", job_id=job_id)
        status = str(body.get("status") or "").strip().lower()
        if status == "completed":
            return body
        if status in {"failed", "cancelled", "canceled", "expired"}:
            detail = str(body.get("error") or status or "failed")
            raise OpenRouterVideoError(
                f"OpenRouter video job {job_id} {status}: {detail}. It was not submitted again.",
                job_id=job_id,
            )
        if clock() >= deadline:
            raise OpenRouterVideoError(
                f"OpenRouter video job {job_id} is still {status or 'running'}. It was not submitted again.",
                job_id=job_id,
            )
        sleep(_POLL_INTERVAL_SECONDS)


def _download(client: requests.Session, url: str, destination: str) -> None:
    """Download the same content URL. A failed download never starts another job."""
    last_error: Exception | None = None
    for _attempt in range(2):
        try:
            response = client.get(url, headers=_headers(), timeout=120)
        except requests.RequestException as exc:
            last_error = exc
            continue
        content = response.content or b""
        if response.status_code == 200 and len(content) >= 1024 and content[:1] not in {b"{", b"<"}:
            with open(destination, "wb") as handle:
                handle.write(content)
            return
        last_error = OpenRouterVideoError("The finished video could not be downloaded. It was not generated again.")
    raise OpenRouterVideoError("The finished video could not be downloaded. It was not generated again.") from last_error


def generate_for_task(
    task_id: str,
    params: Any,
    search_terms: list[str] | None = None,
    *,
    session: requests.Session | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> list[str]:
    """Generate exactly one approved clip and return its local MP4 path.

    Stock footage is never consulted. A missing approval or an over-budget
    estimate returns before POST /videos.
    """
    if getattr(params, "openrouter_video_approved", False) is not True:
        raise OpenRouterVideoApprovalRequired(
            "Approve the OpenRouter video cost before generating. No clip was submitted."
        )
    model_id = str(getattr(params, "openrouter_video_model", "") or _configured_model()).strip() or DEFAULT_MODEL
    resolution = str(getattr(params, "openrouter_video_resolution", "") or DEFAULT_RESOLUTION).strip() or DEFAULT_RESOLUTION
    raw_aspect = getattr(params, "video_aspect", "") or DEFAULT_ASPECT
    aspect = str(getattr(raw_aspect, "value", raw_aspect) or DEFAULT_ASPECT)
    duration = int(getattr(params, "openrouter_video_duration", None) or DEFAULT_DURATION)
    budget = _authorized_budget(getattr(params, "openrouter_video_budget_usd", None))
    client = session or requests.Session()
    priced = quote(
        {
            "model": model_id,
            "resolution": resolution,
            "aspect_ratio": aspect,
            "duration": duration,
            "budget_usd": format(budget, "f"),
        },
        session=client,
    )
    if not priced["within_budget"]:
        raise OpenRouterVideoBudgetError(
            f"This clip is estimated at ${priced['estimated_usd']}, above the ${priced['budget_usd']} budget. It was not submitted."
        )
    prompt = str(
        getattr(params, "video_script", "")
        or getattr(params, "video_subject", "")
        or (search_terms[0] if search_terms else "")
        or ""
    ).strip()
    if not prompt:
        raise OpenRouterVideoError("Write a subject or script before generating OpenRouter video.")
    job_id = _submit_once(
        client,
        {
            "model": model_id,
            "prompt": prompt[:8000],
            "duration": duration,
            "resolution": resolution,
            "aspect_ratio": aspect,
        },
    )
    logger.info("OpenRouter video job created: task_id=%s job_id=%s model=%s", task_id, job_id, model_id)
    _remember_job(task_id, job_id, model_id)
    status = _poll(client, job_id, sleep, clock)
    destination = os.path.join(utils.task_dir(task_id), "openrouter-video.mp4")
    _download(client, _content_url(job_id, status.get("unsigned_urls")), destination)
    return [destination]


def generate_clip(
    task_id: str,
    *,
    prompt: str,
    model_id: str,
    resolution: str,
    aspect: str,
    duration: int,
    budget: Decimal,
    destination: str,
    existing_job_id: str = "",
    model: dict[str, Any] | None = None,
    session: requests.Session | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Submit one clip, or resume a saved job id without posting again.

    ``budget`` is the money still available. An existing job is polled even when
    that remainder would not pay for a new clip.
    """
    text = str(prompt or "").strip()
    if not text:
        raise OpenRouterVideoError("A scene prompt is required before a clip can be submitted.")
    client = session or requests.Session()
    row = model if isinstance(model, dict) else _find_model(list_models(client), model_id)
    _require_choice(row, "supported_durations", duration, "Duration")
    _require_choice(row, "supported_resolutions", resolution, "Resolution")
    _require_choice(row, "supported_aspect_ratios", aspect, "Aspect ratio")
    estimated = _money(estimate_cost(row, duration, resolution)["estimated_usd"])
    job_id = str(existing_job_id or "").strip()
    if not job_id:
        if estimated > budget:
            raise OpenRouterVideoBudgetError(
                f"budget-exhausted: this scene is estimated at ${_format_money(estimated)}, "
                f"above the ${_format_money(budget)} still available. It was not submitted."
            )
        job_id = _submit_once(
            client,
            {
                "model": model_id,
                "prompt": text[:8000],
                "duration": duration,
                "resolution": resolution,
                "aspect_ratio": aspect,
            },
        )
        logger.info("OpenRouter video job created: task_id=%s job_id=%s model=%s", task_id, job_id, model_id)
        _remember_job(task_id, job_id, model_id)
        submitted = True
    else:
        submitted = False
    status = _poll(client, job_id, sleep, clock)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    _download(client, _content_url(job_id, status.get("unsigned_urls")), destination)
    return {
        "path": destination,
        "job_id": job_id,
        "estimated_usd": _format_money(estimated),
        "submitted": submitted,
    }
