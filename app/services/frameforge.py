"""Script-to-Reality Renderer Studio project, storyboard, and render preparation.

The video pipeline itself stays in the original services. This module stores
editable projects and turns a storyboard into the request those services
already accept.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlencode, urlsplit
from uuid import uuid4

import requests

from loguru import logger

from app.config import config
from app.models.llm_provider import LLM_PROVIDER_REGISTRY, get_llm_provider
from app.models.schema import MaterialInfo, TaskVideoRequest, VideoAspect
from app.services import llm, material, state as sm, voice as voice_service
from app.utils import utils

_storage_override: str | None = None

_PROJECT_ID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
_SECRET = re.compile(
    r"(sk-[A-Za-z0-9_\-]{8,}|Bearer\s+\S+|api[_-]?key\s*[=:]\s*\S+)",
    re.IGNORECASE,
)
_HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")
_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "for",
    "with", "from", "that", "this", "these", "those", "is", "are", "was",
    "were", "be", "as", "at", "by", "it", "its", "into", "about", "after",
    "before", "over", "under", "their", "his", "her", "they", "you", "your",
}

VOICES = (
    {"id": "en-US-AriaNeural-Female", "label": "Aria", "detail": "Clear American"},
    {"id": "en-US-JennyNeural-Female", "label": "Jenny", "detail": "Bright American"},
    {"id": "en-US-GuyNeural-Male", "label": "Guy", "detail": "Steady American"},
    {"id": "en-US-DavisNeural-Male", "label": "Davis", "detail": "Direct American"},
    {"id": "en-GB-SoniaNeural-Female", "label": "Sonia", "detail": "British"},
    {"id": "en-GB-RyanNeural-Male", "label": "Ryan", "detail": "British"},
    {"id": "en-AU-NatashaNeural-Female", "label": "Natasha", "detail": "Australian"},
    {"id": "en-IN-NeerjaNeural-Female", "label": "Neerja", "detail": "Indian English"},
)
VOICE_IDS = {item["id"] for item in VOICES}
ASPECTS = {"9:16", "16:9", "1:1"}
TRANSITIONS = {"FadeIn", "FadeOut", "SlideIn", "SlideOut", "ZoomIn", "ZoomOut", "Shuffle", "none"}
POSITIONS = {"bottom", "center", "top", "two_thirds_bottom", "custom"}
DISPLAY_MODES = {"sentence", "word_by_word"}
ANIMATIONS = {"none", "pop_spring"}
FIT_MODES = {"cover", "contain"}
CONCAT_MODES = {"sequential", "random"}
SUBTITLE_PROVIDERS = {"edge", "whisper"}
VIDEO_SOURCES = (
    "pexels", "pixabay", "coverr", "local",
    "wavespeed", "volcengine_seedance", "ofox", "metaso_minimax",
    "muapi", "loomloom", "openai_image", "openrouter",
)
CODECS = {"__default__", "libx264", "h264_nvenc", "h264_amf", "h264_qsv", "h264_mf", "h264_videotoolbox"}
_VOICE_ID = re.compile(r"^[A-Za-z]{2,3}-[A-Za-z]{2}-[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*$")
FONT_CANDIDATES = (
    ("BeVietnamPro-Bold.ttf", "Be Vietnam Pro Bold"),
    ("BeVietnamPro-Medium.ttf", "Be Vietnam Pro Medium"),
    ("STHeitiMedium.ttc", "Heiti Medium"),
)
SCRIPT_PROMPT = (
    "Write a clear English narration of 45 to 70 words for a short vertical video. "
    "Use three or four short sentences. Do not include a title, hashtags, "
    "stage directions, or bullet points."
)


def _stock_configuration_message(exc: Exception) -> str:
    text = str(exc)
    if "pexels_api_keys" in text or "not set" in text:
        return "Pexels is not configured. Add PEXELS_API_KEY to the local environment."
    return public_error(text)


def public_error(message: str) -> str:
    text = _SECRET.sub("[redacted]", message or "")
    text = " ".join(text.split())
    return text[:300] or "The request failed."


def split_narration(script: str) -> list[str]:
    """Turn a narration into ordered scenes without calling a model."""
    text = (script or "").strip()
    if not text:
        return []
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if len(paragraphs) > 1:
        return paragraphs[:8]
    sentences = [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+", text)
        if part.strip()
    ]
    if len(sentences) <= 3:
        return sentences or [text]
    group_size = max(1, (len(sentences) + 3) // 4)
    groups = [
        " ".join(sentences[index:index + group_size])
        for index in range(0, len(sentences), group_size)
    ]
    return [group for group in groups if group][:8]


def fallback_term(narration: str) -> str:
    words = re.findall(r"[A-Za-z0-9']+", narration or "")
    kept = [word for word in words if word.lower() not in _STOP_WORDS]
    phrase = " ".join(kept[:4]).strip()
    return phrase or "city street"


def assign_terms(narrations: list[str], terms: list[str] | None) -> list[str]:
    """Pair ordered search terms with scenes, filling any gap from the narration."""
    source = [term.strip() for term in (terms or []) if isinstance(term, str) and term.strip()]
    assigned = []
    for index, narration in enumerate(narrations):
        assigned.append(source[index] if index < len(source) else fallback_term(narration))
    return assigned


def clip_url_allowed(url: str) -> bool:
    parsed = urlsplit(url or "")
    host = (parsed.hostname or "").lower()
    return parsed.scheme == "https" and (host == "pexels.com" or host.endswith(".pexels.com"))


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _root() -> str:
    if _storage_override:
        os.makedirs(_storage_override, exist_ok=True)
        return _storage_override
    return utils.storage_dir("frameforge", create=True)


def _projects_dir() -> str:
    path = os.path.join(_root(), "projects")
    os.makedirs(path, exist_ok=True)
    return path


def _settings_path() -> str:
    return os.path.join(_root(), "settings.json")


def _limit(value: Any, length: int) -> str:
    return " ".join(str(value or "").split())[:length].strip()


def _require_project_id(project_id: str) -> str:
    if not isinstance(project_id, str) or not _PROJECT_ID.fullmatch(project_id):
        raise LookupError("Project not found.")
    return project_id


def _project_path(project_id: str) -> str:
    project_id = _require_project_id(project_id)
    path = os.path.join(_projects_dir(), f"{project_id}.json")
    if os.path.dirname(os.path.abspath(path)) != os.path.abspath(_projects_dir()):
        raise LookupError("Project not found.")
    return path


def _write_json(path: str, payload: dict) -> None:
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=directory,
        delete=False,
        prefix=".frameforge-",
        suffix=".json",
    )
    try:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.close()
        os.replace(handle.name, path)
    finally:
        if os.path.exists(handle.name):
            os.remove(handle.name)


def available_fonts() -> list[dict[str, str]]:
    directory = utils.font_dir()
    found = []
    for root, _dirs, files in os.walk(directory):
        for name in files:
            if not name.lower().endswith((".ttf", ".ttc", ".otf")):
                continue
            relative = os.path.relpath(os.path.join(root, name), directory).replace("\\", "/")
            if ".." in relative:
                continue
            found.append({"id": relative, "label": relative})
    found.sort(key=lambda item: item["label"].lower())
    if found:
        return found
    return [
        {"id": filename, "label": label}
        for filename, label in FONT_CANDIDATES
        if os.path.isfile(os.path.join(directory, filename))
    ]


def resolve_font(preferred: str | None) -> str:
    fonts = available_fonts()
    ids = {item["id"] for item in fonts}
    if preferred in ids:
        return preferred or fonts[0]["id"]
    if fonts:
        return fonts[0]["id"]
    return "STHeitiMedium.ttc"


def voice_allowed(name: str) -> bool:
    if not name or len(name) > 240 or ".." in name or "\\" in name:
        return False
    return "\n" not in name and "\r" not in name


def narration_voices() -> list[dict[str, str]]:
    try:
        names = [item for item in voice_service.get_all_azure_voices() if "V2" not in item]
    except Exception:
        names = []
    if not names:
        return list(VOICES)
    return [{"id": name, "label": name, "detail": "Edge TTS"} for name in names]


def default_project_fields() -> dict[str, Any]:
    settings = load_settings()
    return {
        "aspect": settings["default_aspect"],
        "voice_name": settings["default_voice"],
        "voice_rate": settings["voice_rate"],
        "voice_volume": settings["voice_volume"],
        "bgm_volume": settings["bgm_volume"],
        "subtitle_enabled": settings["subtitle_enabled"],
        "subtitle_position": settings["subtitle_position"],
        "subtitle_display_mode": settings["subtitle_display_mode"],
        "subtitle_animation": settings["subtitle_animation"],
        "custom_position": settings["custom_position"],
        "font_name": resolve_font(settings.get("font_name")),
        "font_size": settings["font_size"],
        "text_fore_color": settings["text_fore_color"],
        "text_background_color": settings["text_background_color"],
        "use_caption_bar": settings["use_caption_bar"],
        "stroke_color": settings["stroke_color"],
        "stroke_width": settings["stroke_width"],
        "clip_duration": settings["clip_duration"],
        "video_clip_speed": settings["video_clip_speed"],
        "video_count": settings["video_count"],
        "video_source": settings["video_source"],
        "video_fit_mode": settings["video_fit_mode"],
        "video_concat_mode": settings["video_concat_mode"],
        "match_materials_to_script": settings["match_materials_to_script"],
        "transition": settings["transition"],
        "paragraph_number": settings["paragraph_number"],
        "video_language": settings["video_language"],
    }


def _empty_settings() -> dict[str, Any]:
    return {
        "openrouter_model_name": "",
        "llm_provider": "openrouter",
        "llm_base_url": "",
        "default_voice": "en-US-AriaNeural-Female",
        "default_aspect": "9:16",
        "font_name": resolve_font("BeVietnamPro-Bold.ttf"),
        "voice_rate": 1.0,
        "voice_volume": 1.0,
        "bgm_volume": 0.12,
        "subtitle_enabled": True,
        "subtitle_position": "bottom",
        "subtitle_display_mode": "sentence",
        "subtitle_animation": "none",
        "custom_position": 70.0,
        "font_size": 54,
        "text_fore_color": "#FFFFFF",
        "text_background_color": "#141210",
        "use_caption_bar": True,
        "stroke_color": "#141210",
        "stroke_width": 1.2,
        "clip_duration": 4,
        "video_clip_speed": 1.0,
        "video_count": 1,
        "video_source": "pexels",
        "video_fit_mode": "cover",
        "video_concat_mode": "sequential",
        "match_materials_to_script": True,
        "transition": "FadeIn",
        "paragraph_number": 1,
        "video_language": "English",
        "subtitle_provider": "edge",
        "video_codec": "__default__",
    }


def load_settings() -> dict[str, Any]:
    settings = _empty_settings()
    path = _settings_path()
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as handle:
                stored = json.load(handle)
            if isinstance(stored, dict):
                settings.update({key: stored.get(key, settings[key]) for key in settings})
        except (OSError, json.JSONDecodeError):
            logger.warning("Script-to-Reality Renderer Studio settings could not be read; using defaults.")
    if not voice_allowed(str(settings["default_voice"])):
        settings["default_voice"] = "en-US-AriaNeural-Female"
    if settings["default_aspect"] not in ASPECTS:
        settings["default_aspect"] = "9:16"
    if settings["video_source"] not in VIDEO_SOURCES:
        settings["video_source"] = "pexels"
    if settings["subtitle_provider"] not in SUBTITLE_PROVIDERS:
        settings["subtitle_provider"] = "edge"
    if settings["video_codec"] not in CODECS:
        settings["video_codec"] = "__default__"
    settings["font_name"] = resolve_font(str(settings.get("font_name") or ""))
    settings["openrouter_model_name"] = _limit(settings.get("openrouter_model_name"), 120)
    settings["llm_provider"] = str(settings.get("llm_provider") or "openrouter").lower()
    return settings


def _write_model_name(model_name: str) -> None:
    config.app["openrouter_model_name"] = model_name
    config.save_config()


def _persist_config() -> None:
    config.save_config()


def _set_secret_list(key: str, value: str) -> None:
    config.app[key] = [value]


def update_settings(patch: dict[str, Any]) -> dict[str, Any]:
    settings = load_settings()
    model_name = patch.get("openrouter_model_name", patch.get("model_name"))
    if model_name is not None:
        model_name = str(model_name).strip()
        if len(model_name) > 120 or any(char.isspace() for char in model_name):
            raise ValueError("Enter a model id such as openrouter/free.")
        if model_name.lower().startswith("sk-"):
            raise ValueError("That field is for a model id, not a key.")
    provider_id = str(patch.get("llm_provider") or settings["llm_provider"] or "openrouter").lower()
    if get_llm_provider(provider_id) is None:
        raise ValueError("Choose a supported language-model provider.")
    settings["llm_provider"] = provider_id
    config.app["llm_provider"] = provider_id

    if "default_voice" in patch and patch["default_voice"] is not None:
        voice_name = str(patch["default_voice"])
        if not voice_allowed(voice_name):
            raise ValueError("Choose a voice from the list.")
        settings["default_voice"] = voice_name
        config.ui["voice_name"] = voice_name
    if "default_aspect" in patch and patch["default_aspect"] is not None:
        aspect = str(patch["default_aspect"])
        if aspect not in ASPECTS:
            raise ValueError("Aspect must be 9:16, 16:9, or 1:1.")
        settings["default_aspect"] = aspect
    if "font_name" in patch and patch["font_name"] is not None:
        settings["font_name"] = resolve_font(str(patch["font_name"]))
        config.ui["font_name"] = settings["font_name"]
    choice_fields = {
        "video_source": VIDEO_SOURCES,
        "video_fit_mode": FIT_MODES,
        "video_concat_mode": CONCAT_MODES,
        "transition": TRANSITIONS,
        "subtitle_position": POSITIONS,
        "subtitle_display_mode": DISPLAY_MODES,
        "subtitle_animation": ANIMATIONS,
        "subtitle_provider": SUBTITLE_PROVIDERS,
        "video_codec": CODECS,
    }
    for field, allowed in choice_fields.items():
        if field in patch and patch[field] is not None:
            value = str(patch[field])
            if value not in allowed:
                raise ValueError(f"Unsupported value for {field}.")
            settings[field] = value
    if "video_language" in patch and patch["video_language"] is not None:
        settings["video_language"] = _limit(patch["video_language"], 40) or "English"
    if "llm_base_url" in patch and patch["llm_base_url"] is not None:
        base_url = str(patch["llm_base_url"]).strip()
        if len(base_url) > 300 or any(char.isspace() for char in base_url):
            raise ValueError("Enter a base URL without spaces.")
        settings["llm_base_url"] = base_url
        if base_url:
            config.app[f"{provider_id}_base_url"] = base_url
    for field, low, high, kind in (
        ("voice_rate", 0.8, 1.25, float),
        ("voice_volume", 0.5, 1.5, float),
        ("bgm_volume", 0.0, 0.4, float),
        ("stroke_width", 0.0, 10.0, float),
        ("custom_position", 0.0, 100.0, float),
        ("video_clip_speed", 0.5, 2.0, float),
        ("font_size", 30, 100, int),
        ("clip_duration", 2, 15, int),
        ("video_count", 1, 5, int),
        ("paragraph_number", 1, 10, int),
    ):
        if field in patch and patch[field] is not None:
            try:
                value = kind(patch[field])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{field} must be a number.") from exc
            settings[field] = min(max(value, low), high)
    for field in ("subtitle_enabled", "use_caption_bar", "match_materials_to_script"):
        if field in patch and patch[field] is not None:
            settings[field] = bool(patch[field])
    for field in ("text_fore_color", "text_background_color", "stroke_color"):
        if field in patch and patch[field] is not None:
            color = str(patch[field]).strip()
            if not _HEX.fullmatch(color):
                raise ValueError("Colors must be hex values such as #FFFFFF.")
            settings[field] = color.upper()
    if settings["video_source"] != "local":
        config.app["video_source"] = settings["video_source"]
    config.app["subtitle_provider"] = settings["subtitle_provider"]
    config.app["match_materials_to_script"] = settings["match_materials_to_script"]
    if settings["video_codec"] == "__default__":
        config.app.pop("video_codec", None)
    else:
        config.app["video_codec"] = settings["video_codec"]
    for ui_key in (
        "voice_rate", "voice_volume", "bgm_volume", "subtitle_enabled", "subtitle_position",
        "subtitle_display_mode", "subtitle_animation", "custom_position", "font_size",
        "text_fore_color", "stroke_color", "stroke_width", "video_clip_speed", "video_count",
        "video_fit_mode", "video_concat_mode", "paragraph_number",
    ):
        config.ui[ui_key] = settings[ui_key]
    config.ui["video_language"] = settings["video_language"]
    config.ui["video_transition_mode"] = None if settings["transition"] == "none" else settings["transition"]
    config.ui["subtitle_background_enabled"] = settings["use_caption_bar"]
    config.ui["subtitle_background_color"] = settings["text_background_color"]

    for secret_field, config_key, as_list in (
        ("pexels_secret", "pexels_api_keys", True),
        ("pixabay_secret", "pixabay_api_keys", True),
        ("coverr_secret", "coverr_api_keys", True),
    ):
        raw = patch.get(secret_field)
        if isinstance(raw, str) and raw.strip():
            if as_list:
                _set_secret_list(config_key, raw.strip())
            else:
                config.app[config_key] = raw.strip()
    llm_secret = patch.get("llm_secret")
    if isinstance(llm_secret, str) and llm_secret.strip():
        config.app[f"{provider_id}_api_key"] = llm_secret.strip()

    if isinstance(model_name, str):
        settings["openrouter_model_name"] = model_name
        if provider_id == "openrouter":
            _write_model_name(model_name)
        else:
            config.app[f"{provider_id}_model_name"] = model_name
    from app.services import dashboard

    dashboard.apply_config(patch)
    _write_json(_settings_path(), settings)
    _persist_config()
    return public_settings()


def _configured_secret(config_value: Any, environment_name: str) -> bool:
    if isinstance(config_value, str) and config_value.strip():
        return True
    if isinstance(config_value, (list, tuple)) and any(str(item).strip() for item in config_value):
        return True
    return bool(os.getenv(environment_name, "").strip())


def public_settings() -> dict[str, Any]:
    """Return dashboard defaults and whether credentials exist. Never return secrets."""
    settings = load_settings()
    provider_id = str(config.app.get("llm_provider", settings["llm_provider"]) or "openrouter").lower()
    provider = get_llm_provider(provider_id) or get_llm_provider("openrouter")
    provider_id = provider.provider_id if provider else provider_id
    configured_model = str(config.app.get(f"{provider_id}_model_name", "") or "").strip()
    active_model = settings["openrouter_model_name"] or configured_model
    resolved = provider.resolve_model_name(active_model) if provider else active_model
    configured_base = str(config.app.get(f"{provider_id}_base_url", "") or "").strip()
    ffmpeg_path = str(config.app.get("ffmpeg_path", "") or "").strip()
    safe = {key: value for key, value in settings.items() if key != "openrouter_model_name"}
    safe.update({
        "product": "Script-to-Reality Renderer Studio",
        "llm_provider": provider_id,
        "model_name": resolved or "openrouter/free",
        "model_override": settings["openrouter_model_name"] or configured_model,
        "llm_base_url": settings["llm_base_url"] or configured_base,
        "llm_secret_set": _configured_secret(config.app.get(f"{provider_id}_api_key", ""), "OPENROUTER_API_KEY" if provider_id == "openrouter" else ""),
        "openrouter_configured": _configured_secret(config.app.get("openrouter_api_key", ""), "OPENROUTER_API_KEY"),
        "pexels_configured": _configured_secret(config.app.get("pexels_api_keys", []), "PEXELS_API_KEY"),
        "pixabay_configured": _configured_secret(config.app.get("pixabay_api_keys", []), "PIXABAY_API_KEY"),
        "coverr_configured": _configured_secret(config.app.get("coverr_api_keys", []), "COVERR_API_KEY"),
        "video_source": settings["video_source"] if settings["video_source"] in VIDEO_SOURCES else str(config.app.get("video_source", "pexels") or "pexels"),
        "subtitle_provider": settings["subtitle_provider"],
        "ffmpeg_ready": bool(ffmpeg_path and os.path.isfile(ffmpeg_path)),
        "providers": [
            {"id": item.provider_id, "label": item.default_label, "show_base_url": item.show_base_url, "show_secret": item.show_api_key}
            for item in LLM_PROVIDER_REGISTRY
        ],
        "sources": list(VIDEO_SOURCES),
        "voices": narration_voices(),
        "fonts": available_fonts(),
    })
    from app.services import dashboard

    safe.update(dashboard.public_flags())
    return safe


def _normalize_clip(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    url = str(value.get("url") or "").strip()
    if not url:
        return None
    if not clip_url_allowed(url):
        raise ValueError("Only Pexels clip links can be saved on a scene.")
    try:
        duration = int(value.get("duration") or 0)
    except (TypeError, ValueError):
        duration = 0
    duration = min(max(duration, 0), 600)
    return {
        "provider": "pexels",
        "url": url,
        "thumbnail": str(value.get("thumbnail") or "").strip() if clip_url_allowed(str(value.get("thumbnail") or "")) else "",
        "duration": duration,
        "page": str(value.get("page") or "").strip() if clip_url_allowed(str(value.get("page") or "")) else "",
        "creator": _limit(value.get("creator"), 80),
        "asset_id": _limit(value.get("asset_id"), 40),
    }


def normalize_scenes(scenes: Any) -> list[dict[str, Any]]:
    if not isinstance(scenes, list):
        raise ValueError("Scenes must be a list.")
    cleaned = []
    for raw in scenes[:8]:
        if not isinstance(raw, dict):
            continue
        scene_id = str(raw.get("id") or "")
        if not _PROJECT_ID.fullmatch(scene_id):
            scene_id = str(uuid4())
        cleaned.append({
            "id": scene_id,
            "narration": _limit(raw.get("narration"), 600),
            "term": _limit(raw.get("term"), 80),
            "clip": _normalize_clip(raw.get("clip")),
        })
    return cleaned


def scenes_from_script(script: str, previous: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    narrations = split_narration(script)
    previous_clips = {
        item.get("narration"): item.get("clip")
        for item in (previous or [])
        if isinstance(item, dict)
    }
    return [
        {
            "id": str(uuid4()),
            "narration": narration,
            "term": "",
            "clip": previous_clips.get(narration),
        }
        for narration in narrations
    ]


def _blank_project(topic: str, title: str = "") -> dict[str, Any]:
    now = _now()
    project = {
        "id": str(uuid4()),
        "title": _limit(title, 120) or _limit(topic, 120),
        "topic": _limit(topic, 500),
        "script": "",
        "scenes": [],
        "task_id": "",
        "notice": "",
        "created_at": now,
        "updated_at": now,
    }
    project.update(default_project_fields())
    return project


def _read_project(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not _PROJECT_ID.fullmatch(str(payload.get("id") or "")):
        raise LookupError("Project not found.")
    return payload


def list_projects() -> list[dict[str, Any]]:
    projects = []
    for name in os.listdir(_projects_dir()):
        if not name.endswith(".json"):
            continue
        try:
            project = _read_project(os.path.join(_projects_dir(), name))
        except (OSError, json.JSONDecodeError, LookupError):
            continue
        projects.append(_summarize(project))
    projects.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
    return projects


def get_project(project_id: str) -> dict[str, Any]:
    path = _project_path(project_id)
    if not os.path.isfile(path):
        raise LookupError("Project not found.")
    try:
        return _hydrate_project(_read_project(path))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(f"Script-to-Reality Renderer Studio project could not be read: {type(exc).__name__}")
        raise LookupError("Project not found.") from exc


def save_project(project: dict[str, Any]) -> dict[str, Any]:
    project["updated_at"] = _now()
    _write_json(_project_path(project["id"]), project)
    return project


def _hydrate_project(project: dict[str, Any]) -> dict[str, Any]:
    for key, value in default_project_fields().items():
        project.setdefault(key, value)
    return project


def create_project(topic: str, title: str = "") -> dict[str, Any]:
    topic = _limit(topic, 500)
    if not topic:
        raise ValueError("Enter a topic before starting a project.")
    return save_project(_blank_project(topic, title))


def apply_project_patch(project: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    if "title" in patch and patch["title"] is not None:
        project["title"] = _limit(patch["title"], 120) or project["title"]
    if "topic" in patch and patch["topic"] is not None:
        topic = _limit(patch["topic"], 500)
        if topic:
            project["topic"] = topic
    if "script" in patch and patch["script"] is not None:
        project["script"] = str(patch["script"] or "")[:8000].strip()
    if "aspect" in patch and patch["aspect"] is not None:
        if str(patch["aspect"]) not in ASPECTS:
            raise ValueError("Aspect must be 9:16, 16:9, or 1:1.")
        project["aspect"] = str(patch["aspect"])
    if "voice_name" in patch and patch["voice_name"] is not None:
        if not voice_allowed(str(patch["voice_name"])):
            raise ValueError("Choose a voice from the list.")
        project["voice_name"] = str(patch["voice_name"])
    for field, low, high in (
        ("voice_rate", 0.8, 1.25),
        ("voice_volume", 0.5, 1.5),
        ("bgm_volume", 0.0, 0.4),
        ("stroke_width", 0.0, 10.0),
        ("custom_position", 0.0, 100.0),
        ("video_clip_speed", 0.5, 2.0),
    ):
        if field in patch and patch[field] is not None:
            try:
                value = float(patch[field])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{field} must be a number.") from exc
            project[field] = min(max(value, low), high)
    if "subtitle_enabled" in patch and patch["subtitle_enabled"] is not None:
        project["subtitle_enabled"] = bool(patch["subtitle_enabled"])
    if "use_caption_bar" in patch and patch["use_caption_bar"] is not None:
        project["use_caption_bar"] = bool(patch["use_caption_bar"])
    if "match_materials_to_script" in patch and patch["match_materials_to_script"] is not None:
        project["match_materials_to_script"] = bool(patch["match_materials_to_script"])
    if "subtitle_position" in patch and patch["subtitle_position"] is not None:
        if str(patch["subtitle_position"]) not in POSITIONS:
            raise ValueError("Choose a caption position from the list.")
        project["subtitle_position"] = str(patch["subtitle_position"])
    for field, allowed in (
        ("subtitle_display_mode", DISPLAY_MODES),
        ("subtitle_animation", ANIMATIONS),
        ("video_source", VIDEO_SOURCES),
        ("video_fit_mode", FIT_MODES),
        ("video_concat_mode", CONCAT_MODES),
    ):
        if field in patch and patch[field] is not None:
            value = str(patch[field])
            if value not in allowed:
                raise ValueError(f"Unsupported value for {field}.")
            project[field] = value
    if "font_name" in patch and patch["font_name"] is not None:
        project["font_name"] = resolve_font(str(patch["font_name"]))
    if "font_size" in patch and patch["font_size"] is not None:
        try:
            project["font_size"] = min(max(int(patch["font_size"]), 30), 100)
        except (TypeError, ValueError) as exc:
            raise ValueError("Caption size must be a number.") from exc
    for field in ("text_fore_color", "text_background_color", "stroke_color"):
        if field in patch and patch[field] is not None:
            color = str(patch[field]).strip()
            if not _HEX.fullmatch(color):
                raise ValueError("Colors must be hex values such as #FFFFFF.")
            project[field] = color.upper()
    if "clip_duration" in patch and patch["clip_duration"] is not None:
        try:
            project["clip_duration"] = min(max(int(patch["clip_duration"]), 2), 15)
        except (TypeError, ValueError) as exc:
            raise ValueError("Clip length must be a number of seconds.") from exc
    if "transition" in patch and patch["transition"] is not None:
        if str(patch["transition"]) not in TRANSITIONS:
            raise ValueError("Choose a supported transition.")
        project["transition"] = str(patch["transition"])
    if "video_count" in patch and patch["video_count"] is not None:
        try:
            project["video_count"] = min(max(int(patch["video_count"]), 1), 5)
        except (TypeError, ValueError) as exc:
            raise ValueError("Video count must be a number.") from exc
    if "paragraph_number" in patch and patch["paragraph_number"] is not None:
        try:
            project["paragraph_number"] = min(max(int(patch["paragraph_number"]), 1), 10)
        except (TypeError, ValueError) as exc:
            raise ValueError("Paragraph count must be a number.") from exc
    if "video_language" in patch and patch["video_language"] is not None:
        project["video_language"] = _limit(patch["video_language"], 40) or "English"
    if "scenes" in patch and patch["scenes"] is not None:
        project["scenes"] = normalize_scenes(patch["scenes"])
    return save_project(project)


def generate_project_script(project: dict[str, Any]) -> dict[str, Any]:
    try:
        script = llm.generate_script(
            video_subject=project["topic"] or project["title"],
            language=project.get("video_language") or "English",
            paragraph_number=int(project.get("paragraph_number") or 1),
            video_script_prompt=SCRIPT_PROMPT,
        )
    except Exception as exc:
        raise RuntimeError(public_error(str(exc))) from exc
    if not isinstance(script, str) or not script.strip() or script.startswith("Error:"):
        detail = script.removeprefix("Error:").strip() if isinstance(script, str) else ""
        raise RuntimeError(public_error(detail or "Script writing is unavailable."))
    project["script"] = script.strip()[:8000]
    project["notice"] = ""
    return save_project(project)


def build_storyboard(project: dict[str, Any]) -> dict[str, Any]:
    script = (project.get("script") or "").strip()
    if not script:
        raise ValueError("Write a script before building the storyboard.")
    scenes = scenes_from_script(script, project.get("scenes"))
    narrations = [scene["narration"] for scene in scenes]
    terms: list[str] = []
    notice = ""
    try:
        generated = llm.generate_terms(
            video_subject=project["topic"] or project["title"],
            video_script=script,
            amount=len(narrations),
            match_script_order=True,
        )
        if isinstance(generated, list):
            terms = [term for term in generated if isinstance(term, str)]
        elif isinstance(generated, str) and generated.startswith("Error:"):
            notice = "OpenRouter did not return search terms. Each scene uses words from its narration."
    except Exception:
        notice = "OpenRouter did not return search terms. Each scene uses words from its narration."
    if not terms and not notice:
        notice = "OpenRouter did not return search terms. Each scene uses words from its narration."
    for scene, term in zip(scenes, assign_terms(narrations, terms)):
        scene["term"] = term
    project["scenes"] = scenes
    project["notice"] = notice
    return save_project(project)


def search_scene_clips(project: dict[str, Any], scene_id: str, term: str = "") -> list[dict[str, Any]]:
    scene = next((item for item in project.get("scenes") or [] if item.get("id") == scene_id), None)
    if scene is None:
        raise LookupError("Scene not found.")
    query = _limit(term or scene.get("term") or fallback_term(scene.get("narration") or ""), 80)
    if not query:
        raise ValueError("Add a search term before looking for footage.")
    if term:
        scene["term"] = query
        save_project(project)
    try:
        return _search_pexels_previews(
            query,
            VideoAspect(project.get("aspect") or "9:16"),
            int(project.get("clip_duration") or 4),
        )
    except ValueError:
        raise
    except Exception as exc:
        raise RuntimeError(public_error(str(exc))) from exc


def _search_pexels_previews(term: str, aspect: VideoAspect, minimum_duration: int) -> list[dict[str, Any]]:
    try:
        api_key = material.get_api_key("pexels_api_keys")
    except ValueError as exc:
        raise ValueError(_stock_configuration_message(exc)) from exc
    params = {"query": term, "per_page": 12}
    if aspect != VideoAspect.square:
        params["orientation"] = aspect.name

    try:
        response = requests.get(
            f"https://api.pexels.com/v1/videos/search?{urlencode(params)}",
            headers={"Authorization": api_key},
            proxies=config.proxy,
            timeout=(20, 40),
        )
        payload = response.json()
    except Exception as exc:
        logger.warning(f"Script-to-Reality Renderer Studio stock search failed: {type(exc).__name__}")
        raise RuntimeError("Stock search is unavailable right now.") from exc
    videos = payload.get("videos") if isinstance(payload, dict) else None
    if not isinstance(videos, list):
        raise RuntimeError("Stock search returned an unexpected response.")
    width, height = aspect.to_resolution()
    previews = []
    for video in videos:
        if not isinstance(video, dict) or len(previews) >= 8:
            continue
        duration = video.get("duration")
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration < minimum_duration:
            continue
        files = video.get("video_files")
        if not isinstance(files, list):
            continue
        candidates = []
        for rendition in files:
            if not isinstance(rendition, dict):
                continue
            link = rendition.get("link")
            if not isinstance(link, str) or not clip_url_allowed(link):
                continue
            try:
                rendition_width = int(rendition.get("width"))
                rendition_height = int(rendition.get("height"))
            except (TypeError, ValueError):
                continue
            if rendition_width >= width and rendition_height >= height:
                candidates.append((rendition_width * rendition_height, link))
        if not candidates:
            continue
        _, link = min(candidates, key=lambda item: item[0])
        user = video.get("user") if isinstance(video.get("user"), dict) else {}
        thumbnail = video.get("image") if isinstance(video.get("image"), str) else ""
        page = video.get("url") if isinstance(video.get("url"), str) else ""
        previews.append({
            "provider": "pexels",
            "url": link,
            "thumbnail": thumbnail if clip_url_allowed(thumbnail) else "",
            "duration": int(duration),
            "page": page if clip_url_allowed(page) else "",
            "creator": _limit(user.get("name"), 80),
            "asset_id": _limit(video.get("id"), 40),
        })
    return previews


def spoken_script(project: dict[str, Any]) -> str:
    narrations = [
        scene["narration"].strip()
        for scene in project.get("scenes") or []
        if isinstance(scene, dict) and scene.get("narration", "").strip()
    ]
    if narrations:
        return "\n\n".join(narrations)
    return (project.get("script") or "").strip()


def ensure_renderable(project: dict[str, Any]) -> dict[str, Any]:
    if not project.get("scenes"):
        script = (project.get("script") or "").strip()
        if not script:
            raise ValueError("Write a script or add a scene before rendering.")
        project["scenes"] = scenes_from_script(script, [])
    ready = []
    for scene in project["scenes"]:
        if not scene.get("narration", "").strip():
            continue
        if not scene.get("term"):
            scene["term"] = fallback_term(scene["narration"])
        ready.append(scene)
    if not ready:
        raise ValueError("Add narration to at least one scene before rendering.")
    project["scenes"] = ready
    project["script"] = spoken_script(project)
    return project


def uses_locked_footage(project: dict[str, Any]) -> bool:
    return any(isinstance(scene.get("clip"), dict) and scene["clip"].get("url") for scene in project.get("scenes") or [])


def assemble_local_materials(project: dict[str, Any]) -> list[MaterialInfo]:
    save_dir = utils.storage_dir("local_videos", create=True)
    aspect = VideoAspect(project.get("aspect") or "9:16")
    minimum = int(project.get("clip_duration") or 4)
    materials = []
    for index, scene in enumerate(project["scenes"], start=1):
        clip = scene.get("clip") if isinstance(scene.get("clip"), dict) else None
        if clip and clip.get("url"):
            path = material.save_video(clip["url"], save_dir)
            duration = int(clip.get("duration") or minimum)
        else:
            try:
                found = material.search_videos_pexels(scene["term"], minimum, aspect)
            except ValueError as exc:
                raise ValueError(_stock_configuration_message(exc)) from exc
            if not found:
                raise ValueError(f"No stock clip matched scene {index} ({scene['term']}).")
            path = material.save_video(found[0].url, save_dir)
            duration = int(found[0].duration or minimum)
        if not path or not os.path.isfile(path):
            raise ValueError(f"Could not download footage for scene {index}.")
        materials.append(MaterialInfo(provider="local", url=path, duration=duration))
    return materials


def build_task_request(project: dict[str, Any], materials: list[MaterialInfo] | None = None) -> TaskVideoRequest:
    project = ensure_renderable(project)
    terms = [scene["term"] for scene in project["scenes"]]
    transition = None if project.get("transition") == "none" else project.get("transition") or "FadeIn"
    background = project.get("text_background_color") or "#141210"
    if not project.get("use_caption_bar", True):
        background = False
    match_order = bool(project.get("match_materials_to_script", True))
    source = "local" if materials else str(project.get("video_source") or "pexels")
    if source not in VIDEO_SOURCES:
        source = "pexels"
    return TaskVideoRequest(
        video_subject=project["topic"] or project["title"] or "Script-to-Reality Renderer Studio video",
        video_script=project["script"],
        video_terms=terms,
        video_aspect=project.get("aspect") or "9:16",
        video_concat_mode="sequential" if match_order else project.get("video_concat_mode") or "sequential",
        video_transition_mode=transition,
        video_clip_duration=int(project.get("clip_duration") or 4),
        video_clip_speed=float(project.get("video_clip_speed") or 1),
        video_fit_mode=project.get("video_fit_mode") or "cover",
        match_materials_to_script=match_order,
        video_count=min(max(int(project.get("video_count") or 1), 1), 5),
        video_source=source,
        video_materials=materials,
        video_language=project.get("video_language") or "English",
        voice_name=project.get("voice_name") or "en-US-AriaNeural-Female",
        voice_volume=float(project.get("voice_volume") or 1),
        voice_rate=float(project.get("voice_rate") or 1),
        bgm_type="random" if float(project.get("bgm_volume") or 0) > 0 else "",
        bgm_volume=float(project.get("bgm_volume") or 0),
        subtitle_enabled=bool(project.get("subtitle_enabled", True)),
        subtitle_position=project.get("subtitle_position") or "bottom",
        subtitle_display_mode=project.get("subtitle_display_mode") or "sentence",
        subtitle_animation=project.get("subtitle_animation") or "none",
        custom_position=float(project.get("custom_position") or 70),
        font_name=resolve_font(project.get("font_name")),
        text_fore_color=project.get("text_fore_color") or "#FFFFFF",
        text_background_color=background,
        rounded_subtitle_background=bool(project.get("use_caption_bar")),
        font_size=int(project.get("font_size") or 54),
        stroke_color=project.get("stroke_color") or "#141210",
        stroke_width=float(project.get("stroke_width") or 1.2),
        n_threads=2,
        paragraph_number=min(10, max(1, len(project["scenes"]))),
    )


def remember_task(project: dict[str, Any], task_id: str) -> dict[str, Any]:
    project["task_id"] = task_id
    project["notice"] = ""
    return save_project(project)


def _video_for_task(task_id: str) -> str:
    if not _PROJECT_ID.fullmatch(task_id or ""):
        return ""
    folder = os.path.join(utils.task_dir(), task_id)
    if not os.path.isdir(folder):
        return ""
    finals = sorted(
        name for name in os.listdir(folder)
        if re.fullmatch(r"final-\d+\.mp4", name) and os.path.isfile(os.path.join(folder, name))
    )
    if not finals:
        return ""
    return f"/tasks/{task_id}/{finals[-1]}"


def _summarize(project: dict[str, Any]) -> dict[str, Any]:
    task_id = str(project.get("task_id") or "")
    video = _video_for_task(task_id)
    status = "draft"
    progress = 0
    if task_id:
        task = None
        try:
            task = sm.state.get_task(task_id)
        except Exception:
            task = None
        if task and task.get("state") == 4:
            status = "rendering"
            progress = int(task.get("progress") or 0)
        elif task and task.get("state") == -1 and not video:
            status = "failed"
        elif video:
            status = "ready"
    elif video:
        status = "ready"
    return {
        "id": project["id"],
        "title": project.get("title") or "Untitled reel",
        "topic": project.get("topic") or "",
        "scene_count": len(project.get("scenes") or []),
        "status": status,
        "progress": progress,
        "video": video,
        "updated_at": project.get("updated_at") or "",
        "task_id": task_id,
    }


def delete_project(project_id: str) -> None:
    path = _project_path(project_id)
    if not os.path.isfile(path):
        raise LookupError("Project not found.")
    os.remove(path)


def list_library() -> list[dict[str, Any]]:
    root = utils.task_dir()
    renders = []
    try:
        names = os.listdir(root)
    except OSError:
        return []
    for name in names:
        if not _PROJECT_ID.fullmatch(name):
            continue
        video = _video_for_task(name)
        if not video:
            continue
        folder = os.path.join(root, name)
        title = "Rendered reel"
        script = ""
        record_path = os.path.join(folder, "script.json")
        try:
            with open(record_path, encoding="utf-8") as handle:
                record = json.load(handle)
            if isinstance(record, dict):
                params = record.get("params") if isinstance(record.get("params"), dict) else {}
                title = _limit(params.get("video_subject"), 120) or title
                script = _limit(record.get("script"), 220)
        except (OSError, json.JSONDecodeError):
            pass
        video_name = video.rsplit("/", 1)[-1]
        renders.append({
            "id": name,
            "title": title,
            "script": script,
            "video": video,
            "updated_at": datetime.fromtimestamp(
                os.path.getmtime(os.path.join(folder, video_name)),
                timezone.utc,
            ).replace(microsecond=0).isoformat(),
        })
    renders.sort(key=lambda item: item["updated_at"], reverse=True)
    return renders[:100]
