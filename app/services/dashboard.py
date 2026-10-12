"""Controls the Streamlit dashboard exposed, without returning stored secrets."""

from __future__ import annotations

import os
import shutil
from typing import Any
from uuid import uuid4

from app.config import config
from app.services import bgm as bgm_service
from app.services import cache_manager
from app.services import llm
from app.services import voice as voice_service
from app.services import state as sm
from app.services import webui_task
from app.utils import utils

LANGUAGES = [
    "",
    "English",
    "ca-ES",
    "zh-CN",
    "zh-HK",
    "zh-TW",
    "de-DE",
    "en-US",
    "es-ES",
    "fr-FR",
    "it-IT",
    "ru-RU",
    "vi-VN",
    "th-TH",
    "tr-TR",
    "ja-JP",
    "ko-KR",
    "pt-BR",
    "hi-IN",
    "ar-SA",
]

TTS_SERVERS = [
    ("azure-tts-v1", "Azure TTS V1 (Edge TTS)"),
    ("azure-tts-v2", "Azure TTS V2"),
    ("siliconflow", "SiliconFlow TTS"),
    ("gemini-tts", "Google Gemini TTS"),
    ("mimo-tts", "Xiaomi MiMo TTS"),
    ("minimax-tts", "MiniMax TTS"),
    ("elevenlabs", "ElevenLabs TTS"),
    ("chatterbox", "Chatterbox TTS"),
    ("kokoro", "Kokoro TTS"),
    ("fish_audio", "Fish Audio TTS"),
    ("voxcpm", "VoxCPM TTS"),
]

BGM_TYPES = [
    ("", "No background music"),
    ("random", "Random"),
    ("preset", "Preset song"),
    ("custom", "Custom file"),
    ("sonilo", "Sonilo"),
    ("elevenlabs", "ElevenLabs"),
]

UI_LOCALES = [
    "en", "zh", "de", "es", "fr", "pt", "ru", "ko", "vi", "tr", "it", "id", "ca", "az",
]

# Request field -> where the secret is stored. Names here never go back to the browser.
_APP_LIST_SECRETS = {
    "pexels_secret": "pexels_api_keys",
    "pixabay_secret": "pixabay_api_keys",
    "coverr_secret": "coverr_api_keys",
    "wavespeed_secret": "wavespeed_api_keys",
    "openai_image_secret": "openai_image_api_keys",
    "twelvelabs_secret": "twelvelabs_api_keys",
}
_APP_TEXT_SECRETS = {
    "seedance_secret": "volcengine_seedance_api_key",
    "ofox_secret": "ofox_api_key",
    "muapi_secret": "muapi_api_key",
    "metaso_secret": "metaso_minimax_api_key",
    "sonilo_secret": "sonilo_api_key",
    "upload_post_secret": "upload_post_api_key",
    "gemini_secret": "gemini_api_key",
    "mimo_secret": "mimo_api_key",
    "loomloom_secret": "loomloom_api_token",
}
_SECTION_SECRETS = {
    "siliconflow_secret": ("siliconflow", "api_key"),
    "elevenlabs_secret": ("elevenlabs", "api_key"),
    "minimax_tts_secret": ("minimax_tts", "api_key"),
    "fish_secret": ("fish_audio", "api_key"),
    "voxcpm_secret": ("voxcpm", "api_key"),
    "chatterbox_secret": ("chatterbox", "api_key"),
    "kokoro_secret": ("kokoro", "api_key"),
    "azure_speech_secret": ("azure", "speech_key"),
}

_TEXT_FIELDS = {
    "azure_speech_region": ("azure", "speech_region", 40),
    "siliconflow_base_url": ("siliconflow", "base_url", 300),
    "minimax_tts_base_url": ("minimax_tts", "base_url", 300),
    "minimax_tts_model": ("minimax_tts", "model_id", 80),
    "elevenlabs_model": ("elevenlabs", "model_id", 80),
    "elevenlabs_music_model": ("elevenlabs", "music_model_id", 80),
    "chatterbox_base_url": ("chatterbox", "base_url", 300),
    "kokoro_base_url": ("kokoro", "base_url", 300),
    "fish_model": ("fish_audio", "model", 80),
    "voxcpm_base_url": ("voxcpm", "base_url", 300),
    "voxcpm_model": ("voxcpm", "model_id", 80),
    "voxcpm_voice": ("voxcpm", "voice_id", 80),
    "seedance_base_url": ("app", "volcengine_seedance_base_url", 300),
    "ofox_base_url": ("app", "ofox_base_url", 300),
    "ofox_provider": ("app", "ofox_provider", 40),
    "muapi_base_url": ("app", "muapi_base_url", 300),
    "metaso_base_url": ("app", "metaso_minimax_base_url", 300),
    "openai_image_base_url": ("app", "openai_image_base_url", 300),
    "sonilo_base_url": ("app", "sonilo_base_url", 300),
    "wavespeed_model": ("app", "wavespeed_text_to_video_model", 180),
    "seedance_model": ("app", "volcengine_seedance_model", 180),
    "seedance_resolution": ("app", "volcengine_seedance_resolution", 20),
    "ofox_model": ("app", "ofox_text_to_video_model", 180),
    "ofox_resolution": ("app", "ofox_resolution", 20),
    "muapi_endpoint": ("app", "muapi_video_endpoint", 180),
    "muapi_resolution": ("app", "muapi_resolution", 20),
    "metaso_resolution": ("app", "metaso_minimax_resolution", 20),
    "openai_image_model": ("app", "openai_image_model", 120),
    "openai_image_size": ("app", "openai_image_size", 40),
    "openai_image_prompt_template": ("app", "openai_image_prompt_template", 500),
    "loomloom_base_url": ("app", "loomloom_base_url", 300),
    "openrouter_video_model": ("app", "openrouter_video_model", 120),
    "openrouter_video_resolution": ("app", "openrouter_video_resolution", 12),
    "upload_post_username": ("app", "upload_post_username", 80),
    "upload_post_youtube_privacy": ("app", "upload_post_youtube_privacy_status", 20),
    "script_generation_backend": ("app", "script_generation_backend", 20),
    "ui_language": ("ui", "language", 12),
}


def _section(name: str):
    if name == "app":
        return config.app
    if name == "ui":
        return config.ui
    return getattr(config, name)


def _filled(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple)):
        return any(str(item).strip() for item in value)
    return bool(value)


def _limit(value: Any, length: int) -> str:
    return " ".join(str(value or "").split())[:length].strip()


def catalog() -> dict[str, Any]:
    songs = []
    try:
        songs = [os.path.basename(path) for path in bgm_service.list_builtin_bgm_files()]
    except Exception:
        songs = []
    uploads = []
    try:
        uploads = [os.path.basename(path) for path in bgm_service.list_bgm_files()]
    except Exception:
        uploads = []
    return {
        "languages": LANGUAGES,
        "tts_servers": [{"id": item_id, "label": label} for item_id, label in TTS_SERVERS],
        "bgm_types": [{"id": item_id, "label": label} for item_id, label in BGM_TYPES],
        "songs": songs,
        "music_files": uploads,
        "ui_locales": UI_LOCALES,
        "default_system_prompt": llm.DEFAULT_SCRIPT_SYSTEM_PROMPT.strip(),
        "script_backends": [
            {"id": "local", "label": "This computer"},
            {"id": "loomloom", "label": "Shengsuan Cloud batch"},
        ],
    }


def voices_for(server: str) -> list[dict[str, str]]:
    server = (server or "azure-tts-v1").strip()
    names: list[str] = []
    try:
        if server == "siliconflow":
            names = voice_service.get_siliconflow_voices()
        elif server == "gemini-tts":
            names = voice_service.get_gemini_voices()
        elif server == "mimo-tts":
            names = voice_service.get_mimo_voices()
        elif server == "minimax-tts":
            names = voice_service.get_minimax_voices()
        elif server == "elevenlabs":
            names = voice_service.get_elevenlabs_voices(voice_service.get_elevenlabs_api_key())
        elif server == "chatterbox":
            names = voice_service.get_chatterbox_voices()
        elif server == "kokoro":
            names = voice_service.get_kokoro_voices()
        elif server == "fish_audio":
            names = voice_service.get_fish_audio_voices()
        elif server == "voxcpm":
            names = voice_service.get_voxcpm_voices()
        else:
            names = [
                name for name in voice_service.get_all_azure_voices()
                if ("V2" in name) == (server == "azure-tts-v2")
            ]
    except Exception:
        names = []
    return [{"id": name, "label": name} for name in names if name]


def prompt_preview(body: dict[str, Any]) -> str:
    return llm.build_script_prompt(
        video_subject=str(body.get("video_subject") or ""),
        language=str(body.get("video_language") or ""),
        paragraph_number=int(body.get("paragraph_number") or 1),
        video_script_prompt=str(body.get("video_script_prompt") or ""),
        custom_system_prompt=str(body.get("custom_system_prompt") or ""),
    )


def public_flags() -> dict[str, Any]:
    """Non-secret dashboard values. Keys must not contain the text api_key."""
    flags = {
        "wavespeed_configured": _filled(config.app.get("wavespeed_api_keys", [])),
        "seedance_configured": _filled(config.app.get("volcengine_seedance_api_key", "")),
        "ofox_configured": _filled(config.app.get("ofox_api_key", "")),
        "muapi_configured": _filled(config.app.get("muapi_api_key", "")),
        "metaso_configured": _filled(config.app.get("metaso_minimax_api_key", "")),
        "openai_image_configured": _filled(config.app.get("openai_image_api_keys", [])),
        "twelvelabs_configured": _filled(config.app.get("twelvelabs_api_keys", [])),
        "sonilo_configured": _filled(config.app.get("sonilo_api_key", "")),
        "upload_post_configured": _filled(config.app.get("upload_post_api_key", "")),
        "gemini_configured": _filled(config.app.get("gemini_api_key", "")),
        "mimo_configured": _filled(config.app.get("mimo_api_key", "")),
        "siliconflow_configured": _filled(_section("siliconflow").get("api_key", "")),
        "elevenlabs_configured": _filled(_section("elevenlabs").get("api_key", "")),
        "minimax_tts_configured": _filled(_section("minimax_tts").get("api_key", "")),
        "fish_configured": _filled(_section("fish_audio").get("api_key", "")),
        "voxcpm_configured": _filled(_section("voxcpm").get("api_key", "")),
        "chatterbox_configured": _filled(_section("chatterbox").get("api_key", "")),
        "kokoro_configured": _filled(_section("kokoro").get("api_key", "")),
        "loomloom_configured": _filled(config.app.get("loomloom_api_token", "")),
        "azure_speech_configured": _filled(_section("azure").get("speech_key", "")),
        "upload_post_enabled": bool(config.app.get("upload_post_enabled", False)),
        "upload_post_auto_upload": bool(config.app.get("upload_post_auto_upload", False)),
        "upload_post_username": str(config.app.get("upload_post_username", "") or ""),
        "upload_post_platforms": list(config.app.get("upload_post_platforms", ["tiktok", "instagram"]) or []),
        "upload_post_youtube_privacy": str(config.app.get("upload_post_youtube_privacy_status", "public") or "public"),
        "upload_post_youtube_made_for_kids": bool(config.app.get("upload_post_youtube_made_for_kids", False)),
        "hide_log": bool(config.ui.get("hide_log", False)),
        "open_task_folder_on_completion": bool(config.ui.get("open_task_folder_on_completion", True)),
        "ui_language": str(config.ui.get("language", "") or "en"),
        "script_generation_backend": str(config.app.get("script_generation_backend", "local") or "local"),
        "tts_server": str(config.ui.get("tts_server", "azure-tts-v1") or "azure-tts-v1"),
        "bgm_type": str(config.ui.get("bgm_type", "random") or ""),
        "bgm_file": str(config.ui.get("bgm_file", "") or ""),
    }
    for field, (section, key, _length) in _TEXT_FIELDS.items():
        if field in flags:
            continue
        flags[field] = str(_section(section).get(key, "") or "")
    return flags


def apply_config(patch: dict[str, Any]) -> None:
    for field, key in _APP_LIST_SECRETS.items():
        raw = patch.get(field)
        if isinstance(raw, str) and raw.strip():
            config.app[key] = [raw.strip()]
    for field, key in _APP_TEXT_SECRETS.items():
        raw = patch.get(field)
        if isinstance(raw, str) and raw.strip():
            config.app[key] = raw.strip()
    for field, (section, key) in _SECTION_SECRETS.items():
        raw = patch.get(field)
        if isinstance(raw, str) and raw.strip():
            _section(section)[key] = raw.strip()
    for field, (section, key, length) in _TEXT_FIELDS.items():
        if field not in patch or patch[field] is None:
            continue
        _section(section)[key] = _limit(patch[field], length)
    if "upload_post_enabled" in patch:
        config.app["upload_post_enabled"] = bool(patch["upload_post_enabled"])
    if "upload_post_auto_upload" in patch:
        config.app["upload_post_auto_upload"] = bool(patch["upload_post_auto_upload"])
    if "upload_post_youtube_made_for_kids" in patch:
        config.app["upload_post_youtube_made_for_kids"] = bool(patch["upload_post_youtube_made_for_kids"])
    if isinstance(patch.get("upload_post_platforms"), list):
        allowed = {"tiktok", "instagram", "youtube"}
        config.app["upload_post_platforms"] = [item for item in patch["upload_post_platforms"] if item in allowed]
    if "hide_log" in patch:
        config.ui["hide_log"] = bool(patch["hide_log"])
    if "open_task_folder_on_completion" in patch:
        config.ui["open_task_folder_on_completion"] = bool(patch["open_task_folder_on_completion"])
    if isinstance(patch.get("tts_server"), str) and patch["tts_server"] in {item[0] for item in TTS_SERVERS}:
        config.ui["tts_server"] = patch["tts_server"]
    if isinstance(patch.get("bgm_type"), str) and patch["bgm_type"] in {item[0] for item in BGM_TYPES}:
        config.ui["bgm_type"] = patch["bgm_type"]
    if isinstance(patch.get("bgm_file"), str):
        config.ui["bgm_file"] = _limit(patch["bgm_file"], 180)


def cache_status(max_age_days: int | None = None) -> dict[str, Any]:
    stats = cache_manager.get_video_cache_stats(max_age_days)
    return {
        "file_count": stats.file_count,
        "total_size": stats.total_size,
        "oldest_mtime": stats.oldest_mtime,
        "newest_mtime": stats.newest_mtime,
    }


def clean_cache(max_age_days: int | None = None) -> dict[str, Any]:
    result = cache_manager.clean_video_cache(max_age_days)
    return {
        "deleted_count": getattr(result, "deleted_count", 0),
        "deleted_size": getattr(result, "deleted_size", 0),
        "failed_count": getattr(result, "failed_count", 0),
    }


def _stage_dir() -> str:
    path = os.path.join(utils.storage_dir("frameforge", create=True), "staged")
    os.makedirs(path, exist_ok=True)
    return path


def stage_audio(filename: str, source) -> str:
    ext = os.path.splitext(filename or "")[1].lower()
    if ext not in {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma"}:
        raise ValueError("Upload an audio file such as mp3 or wav.")
    token = f"{uuid4().hex}{ext}"
    target = os.path.join(_stage_dir(), token)
    with open(target, "wb") as handle:
        shutil.copyfileobj(source, handle)
    return token


def _read_staged(token: str) -> bytes:
    if not token or "/" in token or "\\" in token or token.startswith("."):
        raise ValueError("Upload the reference audio again.")
    source = os.path.join(_stage_dir(), token)
    if not os.path.isfile(source):
        raise ValueError("Upload the reference audio again.")
    with open(source, "rb") as handle:
        return handle.read()


def preview_voice(voice_name: str, voice_rate: float, voice_volume: float) -> str:
    name = (voice_name or "").strip()
    if not name or name == "no-voice":
        raise ValueError("Choose a voice before previewing it.")
    folder = os.path.join(utils.storage_dir("frameforge", create=True), "previews")
    os.makedirs(folder, exist_ok=True)
    filename = f"{uuid4().hex}.mp3"
    voice_service.tts(
        text="This is a short voice preview.",
        voice_name=name,
        voice_rate=float(voice_rate or 1),
        voice_file=os.path.join(folder, filename),
        voice_volume=float(voice_volume or 1),
    )
    return filename


def preview_path(filename: str) -> str:
    if not filename or "/" in filename or "\\" in filename or not filename.endswith(".mp3"):
        raise ValueError("That preview is not available.")
    path = os.path.join(utils.storage_dir("frameforge", create=False), "previews", filename)
    if not os.path.isfile(path):
        raise ValueError("That preview is not available.")
    return path


def groq_models() -> list[str]:
    key = str(config.app.get("groq_api_key", "") or "").strip()
    if not key:
        return []
    import requests

    base = str(config.app.get("groq_base_url", "") or "https://api.groq.com/openai/v1").strip().rstrip("/")
    try:
        response = requests.get(f"{base}/models", headers={"Authorization": f"Bearer {key}"}, timeout=10)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return []
    names = []
    for item in payload.get("data", []) if isinstance(payload, dict) else []:
        if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"].strip():
            names.append(item["id"].strip())
    return sorted(set(names))


def subtitle_warnings(body: dict[str, Any]) -> dict[str, bool]:
    from app.models.schema import VideoParams
    from app.services import video

    payload = dict(body)
    payload.setdefault("video_subject", "preview")
    params = VideoParams.model_validate(payload)
    font_path = os.path.join(utils.font_dir(), str(params.font_name or ""))
    text = str(params.video_script or params.video_subject or "")
    unsupported = bool(
        params.subtitle_enabled
        and text
        and not video.subtitle_font_supports_text(font_path, text)
    )
    return {
        "colors": bool(video.subtitle_colors_are_indistinguishable(params)),
        "font": unsupported,
    }


def test_language_model() -> dict[str, Any]:
    ok, error, elapsed = llm.test_connection()
    return {"ok": bool(ok), "error": "" if ok else str(error or "The language model did not respond."), "elapsed": round(float(elapsed or 0), 2)}


_LOOM_SCRIPTS: dict[str, Any] = {}
_LOOM_VIDEOS: dict[str, Any] = {}


def _loomloom_script_backend():
    from app.services.loomloom import LoomLoomScriptBackend, LoomLoomSettings

    return LoomLoomScriptBackend(LoomLoomSettings.from_mapping(config.app))


def _loomloom_video_backend():
    from app.services.loomloom import LoomLoomVideoBackend, video_settings_from_mapping

    return LoomLoomVideoBackend(video_settings_from_mapping(config.app))


def loomloom_script_quote(body: dict[str, Any]) -> dict[str, Any]:
    backend = _loomloom_script_backend()
    batch = backend.prepare_script_batch(
        subject=str(body.get("video_subject") or ""),
        candidate_count=int(body.get("candidate_count") or 1),
        language=str(body.get("video_language") or "auto"),
        duration_seconds=int(body.get("duration_seconds") or 60),
        style=str(body.get("video_script_prompt") or ""),
    )
    quoted = backend.quote(batch)
    token = uuid4().hex
    _LOOM_SCRIPTS[token] = {"batch": batch, "quote": quoted, "request_id": f"ff-{uuid4().hex}"}
    amount = quoted.estimated_buyer_payable_amount or f"{quoted.estimated_buyer_payable_t} T"
    return {
        "token": token,
        "task_count": quoted.task_count,
        "currency": quoted.currency,
        "amount": amount,
    }


def loomloom_script_run(token: str, confirm: bool) -> dict[str, Any]:
    if confirm is not True:
        raise ValueError("Confirm the charge before starting a paid script batch.")
    stored = _LOOM_SCRIPTS.get(token)
    if not stored:
        raise ValueError("Get a new quote before starting the batch.")
    execution = _loomloom_script_backend().execute(
        stored["batch"],
        client_request_id=stored["request_id"],
        listing_version_id=stored["quote"].listing_version_id,
        confirm=True,
    )
    stored["run_id"] = execution.run_id
    return {"run_id": execution.run_id}


def loomloom_script_status(run_id: str) -> dict[str, Any]:
    backend = _loomloom_script_backend()
    run = backend.get_run(run_id)
    payload: dict[str, Any] = {
        "status": run.status,
        "total": run.total_tasks,
        "completed": run.completed_tasks,
        "failed": run.failed_tasks,
        "error": run.first_error_message,
        "candidates": [],
        "errors": [],
    }
    if run.status == "completed":
        result = backend.get_script_results(run_id)
        payload["candidates"] = [
            {"script": item.script, "terms": list(item.video_terms)}
            for item in result.candidates
        ]
        payload["errors"] = [item.message for item in result.errors]
    return payload


def loomloom_video_options() -> dict[str, Any]:
    capability = _loomloom_video_backend().resolve_video_capability()
    return {
        "models": [{"id": item.model_id, "label": item.display_name} for item in capability.models],
        "default_model": capability.default_model_id,
        "aspects": list(capability.aspect_ratios),
    }


def loomloom_video_quote(body: dict[str, Any]) -> dict[str, Any]:
    from app.services.loomloom import LoomLoomConfirmedVideoRequest

    backend = _loomloom_video_backend()
    scenes = body.get("scenes") if isinstance(body.get("scenes"), list) else []
    batch = backend.prepare_video_batch(
        subject=str(body.get("video_subject") or ""),
        scene_prompts=[str(item) for item in scenes],
        model_id=str(body.get("model_id") or ""),
        aspect_ratio=str(body.get("aspect") or "9:16"),
    )
    quoted = backend.quote(batch)
    token = uuid4().hex
    confirmed = LoomLoomConfirmedVideoRequest(
        settings=backend.settings,
        batch=batch,
        listing_version_id=quoted.listing_version_id,
        client_request_id=f"ff-{uuid4().hex}",
    )
    _LOOM_VIDEOS[token] = confirmed
    amount = quoted.estimated_buyer_payable_amount or f"{quoted.estimated_buyer_payable_t} T"
    return {"token": token, "task_count": quoted.task_count, "currency": quoted.currency, "amount": amount}


def _take_loomloom_video(token: str):
    confirmed = _LOOM_VIDEOS.pop(token, None)
    if confirmed is None:
        raise ValueError("Get a LoomLoom video quote and confirm it before generating.")
    return confirmed


def place_voiceover(task_id: str, token: str) -> str:
    if not token or "/" in token or "\\" in token or token.startswith("."):
        raise ValueError("Upload the voiceover again before generating.")
    source = os.path.join(_stage_dir(), token)
    if not os.path.isfile(source):
        raise ValueError("Upload the voiceover again before generating.")
    destination_dir = utils.task_dir(task_id)
    stored = f"custom-audio{os.path.splitext(token)[1].lower()}"
    shutil.copyfile(source, os.path.join(destination_dir, stored))
    return stored


def task_logs(task_id: str) -> list[str]:
    return webui_task.get_task_logs(task_id)


def queue_generation(
    params: Any,
    voiceover_token: str = "",
    loomloom_token: str = "",
    voxcpm_reference_token: str = "",
    voxcpm_prompt_token: str = "",
    voxcpm_prompt_text: str = "",
) -> str:
    task_id = utils.get_uuid()
    if voiceover_token:
        params.custom_audio_file = place_voiceover(task_id, voiceover_token)
    confirmed = _take_loomloom_video(loomloom_token) if loomloom_token else None
    reference = _read_staged(voxcpm_reference_token) if voxcpm_reference_token else None
    prompt_audio = _read_staged(voxcpm_prompt_token) if voxcpm_prompt_token else None
    webui_task.submit_generation(
        task_id=task_id,
        params=params,
        capture_logs=not bool(config.ui.get("hide_log", False)),
        loomloom_video_request=confirmed,
        voxcpm_reference_audio=reference,
        voxcpm_prompt_audio=prompt_audio,
        voxcpm_prompt_text=voxcpm_prompt_text,
    )
    dumped = params.model_dump(mode="json") if hasattr(params, "model_dump") else {}
    sm.state.patch_task(task_id, params=dumped)
    return task_id


_BACKUP_APP_KEYS = list(_APP_LIST_SECRETS.values()) + list(_APP_TEXT_SECRETS.values())
_BACKUP_SECTIONS = {
    "siliconflow": ["api_key"],
    "elevenlabs": ["api_key"],
    "minimax_tts": ["api_key"],
    "fish_audio": ["api_key"],
    "voxcpm": ["api_key"],
    "chatterbox": ["api_key"],
    "kokoro": ["api_key"],
    "azure": ["speech_key"],
}


def export_keys() -> dict[str, Any]:
    app_values = {}
    for key in _BACKUP_APP_KEYS:
        value = config.app.get(key, "")
        if _filled(value):
            app_values[key] = value
    sections = {}
    for name, keys in _BACKUP_SECTIONS.items():
        stored = {}
        for key in keys:
            value = _section(name).get(key, "")
            if _filled(value):
                stored[key] = value
        if stored:
            sections[name] = stored
    return {"app": app_values, "sections": sections}


def import_keys(payload: dict[str, Any]) -> None:
    app_values = payload.get("app") if isinstance(payload.get("app"), dict) else {}
    allowed_app = set(_BACKUP_APP_KEYS)
    for key, value in app_values.items():
        if key in allowed_app and _filled(value):
            config.app[key] = value
    sections = payload.get("sections") if isinstance(payload.get("sections"), dict) else {}
    for name, values in sections.items():
        if name not in _BACKUP_SECTIONS or not isinstance(values, dict):
            continue
        for key, value in values.items():
            if key in _BACKUP_SECTIONS[name] and _filled(value):
                _section(name)[key] = value
    utils_save = getattr(config, "save_config", None)
    if callable(utils_save):
        utils_save()


def open_task_folder(task_id: str) -> None:
    folder = utils.task_dir(task_id)
    if not os.path.isdir(folder):
        raise ValueError("That task folder is not on this computer.")
    if os.name == "nt":
        os.startfile(folder)  # noqa: S606 - local dashboard action, path is the task directory
    else:
        raise ValueError("Opening a folder is available on Windows.")
