"""Script-to-Reality Renderer Studio routes over the existing video and language services."""

from typing import Any, Optional

from fastapi import Body, Depends, File, Path, Query, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from app.controllers import base
from app.controllers.v1 import video as video_api
from app.controllers.v1.base import new_router
from app.models.exception import HttpException
from app.models.schema import TaskVideoRequest
from app.services import dashboard
from app.services import frameforge
from app.services import openrouter_video
from app.utils import utils

router = new_router(dependencies=[Depends(base.verify_token)])


class ProjectCreate(BaseModel):
    topic: str = Field(min_length=1, max_length=500)
    title: str = Field(default="", max_length=120)


class SceneClipIn(BaseModel):
    model_config = ConfigDict(extra="ignore")
    url: str = ""
    thumbnail: str = ""
    duration: int = 0
    page: str = ""
    creator: str = ""
    asset_id: str = ""


class SceneIn(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = ""
    narration: str = ""
    term: str = ""
    clip: Optional[SceneClipIn] = None


class ProjectPatch(BaseModel):
    model_config = ConfigDict(extra="ignore")
    title: Optional[str] = None
    topic: Optional[str] = None
    script: Optional[str] = None
    aspect: Optional[str] = None
    voice_name: Optional[str] = None
    voice_rate: Optional[float] = None
    voice_volume: Optional[float] = None
    bgm_volume: Optional[float] = None
    subtitle_enabled: Optional[bool] = None
    subtitle_position: Optional[str] = None
    font_name: Optional[str] = None
    font_size: Optional[int] = None
    text_fore_color: Optional[str] = None
    text_background_color: Optional[str] = None
    use_caption_bar: Optional[bool] = None
    stroke_width: Optional[float] = None
    clip_duration: Optional[int] = None
    transition: Optional[str] = None
    video_source: Optional[str] = None
    video_fit_mode: Optional[str] = None
    video_concat_mode: Optional[str] = None
    video_clip_speed: Optional[float] = None
    video_count: Optional[int] = None
    match_materials_to_script: Optional[bool] = None
    subtitle_display_mode: Optional[str] = None
    subtitle_animation: Optional[str] = None
    custom_position: Optional[float] = None
    stroke_color: Optional[str] = None
    paragraph_number: Optional[int] = None
    video_language: Optional[str] = None
    scenes: Optional[list[SceneIn]] = None


class SettingsPatch(BaseModel):
    model_config = ConfigDict(extra="ignore")
    openrouter_model_name: Optional[str] = None
    default_voice: Optional[str] = None
    default_aspect: Optional[str] = None
    font_name: Optional[str] = None


class SceneSearch(BaseModel):
    term: str = ""


def _call(request: Request, action, *args):
    request_id = base.get_task_id(request)
    try:
        return action(*args)
    except LookupError as exc:
        raise HttpException(task_id=request_id, status_code=404, message=str(exc)) from exc
    except ValueError as exc:
        raise HttpException(task_id=request_id, status_code=400, message=frameforge.public_error(str(exc))) from exc
    except RuntimeError as exc:
        raise HttpException(task_id=request_id, status_code=503, message=frameforge.public_error(str(exc))) from exc


def _patch_dict(body: BaseModel) -> dict[str, Any]:
    return body.model_dump(exclude_unset=True)


@router.get("/frameforge/settings", summary="Script-to-Reality Renderer Studio bench settings")
def get_settings():
    return utils.get_response(200, {"settings": frameforge.public_settings()})


@router.patch("/frameforge/settings", summary="Update Script-to-Reality Renderer Studio bench settings")
def patch_settings(request: Request, body: dict[str, Any] = Body(default_factory=dict)):
    settings = _call(request, frameforge.update_settings, body if isinstance(body, dict) else {})
    return utils.get_response(200, {"settings": settings})


@router.get("/frameforge/voices", summary="English narration voices")
def get_voices():
    return utils.get_response(200, {"voices": list(frameforge.VOICES), "fonts": frameforge.available_fonts()})


@router.get("/frameforge/projects", summary="List Script-to-Reality Renderer Studio projects")
def get_projects():
    return utils.get_response(200, {"projects": frameforge.list_projects()})


@router.post("/frameforge/projects", summary="Create a Script-to-Reality Renderer Studio project")
def post_project(request: Request, body: ProjectCreate):
    project = _call(request, frameforge.create_project, body.topic, body.title)
    return utils.get_response(200, {"project": project})


@router.get("/frameforge/projects/{project_id}", summary="Read a Script-to-Reality Renderer Studio project")
def get_project(request: Request, project_id: str = Path(...)):
    project = _call(request, frameforge.get_project, project_id)
    return utils.get_response(200, {"project": project})


@router.patch("/frameforge/projects/{project_id}", summary="Update a Script-to-Reality Renderer Studio project")
def patch_project(request: Request, body: ProjectPatch, project_id: str = Path(...)):
    def update(project_id: str, patch: dict):
        return frameforge.apply_project_patch(frameforge.get_project(project_id), patch)

    project = _call(request, update, project_id, _patch_dict(body))
    return utils.get_response(200, {"project": project})


@router.delete("/frameforge/projects/{project_id}", summary="Delete a Script-to-Reality Renderer Studio project")
def remove_project(request: Request, project_id: str = Path(...)):
    _call(request, frameforge.delete_project, project_id)
    return utils.get_response(200, {"deleted": project_id})


@router.post("/frameforge/projects/{project_id}/script", summary="Write a narration with OpenRouter")
def post_script(request: Request, project_id: str = Path(...)):
    def write(project_id: str):
        return frameforge.generate_project_script(frameforge.get_project(project_id))

    project = _call(request, write, project_id)
    return utils.get_response(200, {"project": project})


@router.post("/frameforge/projects/{project_id}/storyboard", summary="Build an ordered storyboard")
def post_storyboard(request: Request, project_id: str = Path(...)):
    def build(project_id: str):
        return frameforge.build_storyboard(frameforge.get_project(project_id))

    project = _call(request, build, project_id)
    return utils.get_response(200, {"project": project})


@router.post(
    "/frameforge/projects/{project_id}/scenes/{scene_id}/search",
    summary="Search stock footage for one scene",
)
def post_scene_search(
    request: Request,
    body: SceneSearch,
    project_id: str = Path(...),
    scene_id: str = Path(...),
):
    def search(project_id: str, scene_id: str, term: str):
        project = frameforge.get_project(project_id)
        return frameforge.search_scene_clips(project, scene_id, term)

    clips = _call(request, search, project_id, scene_id, body.term)
    return utils.get_response(200, {"clips": clips})


@router.post("/frameforge/projects/{project_id}/render", summary="Render a Script-to-Reality Renderer Studio project")
def post_render(request: Request, body: ProjectPatch, project_id: str = Path(...)):
    request_id = base.get_task_id(request)

    def prepare(project_id: str, patch: dict):
        project = frameforge.get_project(project_id)
        if patch:
            project = frameforge.apply_project_patch(project, patch)
        project = frameforge.ensure_renderable(project)
        materials = None
        if frameforge.uses_locked_footage(project):
            try:
                materials = frameforge.assemble_local_materials(project)
            except ValueError:
                raise
            except Exception as exc:
                raise RuntimeError(frameforge.public_error(str(exc))) from exc
        task_request = frameforge.build_task_request(project, materials)
        return project, task_request

    project, task_request = _call(request, prepare, project_id, _patch_dict(body))
    try:
        queued = video_api.create_task(request, task_request, stop_at="video")
    except HttpException:
        raise
    task = queued.get("data") if isinstance(queued, dict) else None
    task_id = task.get("task_id") if isinstance(task, dict) else ""
    if not task_id:
        raise HttpException(task_id=request_id, status_code=500, message="The render could not be queued.")
    project = frameforge.remember_task(project, task_id)
    return utils.get_response(200, {"project": project, "task_id": task_id})


@router.get("/frameforge/library", summary="Finished renders on disk")
def get_library():
    return utils.get_response(200, {"renders": frameforge.list_library()})


@router.get("/frameforge/catalog", summary="Dashboard choices")
def get_catalog():
    return utils.get_response(200, dashboard.catalog())


@router.get("/frameforge/provider-voices", summary="Voices for one narration service")
def get_provider_voices(server: str = Query(default="azure-tts-v1")):
    return utils.get_response(200, {"voices": dashboard.voices_for(server)})


@router.post("/frameforge/prompt-preview", summary="Show the script prompt that will be sent")
def post_prompt_preview(body: dict[str, Any] = Body(default_factory=dict)):
    return utils.get_response(200, {"prompt": dashboard.prompt_preview(body if isinstance(body, dict) else {})})


@router.post("/frameforge/voiceover", summary="Stage a voiceover upload")
def post_voiceover(request: Request, file: UploadFile = File(...)):
    token = _call(request, dashboard.stage_audio, file.filename or "audio.mp3", file.file)
    return utils.get_response(200, {"token": token})


@router.post("/frameforge/generate", summary="Generate a video with the dashboard controls")
def post_generate(request: Request, body: dict[str, Any] = Body(default_factory=dict)):
    payload = body if isinstance(body, dict) else {}
    token = str(payload.pop("voiceover_token", "") or "")
    loomloom_token = str(payload.pop("loomloom_token", "") or "")
    voxcpm_reference_token = str(payload.pop("voxcpm_reference_token", "") or "")
    voxcpm_prompt_token = str(payload.pop("voxcpm_prompt_token", "") or "")
    voxcpm_prompt_text = str(payload.pop("voxcpm_prompt_text", "") or "")

    def run() -> str:
        params = TaskVideoRequest.model_validate(payload)
        return dashboard.queue_generation(
            params,
            token,
            loomloom_token,
            voxcpm_reference_token,
            voxcpm_prompt_token,
            voxcpm_prompt_text,
        )

    task_id = _call(request, run)
    return utils.get_response(200, {"task_id": task_id})


@router.post("/frameforge/voice-preview", summary="Speak a short sample of the selected voice")
def post_voice_preview(request: Request, body: dict[str, Any] = Body(default_factory=dict)):
    payload = body if isinstance(body, dict) else {}

    def run() -> str:
        return dashboard.preview_voice(
            str(payload.get("voice_name") or ""),
            float(payload.get("voice_rate") or 1),
            float(payload.get("voice_volume") or 1),
        )

    filename = _call(request, run)
    return utils.get_response(200, {"file": filename})


@router.get("/frameforge/previews/{filename}", summary="Play a voice preview")
def get_preview(request: Request, filename: str = Path(...)):
    return FileResponse(_call(request, dashboard.preview_path, filename), media_type="audio/mpeg")


@router.get("/frameforge/groq-models", summary="Models available from Groq")
def get_groq_models(request: Request):
    return utils.get_response(200, {"models": _call(request, dashboard.groq_models)})


@router.post("/frameforge/subtitle-check", summary="Warn about caption color and font coverage")
def post_subtitle_check(request: Request, body: dict[str, Any] = Body(default_factory=dict)):
    return utils.get_response(200, _call(request, dashboard.subtitle_warnings, body if isinstance(body, dict) else {}))


@router.post("/frameforge/llm-test", summary="Check the language model connection")
def post_llm_test(request: Request):
    return utils.get_response(200, _call(request, dashboard.test_language_model))


@router.post("/frameforge/loomloom/scripts/quote", summary="Quote a paid script batch")
def post_loomloom_script_quote(request: Request, body: dict[str, Any] = Body(default_factory=dict)):
    return utils.get_response(200, _call(request, dashboard.loomloom_script_quote, body if isinstance(body, dict) else {}))


@router.post("/frameforge/loomloom/scripts/run", summary="Start a confirmed script batch")
def post_loomloom_script_run(request: Request, body: dict[str, Any] = Body(default_factory=dict)):
    payload = body if isinstance(body, dict) else {}
    result = _call(request, dashboard.loomloom_script_run, str(payload.get("token") or ""), payload.get("confirm") is True)
    return utils.get_response(200, result)


@router.get("/frameforge/loomloom/scripts/{run_id}", summary="Read a script batch")
def get_loomloom_script_run(request: Request, run_id: str = Path(...)):
    return utils.get_response(200, _call(request, dashboard.loomloom_script_status, run_id))


@router.get("/frameforge/loomloom/video-options", summary="Models available for paid video")
def get_loomloom_video_options(request: Request):
    return utils.get_response(200, _call(request, dashboard.loomloom_video_options))


@router.post("/frameforge/loomloom/video/quote", summary="Quote a paid video batch")
def post_loomloom_video_quote(request: Request, body: dict[str, Any] = Body(default_factory=dict)):
    return utils.get_response(200, _call(request, dashboard.loomloom_video_quote, body if isinstance(body, dict) else {}))


@router.get("/frameforge/openrouter/video-models", summary="OpenRouter video models")
def get_openrouter_video_models(request: Request):
    return utils.get_response(200, {"models": _call(request, openrouter_video.public_models)})


@router.post("/frameforge/openrouter/video-quote", summary="Price one OpenRouter video clip")
def post_openrouter_video_quote(request: Request, body: dict[str, Any] = Body(default_factory=dict)):
    return utils.get_response(200, _call(request, openrouter_video.quote, body if isinstance(body, dict) else {}))


@router.get("/frameforge/locale", summary="Interface language strings")
def get_locale(code: str = Query(default="en")):
    import json
    from pathlib import Path as FilePath

    safe = "".join(ch for ch in code if ch.isalpha() or ch == "-")[:12] or "en"
    path = FilePath(__file__).resolve().parents[3] / "webui" / "i18n" / f"{safe}.json"
    if not path.is_file():
        path = FilePath(__file__).resolve().parents[3] / "webui" / "i18n" / "en.json"
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    strings = payload.get("Translation") if isinstance(payload, dict) else {}
    if not isinstance(strings, dict):
        strings = {}
    return utils.get_response(200, {"code": safe, "strings": strings})


@router.get("/frameforge/tasks/{task_id}/logs", summary="Generation log for one task")
def get_logs(task_id: str = Path(...)):
    return utils.get_response(200, {"lines": dashboard.task_logs(task_id)})


@router.post("/frameforge/tasks/{task_id}/open", summary="Open the task folder")
def post_open_task(request: Request, task_id: str = Path(...)):
    _call(request, dashboard.open_task_folder, task_id)
    return utils.get_response(200, {"opened": task_id})


@router.get("/frameforge/cache", summary="Video cache size")
def get_cache(max_age_days: int | None = Query(default=None)):
    return utils.get_response(200, dashboard.cache_status(max_age_days))


@router.post("/frameforge/cache/clean", summary="Delete cached videos")
def post_clean_cache(request: Request, body: dict[str, Any] = Body(default_factory=dict)):
    days = body.get("max_age_days") if isinstance(body, dict) else None
    result = _call(request, dashboard.clean_cache, days if isinstance(days, int) else None)
    return utils.get_response(200, result)


@router.get("/frameforge/key-backup", summary="Download stored keys")
def get_key_backup():
    return utils.get_response(200, dashboard.export_keys())


@router.post("/frameforge/key-backup", summary="Restore stored keys")
def post_key_backup(request: Request, body: dict[str, Any] = Body(default_factory=dict)):
    _call(request, dashboard.import_keys, body if isinstance(body, dict) else {})
    return utils.get_response(200, {"restored": True})
