"""One-click scene generation: storyboard, paid clips, CV, then file paths for assembly.

A saved job id is polled instead of submitted again. A scene whose estimate exceeds
the remaining project budget is not submitted.
"""

from __future__ import annotations

import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from typing import Any, Callable

from loguru import logger

from app.models.schema import VideoConcatMode
from app.services import cv_engine, openrouter_video, state as sm, storyboard
from app.utils import utils

_guard = threading.Lock()
_running: set[str] = set()


class AutonomousError(openrouter_video.OpenRouterVideoError):
    pass


def _complete(prompt: str) -> str:
    from app.services import llm
    return llm._generate_response(prompt)


def _read(task_id: str) -> dict[str, Any]:
    try:
        task = sm.state.get_task(task_id)
    except Exception:
        return {}
    return task if isinstance(task, dict) else {}


def _write(task_id: str, **fields: Any) -> None:
    try:
        sm.state.patch_task(task_id, **fields)
    except Exception:
        return


def prepare_storyboard(task_id: str, params: Any, script: str, *, complete: Callable[[str], str] | None = None, session=None) -> dict[str, Any]:
    duration = int(getattr(params, "openrouter_video_duration", None) or openrouter_video.DEFAULT_DURATION)
    paragraph_cap = int(getattr(params, "paragraph_number", None) or 1)
    requested = 6 if paragraph_cap <= 1 else max(1, min(paragraph_cap, 6))
    board = storyboard.build_storyboard(
        subject=str(getattr(params, "video_subject", "") or ""),
        script=script,
        style=str(getattr(params, "visual_style", "") or ""),
        duration=duration,
        count=requested,
        complete=complete if complete is not None else _complete,
    )
    model_id = str(getattr(params, "openrouter_video_model", "") or "").strip() or openrouter_video.DEFAULT_MODEL
    resolution = str(getattr(params, "openrouter_video_resolution", "") or openrouter_video.DEFAULT_RESOLUTION)
    aspect = getattr(params, "video_aspect", "") or openrouter_video.DEFAULT_ASPECT
    aspect = str(getattr(aspect, "value", aspect) or openrouter_video.DEFAULT_ASPECT)
    quote = openrouter_video.quote(
        {
            "model": model_id,
            "resolution": resolution,
            "aspect_ratio": aspect,
            "duration": duration,
            "budget_usd": format(openrouter_video.project_budget(getattr(params, "project_budget_usd", None)), "f"),
            "project_budget": True,
        },
        session=session,
    )
    affordable = int(quote.get("max_scenes") or 0)
    if affordable < 1 or not quote.get("within_budget"):
        raise openrouter_video.OpenRouterVideoBudgetError(
            f"budget-exhausted: one {duration}s scene is estimated at ${quote.get('estimated_usd')}, "
            f"above the ${quote.get('budget_usd')} project budget. No clip was submitted."
        )
    scenes = board["scenes"][: min(len(board["scenes"]), affordable)]
    for index, scene in enumerate(scenes):
        scene["index"] = index
        scene["duration"] = duration
    payload = {"scenes": scenes, "script": "\n\n".join(scene["narration"] for scene in scenes), "source": board["source"], "quote": quote}
    _write(task_id, storyboard=payload)
    return payload


def _price(model: dict[str, Any], duration: int, resolution: str) -> Decimal:
    return openrouter_video._money(openrouter_video.estimate_cost(model, duration, resolution)["estimated_usd"])


def scenes_for_coverage(scenes: list[dict[str, Any]], required_seconds: float, clip_seconds: int) -> list[dict[str, Any]]:
    """Build enough distinct scene prompts to cover the narration without repeating a file."""
    clip_seconds = max(1, int(clip_seconds))
    count = max(1, math.ceil(max(0.0, float(required_seconds)) / clip_seconds - 1e-9))
    source = scenes or [{"narration": "", "visual_prompt": "Continue the scene", "duration": clip_seconds}]
    fitted = []
    for index in range(count):
        if index < len(source):
            scene = dict(source[index])
        else:
            seed = source[index % len(source)]
            scene = dict(seed)
            scene["visual_prompt"] = (
                f"{seed.get('visual_prompt') or seed.get('narration') or 'Continue the scene'}. "
                f"New camera angle {index + 1}, continuous action, no repeated shot."
            )
        scene["index"] = index
        scene["duration"] = clip_seconds
        fitted.append(scene)
    return fitted


def _clip_seconds(preferred: int, model: dict[str, Any]) -> int:
    allowed = []
    for item in model.get("supported_durations") or []:
        try:
            allowed.append(int(item))
        except (TypeError, ValueError):
            continue
    if allowed and int(preferred) not in allowed:
        return min(allowed, key=lambda value: abs(value - int(preferred)))
    return max(1, int(preferred))


def generate_for_task(
    task_id: str,
    params: Any,
    video_script: str = "",
    audio_duration: float | None = None,
    subtitle_path: str = "",
    *,
    session=None,
    sleep: Callable[[float], None] = None,
    clock: Callable[[], float] = None,
    analyze: Callable | None = None,
    complete: Callable[[str], str] | None = None,
) -> list[str]:
    if sleep is None:
        import time
        sleep = time.sleep
    if clock is None:
        import time
        clock = time.monotonic
    with _guard:
        if task_id in _running:
            raise AutonomousError("This video is already generating. A second paid request was not sent.")
        _running.add(task_id)
    try:
        return _generate(
            task_id, params, video_script, audio_duration, subtitle_path,
            session=session, sleep=sleep, clock=clock, analyze=analyze, complete=complete,
        )
    finally:
        with _guard:
            _running.discard(task_id)


def _generate(task_id, params, video_script, audio_duration, subtitle_path="", *, session, sleep, clock, analyze, complete) -> list[str]:
    stored = _read(task_id)
    board = stored.get("storyboard") if isinstance(stored.get("storyboard"), dict) else None
    if audio_duration is None and (not board or not board.get("scenes")):
        board = prepare_storyboard(task_id, params, video_script, complete=complete, session=session)
    scenes = list((board or {}).get("scenes") or [])
    if audio_duration is None and not scenes:
        raise AutonomousError("The storyboard did not contain any scenes.")
    model_id = str(getattr(params, "openrouter_video_model", "") or "").strip() or openrouter_video.DEFAULT_MODEL
    resolution = str(getattr(params, "openrouter_video_resolution", "") or openrouter_video.DEFAULT_RESOLUTION)
    aspect = getattr(params, "video_aspect", "") or openrouter_video.DEFAULT_ASPECT
    aspect = str(getattr(aspect, "value", aspect) or openrouter_video.DEFAULT_ASPECT)
    client = session or __import__("requests").Session()
    model = openrouter_video._find_model(openrouter_video.list_models(client), model_id)
    clip_seconds = _clip_seconds(int(getattr(params, "openrouter_video_duration", None) or openrouter_video.DEFAULT_DURATION), model)
    timeline = False
    if audio_duration is not None:
        required = max(0.0, float(audio_duration)) + 0.1
        style = str(getattr(params, "visual_style", "") or "")
        cues = storyboard.load_srt(subtitle_path)
        if not cues:
            spoken = str(video_script or "").strip() or " ".join(
                str(scene.get("narration") or "") for scene in scenes
            ).strip()
            cues = [(0.0, float(audio_duration), spoken or "Continue the scene")]
        scenes = storyboard.scenes_from_cues(cues, clip_seconds, style=style, required_seconds=required)
        timeline = True
        logger.info(
            "cv coverage narration=%.2fs required=%.2fs clip=%ss scenes=%s",
            float(audio_duration), required, clip_seconds, len(scenes),
        )
    budget = openrouter_video.project_budget(getattr(params, "project_budget_usd", None))
    estimate = _price(model, clip_seconds, resolution)
    if estimate * len(scenes) > budget:
        raise openrouter_video.OpenRouterVideoBudgetError(
            "budget-exhausted: covering the narration needs "
            f"{len(scenes)} clip(s) at ${openrouter_video._format_money(estimate)} each, "
            f"above the ${openrouter_video._format_money(budget)} project budget. No clip was submitted."
        )
    if timeline:
        scenes = storyboard.apply_shot_prompts(scenes, str(getattr(params, "visual_style", "") or ""), complete or _complete)
        _write(task_id, storyboard={"scenes": scenes, "script": video_script, "source": "timeline"})
    jobs = [dict(row) for row in stored.get("autonomous_jobs") or [] if isinstance(row, dict)]
    retries = max(0, min(int(getattr(params, "max_scene_retries", None) or 0), 3))
    failure_policy = str(getattr(params, "cv_failure_policy", "") or "fail")
    threshold = float(getattr(params, "semantic_threshold", None) or cv_engine.DEFAULT_THRESHOLDS["semantic"])
    checker = analyze or (
            lambda path, description: cv_engine.evaluate(
            cv_engine.sample_frames(path, count=4),
            description,
            thresholds={"semantic": threshold},
            failure_policy=failure_policy,
        )
    )
    history: list[dict[str, Any]] = []
    accepted: dict[int, dict[str, Any]] = {}
    task_dir = utils.task_dir(task_id)
    state_lock = threading.Lock()
    cv_lock = threading.Lock()

    def _one_take(index: int, take: int, prompt: str, duration: int, *, allow_budget_skip: bool) -> dict[str, Any] | None:
        nonlocal jobs
        estimate = _price(model, duration, resolution)
        destination = os.path.join(task_dir, f"scene-{index}-take-{take}.mp4")
        with state_lock:
            prior = next((row for row in jobs if int(row.get("scene", -1)) == index and int(row.get("take", -1)) == take), None)
            existing_job = str((prior or {}).get("job_id") or "")
            if existing_job.startswith("reserve-"):
                existing_job = ""
            existing_path = str((prior or {}).get("path") or "")
            spent = _spent_from(jobs)
            reuse = bool(existing_path and os.path.isfile(existing_path))
            if not reuse and not existing_job and estimate > budget - spent:
                if allow_budget_skip:
                    logger.info("cv scene=%s take=%s skipped reason=budget remaining=%s", index, take, openrouter_video._format_money(budget - spent))
                    return None
                _finish(task_id, history, spent, jobs, "budget-exhausted")
                raise openrouter_video.OpenRouterVideoBudgetError(
                    "budget-exhausted: the remaining budget cannot pay for another scene. "
                    f"{len(accepted)} accepted clip(s) were kept. No further clip was submitted."
                )
            if not reuse and not existing_job:
                jobs.append({
                    "scene": index,
                    "take": take,
                    "job_id": f"reserve-{index}-{take}",
                    "path": "",
                    "status": "reserving",
                    "estimated_usd": openrouter_video._format_money(estimate),
                })
        owns_session = session is None
        clip_session = client if not owns_session else __import__("requests").Session()
        try:
            if reuse:
                clip = {
                    "path": existing_path,
                    "job_id": str((prior or {}).get("job_id") or ""),
                    "estimated_usd": str((prior or {}).get("estimated_usd") or openrouter_video._format_money(estimate)),
                    "submitted": False,
                }
            else:
                clip = openrouter_video.generate_clip(
                    task_id,
                    prompt=prompt,
                    model_id=model_id,
                    resolution=resolution,
                    aspect=aspect,
                    duration=duration,
                    budget=budget,
                    destination=destination,
                    existing_job_id=existing_job,
                    model=model,
                    session=clip_session,
                    sleep=sleep,
                    clock=clock,
                )
        except openrouter_video.OpenRouterVideoError as exc:
            job_id = str(exc.job_id or "")
            with state_lock:
                jobs = [row for row in jobs if not (int(row.get("scene", -1)) == index and int(row.get("take", -1)) == take and str(row.get("job_id") or "").startswith("reserve-"))]
                if job_id and not any(str(row.get("job_id") or "") == job_id for row in jobs):
                    jobs.append({
                        "scene": index,
                        "take": take,
                        "job_id": job_id,
                        "path": "",
                        "status": "submitted",
                        "estimated_usd": openrouter_video._format_money(estimate),
                    })
                _write(task_id, autonomous_jobs=jobs, autonomous_spent_usd=openrouter_video._format_money(_spent_from(jobs)))
            raise
        finally:
            if owns_session:
                clip_session.close()
        with cv_lock:
            playable = cv_engine.validate_playable(clip["path"])
            if not playable["playable"]:
                logger.info("cv Frame sampling: failed — The clip could not be read.")
                report = {
                    "accepted": False,
                    "score": 0.0,
                    "warnings": ["unreadable video"],
                    "metrics": playable,
                    "disclaimer": cv_engine.DISCLAIMER,
                    "components": [{
                        "id": "frames",
                        "name": "Frame sampling",
                        "status": "failed",
                        "detail": "The clip could not be read.",
                    }],
                }
            else:
                report = checker(clip["path"], prompt)
        report = dict(report)
        report["scene"] = index
        report["take"] = take
        report["job_id"] = clip["job_id"]
        row = {
            "scene": index,
            "take": take,
            "job_id": clip["job_id"],
            "path": clip["path"],
            "status": "accepted" if report.get("accepted") else "rejected",
            "estimated_usd": clip["estimated_usd"],
        }
        with state_lock:
            jobs = [item for item in jobs if not (int(item.get("scene", -1)) == index and int(item.get("take", -1)) == take)]
            jobs.append(row)
            spent = _spent_from(jobs)
            metrics = report.get("metrics") or {}
            history.append(report)
            _write(task_id, autonomous_jobs=jobs, autonomous_spent_usd=openrouter_video._format_money(spent), scene_status=_statuses(history))
        logger.info(
            "cv scene=%s take=%s semantic_backend=%s semantic=%s detector=%s objects=%s flow=%s blur=%s score=%s decision=%s reasons=%s spent=%s",
            index,
            take,
            metrics.get("semantic_backend"),
            metrics.get("semantic_similarity"),
            metrics.get("detector"),
            metrics.get("objects"),
            metrics.get("mean_flow"),
            metrics.get("blur"),
            report.get("score"),
            "accepted" if report.get("accepted") else "rejected",
            ",".join(report.get("warnings") or []) or "-",
            openrouter_video._format_money(spent),
        )
        if report.get("capability_failure"):
            _finish(task_id, history, spent, jobs, "failed")
            raise AutonomousError(str(report.get("refinement") or "CV model unavailable. The scene was not regenerated."))
        if not (report.get("accepted") and cv_engine.validate_playable(clip["path"])["playable"]):
            return None
        output_path = clip["path"]
        crops = list(metrics.get("reframe") or [])
        if crops:
            reframed = os.path.join(task_dir, f"scene-{index}-take-{take}-reframed.mp4")
            output_path = cv_engine.apply_reframe(clip["path"], crops[0], reframed)
            if output_path != clip["path"]:
                metrics["reframe_applied"] = True
                logger.info(f"cv YOLOv8n: passed — crop applied on scene {index + 1} take {take}.")
                for item in report.get("components") or []:
                    if item.get("id") == "yolo":
                        item["detail"] = f"{str(item.get('detail') or '').rstrip('.')} Crop applied."
                        break
        return {"path": output_path, "score": float(report.get("score") or 0), "report": report, "refinement": report.get("refinement") or ""}

    def _latest_refinement(index: int) -> str:
        with state_lock:
            for report in reversed(history):
                if report.get("scene") == index:
                    return str(report.get("refinement") or "")
        return ""

    def _generate_scene(scene: dict[str, Any]) -> dict[str, Any]:
        index = int(scene["index"])
        base_prompt = str(scene.get("visual_prompt") or scene.get("narration") or "")
        prompt = base_prompt
        duration = int(scene.get("duration") or openrouter_video.DEFAULT_DURATION)
        accepted_take = None
        for take in range(retries + 1):
            accepted_take = _one_take(index, take, prompt, duration, allow_budget_skip=False)
            if accepted_take is not None:
                break
            prompt = f"{base_prompt}. {_latest_refinement(index)}".strip()
            logger.info(
                "cv regenerate scene=%s next_take=%s estimated=%s remaining=%s",
                index,
                take + 1,
                openrouter_video._format_money(_price(model, duration, resolution)),
                openrouter_video._format_money(budget - _spent_from(jobs)),
            )
        if accepted_take is None:
            logger.info("cv defer scene=%s until the other scenes finish kept=%s", index, len(accepted))
            return {
                "deferred": {
                    "index": index,
                    "prompt": prompt,
                    "base_prompt": base_prompt,
                    "duration": duration,
                    "next_take": retries + 1,
                }
            }
        return {"index": index, "accepted": accepted_take}

    from app.utils import logging_utils

    workers = min(4, max(1, len(scenes)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        first_pass = list(pool.map(logging_utils.bind_log_scope(_generate_scene), scenes))
    deferred = []
    for result in first_pass:
        if result.get("accepted") is not None and "index" in result and "deferred" not in result:
            accepted[int(result["index"])] = result["accepted"]
        elif result.get("deferred"):
            deferred.append(result["deferred"])
    for item in deferred:
        index = int(item["index"])
        duration = int(item["duration"])
        take = int(item["next_take"])
        logger.info("cv retry-at-end scene=%s take=%s", index, take)
        accepted_take = _one_take(index, take, str(item["prompt"]), duration, allow_budget_skip=True)
        if accepted_take is None:
            take += 1
            alternate = f"{item['base_prompt']}. Different camera angle, continuous action, no repeated shot."
            logger.info("cv alternate scene=%s take=%s", index, take)
            accepted_take = _one_take(index, take, alternate, duration, allow_budget_skip=True)
        if accepted_take is not None:
            accepted[index] = accepted_take
    missing = [int(scene["index"]) + 1 for scene in scenes if int(scene["index"]) not in accepted]
    if missing:
        _finish(task_id, history, _spent_from(jobs), jobs, "failed")
        raise AutonomousError(
            f"Scene {', '.join(str(number) for number in missing)} was still rejected after another attempt at the end. "
            f"{len(accepted)} accepted clip(s) were kept and a rejected clip was not assembled."
        )
    chosen = [accepted[int(scene["index"])]["path"] for scene in scenes]
    _write(
        task_id,
        scene_status=_statuses(history),
        cv_report={"scenes": history, "spent_usd": openrouter_video._format_money(_spent_from(jobs)), "disclaimer": cv_engine.DISCLAIMER},
    )
    params.video_concat_mode = VideoConcatMode.sequential
    duration = int(scenes[0].get("duration") or 5)
    if int(getattr(params, "video_clip_duration", 0) or 0) < duration:
        params.video_clip_duration = duration
    _finish(task_id, history, _spent_from(jobs), jobs, "completed")
    return chosen


def _spent_from(jobs: list[dict[str, Any]]) -> Decimal:
    seen: set[str] = set()
    total = Decimal("0")
    for row in jobs:
        job_id = str(row.get("job_id") or "")
        if not job_id or job_id in seen:
            continue
        seen.add(job_id)
        total += openrouter_video._money(row.get("estimated_usd") or "0")
    return total


def _statuses(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for report in reports:
        rows.append({
            "index": report.get("scene"),
            "take": report.get("take"),
            "status": "accepted" if report.get("accepted") else "rejected",
            "score": report.get("score"),
            "warnings": list(report.get("warnings") or [])[:4],
            "components": list(report.get("components") or []),
        })
    return rows


def _finish(task_id: str, reports: list[dict[str, Any]], spent: Decimal, jobs: list[dict[str, Any]], terminal: str) -> None:
    _write(
        task_id,
        autonomous_jobs=jobs,
        autonomous_spent_usd=openrouter_video._format_money(spent),
        autonomous_terminal=terminal,
        cv_report={
            "scenes": reports,
            "spent_usd": openrouter_video._format_money(spent),
            "terminal": terminal,
            "disclaimer": cv_engine.DISCLAIMER,
        },
        scene_status=_statuses(reports),
    )
