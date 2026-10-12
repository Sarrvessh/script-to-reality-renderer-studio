"""One-click scene generation: storyboard, paid clips, CV, then file paths for assembly.

A saved job id is polled instead of submitted again. A scene whose estimate exceeds
the remaining project budget is not submitted.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from decimal import Decimal
from typing import Any, Callable

logger = logging.getLogger(__name__)

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
            task_id, params, video_script, audio_duration,
            session=session, sleep=sleep, clock=clock, analyze=analyze, complete=complete,
        )
    finally:
        with _guard:
            _running.discard(task_id)


def _generate(task_id, params, video_script, audio_duration, *, session, sleep, clock, analyze, complete) -> list[str]:
    stored = _read(task_id)
    board = stored.get("storyboard") if isinstance(stored.get("storyboard"), dict) else None
    if not board or not board.get("scenes"):
        board = prepare_storyboard(task_id, params, video_script, complete=complete, session=session)
    scenes = list(board.get("scenes") or [])
    if not scenes:
        raise AutonomousError("The storyboard did not contain any scenes.")
    model_id = str(getattr(params, "openrouter_video_model", "") or "").strip() or openrouter_video.DEFAULT_MODEL
    resolution = str(getattr(params, "openrouter_video_resolution", "") or openrouter_video.DEFAULT_RESOLUTION)
    aspect = getattr(params, "video_aspect", "") or openrouter_video.DEFAULT_ASPECT
    aspect = str(getattr(aspect, "value", aspect) or openrouter_video.DEFAULT_ASPECT)
    client = session or __import__("requests").Session()
    model = openrouter_video._find_model(openrouter_video.list_models(client), model_id)
    clip_seconds = _clip_seconds(int(getattr(params, "openrouter_video_duration", None) or openrouter_video.DEFAULT_DURATION), model)
    if audio_duration is not None:
        required = max(0.0, float(audio_duration)) + 0.1
        scenes = scenes_for_coverage(scenes, required, clip_seconds)
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
    jobs = [dict(row) for row in stored.get("autonomous_jobs") or [] if isinstance(row, dict)]
    retries = max(0, min(int(getattr(params, "max_scene_retries", None) or 0), 3))
    failure_policy = str(getattr(params, "cv_failure_policy", "") or "fail")
    threshold = float(getattr(params, "semantic_threshold", None) or cv_engine.DEFAULT_THRESHOLDS["semantic"])
    checker = analyze or (
        lambda path, description: cv_engine.evaluate(
            cv_engine.sample_frames(path),
            description,
            thresholds={"semantic": threshold},
            failure_policy=failure_policy,
        )
    )
    chosen = []
    reports = []
    task_dir = utils.task_dir(task_id)
    for scene in scenes:
        index = int(scene["index"])
        prompt = str(scene.get("visual_prompt") or scene.get("narration") or "")
        duration = int(scene.get("duration") or openrouter_video.DEFAULT_DURATION)
        estimate = _price(model, duration, resolution)
        accepted_take = None
        for take in range(retries + 1):
            prior = next((row for row in jobs if int(row.get("scene", -1)) == index and int(row.get("take", -1)) == take), None)
            destination = os.path.join(task_dir, f"scene-{index}-take-{take}.mp4")
            existing_job = str((prior or {}).get("job_id") or "")
            existing_path = str((prior or {}).get("path") or "")
            spent = _spent_from(jobs)
            if existing_path and os.path.isfile(existing_path):
                clip = {
                    "path": existing_path,
                    "job_id": existing_job,
                    "estimated_usd": str((prior or {}).get("estimated_usd") or openrouter_video._format_money(estimate)),
                    "submitted": False,
                }
            else:
                remaining = budget - spent
                if not existing_job and estimate > remaining:
                    _finish(task_id, reports, spent, jobs, "budget-exhausted")
                    raise openrouter_video.OpenRouterVideoBudgetError(
                        "budget-exhausted: the remaining budget cannot pay for another scene. No further clip was submitted."
                    )
                try:
                    clip = openrouter_video.generate_clip(
                        task_id,
                        prompt=prompt,
                        model_id=model_id,
                        resolution=resolution,
                        aspect=aspect,
                        duration=duration,
                        budget=remaining if not existing_job else budget,
                        destination=destination,
                        existing_job_id=existing_job,
                        model=model,
                        session=client,
                        sleep=sleep,
                        clock=clock,
                    )
                except openrouter_video.OpenRouterVideoError as exc:
                    job_id = str(exc.job_id or _read(task_id).get("openrouter_video_job_id") or "")
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
            playable = cv_engine.validate_playable(clip["path"])
            if not playable["playable"]:
                report = {"accepted": False, "score": 0.0, "warnings": ["unreadable video"], "metrics": playable, "disclaimer": cv_engine.DISCLAIMER}
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
            jobs = [item for item in jobs if not (int(item.get("scene", -1)) == index and int(item.get("take", -1)) == take)]
            jobs.append(row)
            spent = _spent_from(jobs)
            metrics = report.get("metrics") or {}
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
            _write(task_id, autonomous_jobs=jobs, autonomous_spent_usd=openrouter_video._format_money(spent), scene_status=_statuses(reports + [report]))
            if report.get("capability_failure"):
                _finish(task_id, reports + [report], spent, jobs, "failed")
                raise AutonomousError(str(report.get("refinement") or "CV model unavailable. The scene was not regenerated."))
            if report.get("accepted") and cv_engine.validate_playable(clip["path"])["playable"]:
                output_path = clip["path"]
                crops = list(metrics.get("reframe") or [])
                if crops:
                    reframed = os.path.join(task_dir, f"scene-{index}-take-{take}-reframed.mp4")
                    output_path = cv_engine.apply_reframe(clip["path"], crops[0], reframed)
                    if output_path != clip["path"]:
                        metrics["reframe_applied"] = True
                candidate = {"path": output_path, "score": float(report.get("score") or 0), "report": report}
                if accepted_take is None or candidate["score"] >= accepted_take["score"]:
                    accepted_take = candidate
                break
            logger.info(
                "cv regenerate scene=%s next_take=%s estimated=%s remaining=%s",
                index,
                take + 1,
                openrouter_video._format_money(estimate),
                openrouter_video._format_money(budget - spent),
            )
            prompt = f"{scene.get('visual_prompt') or prompt}. {report.get('refinement') or ''}".strip()
        policy = str(getattr(params, "cv_terminal_policy", "") or "fail")
        if accepted_take is None:
            _finish(task_id, reports, spent, jobs, "failed")
            raise AutonomousError(
                f"Scene {index + 1} was rejected after {retries + 1} take(s). "
                f"Policy {policy} does not assemble a rejected clip."
            )
        chosen.append(accepted_take["path"])
        reports.append(accepted_take["report"])
        _write(
            task_id,
            scene_status=_statuses(reports),
            cv_report={"scenes": reports, "spent_usd": openrouter_video._format_money(spent), "disclaimer": cv_engine.DISCLAIMER},
        )
    params.video_concat_mode = VideoConcatMode.sequential
    duration = int(scenes[0].get("duration") or 5)
    if int(getattr(params, "video_clip_duration", 0) or 0) < duration:
        params.video_clip_duration = duration
    _finish(task_id, reports, spent, jobs, "completed")
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
            "status": "accepted" if report.get("accepted") else "best-effort",
            "score": report.get("score"),
            "warnings": list(report.get("warnings") or [])[:4],
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
