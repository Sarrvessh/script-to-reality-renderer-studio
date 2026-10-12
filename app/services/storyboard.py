"""Turn a subject or script into timed scenes with narration and visual prompts."""

from __future__ import annotations

import json
import re
from typing import Any, Callable


def _chunks(script: str) -> list[str]:
    text = str(script or "").replace("\r\n", "\n").strip()
    if not text:
        return []
    parts = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if len(parts) > 1:
        return parts
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    return sentences or [text]


def _scene(index: int, narration: str, style: str, duration: int) -> dict[str, Any]:
    spoken = " ".join(narration.split())
    visual = spoken
    if style:
        visual = f"{style}. {spoken}"
    return {
        "index": index,
        "narration": spoken[:1200],
        "visual_prompt": visual[:1500],
        "duration": duration,
    }


def scenes_from_text(
    script: str,
    subject: str,
    style: str,
    duration: int,
    count: int,
) -> list[dict[str, Any]]:
    limit = max(1, min(int(count), 6))
    pieces = _chunks(script) or _chunks(subject)
    if not pieces:
        pieces = ["A short original scene."]
    if len(pieces) > limit:
        pieces = pieces[:limit]
    elif len(pieces) < limit and len(pieces) == 1 and limit > 1:
        pieces = pieces * 1
    return [_scene(index, text, style, duration) for index, text in enumerate(pieces)]


def parse_storyboard(raw: str, style: str, duration: int, count: int) -> list[dict[str, Any]] | None:
    text = str(raw or "")
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    rows = payload.get("scenes") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        return None
    scenes = []
    for row in rows[: max(1, min(int(count), 6))]:
        if not isinstance(row, dict):
            continue
        narration = str(row.get("narration") or row.get("script") or "").strip()
        visual = str(row.get("visual_prompt") or row.get("prompt") or narration).strip()
        if not narration and not visual:
            continue
        if style and style.lower() not in visual.lower():
            visual = f"{style}. {visual}"
        try:
            seconds = int(row.get("duration") or duration)
        except (TypeError, ValueError):
            seconds = duration
        scenes.append({
            "index": len(scenes),
            "narration": (narration or visual)[:1200],
            "visual_prompt": visual[:1500],
            "duration": seconds if seconds > 0 else duration,
        })
    return scenes or None


def build_storyboard(
    *,
    subject: str,
    script: str,
    style: str,
    duration: int,
    count: int,
    complete: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Ask the language model for scenes, then fall back to the script text."""
    limit = max(1, min(int(count), 6))
    source = "script"
    scenes = None
    if complete is not None:
        prompt = (
            "Return JSON only. Create a timed storyboard for an original video. "
            f"Use at most {limit} scenes. Each scene duration is {int(duration)} seconds. "
            "Each scene needs narration and a visual_prompt for text-to-video. "
            "Do not mention stock footage, photos, or slideshows. "
            f"Style: {style or 'cinematic'}. Subject: {subject or script}. "
            f"Script: {script or subject}."
        )
        try:
            scenes = parse_storyboard(complete(prompt), style, int(duration), limit)
            if scenes:
                source = "llm"
        except Exception:
            scenes = None
    if not scenes:
        scenes = scenes_from_text(script, subject, style, int(duration), limit)
    spoken = "\n\n".join(scene["narration"] for scene in scenes)
    return {"scenes": scenes, "script": spoken, "source": source}
