"""Turn a subject or script into timed scenes with narration and visual prompts."""

from __future__ import annotations

import json
import math
import os
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


_SRT_STAMP = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)")


def _stamp_seconds(stamp: str) -> float:
    match = _SRT_STAMP.search(stamp)
    if not match:
        return 0.0
    hours, minutes, seconds, fraction = match.groups()
    millis = int(fraction[:3].ljust(3, "0"))
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + millis / 1000


def parse_srt(text: str) -> list[tuple[float, float, str]]:
    cues = []
    for block in re.split(r"\n\s*\n", str(text or "").strip()):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        timing = next((line for line in lines if "-->" in line), "")
        if not timing:
            continue
        left, right = timing.split("-->", 1)
        spoken = " ".join(lines[lines.index(timing) + 1 :]).strip()
        if spoken:
            cues.append((_stamp_seconds(left), _stamp_seconds(right), spoken))
    return cues


def load_srt(path: str) -> list[tuple[float, float, str]]:
    if not path or not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return parse_srt(handle.read())


def _words_in_window(
    words: list[str],
    cue_start: float,
    cue_end: float,
    window_start: float,
    window_end: float,
) -> list[str]:
    span = max(cue_end - cue_start, 1e-6)
    chosen = []
    for index, word in enumerate(words):
        midpoint = cue_start + span * (index + 0.5) / len(words)
        if window_start <= midpoint < window_end:
            chosen.append(word)
    return chosen


def scenes_from_cues(
    cues: list[tuple[float, float, str]],
    clip_seconds: int,
    *,
    style: str = "",
    required_seconds: float = 0,
) -> list[dict[str, Any]]:
    """One provider-length clip for each slice of the spoken timeline."""
    clip = max(1, int(clip_seconds))
    required = max(float(clip), float(required_seconds or 0))
    count = max(1, math.ceil(required / clip - 1e-9))
    scenes = []
    previous = ""
    for index in range(count):
        start = index * clip
        end = min(required, start + clip)
        words: list[str] = []
        for cue_start, cue_end, text in cues:
            if cue_end <= start or cue_start >= end:
                continue
            words.extend(_words_in_window(text.split(), cue_start, cue_end, start, end))
        spoken = " ".join(words).strip() or (f"Continue showing: {previous}" if previous else "Continue the scene")
        scene = _scene(index, spoken, style, clip)
        scene["start"] = round(start, 3)
        scene["end"] = round(end, 3)
        scenes.append(scene)
        previous = spoken
    return scenes


def apply_shot_prompts(
    scenes: list[dict[str, Any]],
    style: str,
    complete: Callable[[str], str] | None,
) -> list[dict[str, Any]]:
    """Turn each spoken slice into a shot description. The spoken words stay unchanged."""
    if not scenes or complete is None:
        return scenes
    payload = [{"index": scene["index"], "narration": scene["narration"]} for scene in scenes]
    prompt = (
        "Return JSON only: {\"scenes\":[{\"index\":0,\"visual_prompt\":\"...\"}]}. "
        "You are describing footage for a factual narration. "
        "Write one text-to-video shot for each narration. Show only what that narration says, in concrete filmable detail. "
        "Do not invent people, places, objects, or events that the line does not mention. "
        "Do not add subjects from other scenes and do not rewrite the narration. "
        "Do not use clickbait, slogans, camera-move lists, stock-footage wording, or slideshows. One clear shot per scene. "
        f"Style: {style or 'cinematic'}. Scenes: {json.dumps(payload, ensure_ascii=False)}"
    )
    try:
        raw = str(complete(prompt) or "")
        start = raw.find("{")
        end = raw.rfind("}")
        parsed = json.loads(raw[start : end + 1]) if start >= 0 and end > start else {}
    except Exception:
        return scenes
    rows = parsed.get("scenes") if isinstance(parsed, dict) else None
    if not isinstance(rows, list):
        return scenes
    shots = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            shots[int(row.get("index"))] = str(row.get("visual_prompt") or row.get("prompt") or "").strip()
        except (TypeError, ValueError):
            continue
    for scene in scenes:
        visual = shots.get(int(scene["index"]), "")
        if not visual:
            continue
        if style and style.lower() not in visual.lower():
            visual = f"{style}. {visual}"
        scene["visual_prompt"] = visual[:1500]
    return scenes


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
            "Return JSON only. You are planning an original factual video. "
            f"Use at most {limit} scenes. Each scene duration is {int(duration)} seconds. "
            "Each scene needs narration and a visual_prompt for text-to-video. "
            "Narration must be speakable, accurate, and free of titles or stage directions. "
            "Each visual_prompt must show only what that scene's narration says, in concrete filmable detail. "
            "Do not invent events. Do not mention stock footage, photos, or slideshows. "
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
