"""Autonomous scene generation with a mocked OpenRouter video API. No live charges."""

import threading

import pytest

from app.services import autonomous, openrouter_video
from test.services.test_openrouter_video import HEYGEN, _Response


class _Jobs:
    def __init__(self):
        self.calls = []
        self._n = 0
        self._lock = threading.Lock()

    def get(self, url, headers=None, timeout=None):
        with self._lock:
            self.calls.append(("GET", url))
        if url.endswith("/videos/models"):
            return _Response(200, {"data": [HEYGEN]})
        if url.endswith("/content?index=0"):
            return _Response(200, content=b"\x00" * 2048)
        job_id = url.rstrip("/").split("/")[-1]
        return _Response(200, {"id": job_id, "status": "completed"})

    def post(self, url, headers=None, json=None, timeout=None):
        with self._lock:
            self._n += 1
            job_id = f"job-{self._n}"
            self.calls.append(("POST", url, json, job_id))
        return _Response(202, {"id": job_id, "status": "pending"})


class _State:
    def __init__(self, task):
        self.task = task

    def get_task(self, _task_id):
        return self.task

    def patch_task(self, _task_id, **kwargs):
        self.task.update(kwargs)
        return True


def _params(**overrides):
    values = {
        "openrouter_video_model": "heygen/heygen-video-1",
        "openrouter_video_resolution": "480p",
        "openrouter_video_duration": 5,
        "project_budget_usd": 1,
        "video_aspect": "9:16",
        "video_subject": "a paper boat",
        "visual_style": "quiet",
        "paragraph_number": 1,
        "max_scene_retries": 1,
        "video_clip_duration": 5,
        "video_concat_mode": "random",
    }
    values.update(overrides)

    class Params:
        pass

    params = Params()
    for key, value in values.items():
        setattr(params, key, value)
    return params


def _board():
    return {
        "scenes": [{"index": 0, "narration": "A paper boat drifts.", "visual_prompt": "A paper boat on a pond.", "duration": 5}],
        "script": "A paper boat drifts.",
        "source": "script",
    }


@pytest.fixture
def ready(monkeypatch, tmp_path):
    monkeypatch.setattr(openrouter_video, "resolve_api_key", lambda: "test-key")
    monkeypatch.setattr("app.utils.utils.task_dir", lambda _task_id: str(tmp_path))
    state = _State({"storyboard": _board(), "autonomous_jobs": []})
    monkeypatch.setattr(autonomous.sm, "state", state)
    monkeypatch.setattr(
        autonomous.cv_engine,
        "validate_playable",
        lambda path: {"playable": __import__("os").path.isfile(path), "frames": 2, "width": 8, "height": 8, "codec_opened": True},
    )
    session = _Jobs()
    return session, state


def _accept(_path, _prompt):
    return {"accepted": True, "score": 0.8, "warnings": [], "refinement": "", "metrics": {"semantic_similarity": 0.8}}


def test_project_quote_fits_several_scenes_and_does_not_submit(ready):
    session, _state = ready
    priced = openrouter_video.quote(
        {
            "model": "heygen/heygen-video-1",
            "resolution": "480p",
            "aspect_ratio": "9:16",
            "duration": 5,
            "budget_usd": "1.00",
            "project_budget": True,
        },
        session=session,
    )
    assert priced["estimated_usd"] == "0.10"
    assert priced["budget_usd"] == "1.00"
    assert priced["max_scenes"] == 6
    assert priced["requires_approval"] is False
    assert not any(call[0] == "POST" for call in session.calls)


def test_over_budget_scene_is_not_submitted(ready):
    session, _state = ready
    with pytest.raises(openrouter_video.OpenRouterVideoBudgetError):
        autonomous.generate_for_task(
            "task-1",
            _params(project_budget_usd=0.05),
            session=session,
            sleep=lambda _seconds: None,
            clock=lambda: 0,
            analyze=_accept,
            complete=lambda _prompt: "{}",
        )
    assert not any(call[0] == "POST" for call in session.calls)


def test_rejected_scene_retries_once_then_stops(ready):
    session, state = ready
    calls = {"n": 0}

    def analyze(_path, _prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            return {
                "accepted": False,
                "score": 0.2,
                "warnings": ["weak match"],
                "refinement": "Show the boat closer.",
                "metrics": {},
                "components": [{"id": "clip", "name": "OpenCLIP ViT-B-32", "status": "failed", "detail": "similarity 0.10, below 0.24."}],
            }
        return {"accepted": True, "score": 0.9, "warnings": [], "refinement": "", "metrics": {}}

    paths = autonomous.generate_for_task(
        "task-1",
        _params(project_budget_usd=0.20),
        session=session,
        sleep=lambda _seconds: None,
        clock=lambda: 0,
        analyze=analyze,
    )
    posts = [call for call in session.calls if call[0] == "POST"]
    assert len(posts) == 2
    assert len(paths) == 1
    assert state.task["autonomous_terminal"] == "completed"
    assert state.task["autonomous_spent_usd"] == "0.20"
    assert "not a guarantee" in state.task["cv_report"]["disclaimer"]
    assert state.task["scene_status"][0]["components"][0]["id"] == "clip"
    assert state.task["scene_status"][0]["take"] == 0


def test_saved_job_is_resumed_without_another_post(ready, tmp_path):
    session, state = ready
    state.task["autonomous_jobs"] = [{
        "scene": 0,
        "take": 0,
        "job_id": "job-saved",
        "path": "",
        "status": "submitted",
        "estimated_usd": "0.10",
    }]
    paths = autonomous.generate_for_task(
        "task-1",
        _params(),
        session=session,
        sleep=lambda _seconds: None,
        clock=lambda: 0,
        analyze=_accept,
    )
    assert paths
    assert not any(call[0] == "POST" for call in session.calls)
    assert any("job-saved" in call[1] for call in session.calls if call[0] == "GET")
    assert (tmp_path / "scene-0-take-0.mp4").is_file()


def test_rejected_takes_are_not_assembled(ready):
    session, state = ready

    def reject(_path, _prompt):
        return {"accepted": False, "score": 0.1, "warnings": ["weak match"], "refinement": "Show the boat closer.", "metrics": {}}

    with pytest.raises(autonomous.AutonomousError, match="kept"):
        autonomous.generate_for_task(
            "task-1",
            _params(project_budget_usd=0.30, max_scene_retries=1),
            session=session,
            sleep=lambda _seconds: None,
            clock=lambda: 0,
            analyze=reject,
        )
    posts = [call for call in session.calls if call[0] == "POST"]
    assert len(posts) == 3
    assert state.task["autonomous_terminal"] == "failed"
    assert state.task["scene_status"][0]["status"] == "rejected"
    assert "weak match" in state.task["scene_status"][0]["warnings"]
    assert state.task["scene_status"][0]["components"] == []


def test_a_rejected_scene_is_retried_after_the_other_scenes(ready):
    session, state = ready
    state.task["storyboard"] = {
        "scenes": [
            {"index": 0, "narration": "A paper boat drifts.", "visual_prompt": "A paper boat on a pond.", "duration": 5},
            {"index": 1, "narration": "The boat reaches the reeds.", "visual_prompt": "A paper boat beside reeds.", "duration": 5},
        ],
        "script": "A paper boat drifts.",
        "source": "script",
    }
    pond = {"n": 0}

    def analyze(_path, prompt):
        if "beside reeds" in prompt:
            return {"accepted": True, "score": 0.9, "warnings": [], "refinement": "", "metrics": {}}
        if prompt.startswith("A paper boat on a pond"):
            pond["n"] += 1
            if pond["n"] <= 2:
                return {"accepted": False, "score": 0.2, "warnings": ["weak match"], "refinement": "Show the boat closer.", "metrics": {}}
        return {"accepted": True, "score": 0.9, "warnings": [], "refinement": "", "metrics": {}}

    paths = autonomous.generate_for_task(
        "task-1",
        _params(project_budget_usd=1, max_scene_retries=1),
        session=session,
        sleep=lambda _seconds: None,
        clock=lambda: 0,
        analyze=analyze,
    )
    posts = [call for call in session.calls if call[0] == "POST"]
    prompts = [call[2]["prompt"] for call in posts]
    assert len(posts) == 4
    assert any(prompt.startswith("A paper boat beside reeds") for prompt in prompts)
    assert sum(prompt.startswith("A paper boat on a pond") for prompt in prompts) == 3
    assert len(paths) == 2
    assert "scene-0-take-2.mp4" in paths[0].replace("\\", "/")
    assert "scene-1-take-0.mp4" in paths[1].replace("\\", "/")
    assert state.task["autonomous_terminal"] == "completed"


def test_subtitle_time_selects_the_clip_prompt(ready, tmp_path):
    session, state = ready
    srt = tmp_path / "subtitle.srt"
    srt.write_text(
        "1\n00:00:00,000 --> 00:00:05,000\nMercury is the closest planet.\n\n"
        "2\n00:00:05,000 --> 00:00:10,000\nNeptune is the farthest planet.\n",
        encoding="utf-8",
    )

    def complete(prompt):
        if "visual_prompt" not in prompt:
            return "{}"
        return (
            '{"scenes":['
            '{"index":0,"visual_prompt":"Mercury, a small grey planet."},'
            '{"index":1,"visual_prompt":"Neptune, a deep blue ice giant."},'
            '{"index":2,"visual_prompt":"Neptune remains in frame."}'
            "]}"
        )

    autonomous.generate_for_task(
        "task-1",
        _params(project_budget_usd=1, max_scene_retries=0, visual_style=""),
        audio_duration=10,
        subtitle_path=str(srt),
        session=session,
        sleep=lambda _seconds: None,
        clock=lambda: 0,
        analyze=_accept,
        complete=complete,
    )
    prompts = [call[2]["prompt"] for call in session.calls if call[0] == "POST"]
    neptune = next(prompt for prompt in prompts if prompt.startswith("Neptune, a deep blue"))
    assert len(prompts) == 3
    assert any(prompt.startswith("Mercury") for prompt in prompts)
    assert "Mercury" not in neptune
    assert all("New camera angle" not in prompt for prompt in prompts)
    assert state.task["storyboard"]["source"] == "timeline"


def test_narration_coverage_fails_before_any_submission_when_the_budget_is_short(ready):
    session, _state = ready
    with pytest.raises(openrouter_video.OpenRouterVideoBudgetError):
        autonomous.generate_for_task(
            "task-1",
            _params(project_budget_usd=0.20),
            audio_duration=12,
            session=session,
            sleep=lambda _seconds: None,
            clock=lambda: 0,
            analyze=_accept,
        )
    assert not any(call[0] == "POST" for call in session.calls)


def test_coverage_plan_uses_supported_clip_length():
    scenes = autonomous.scenes_for_coverage(
        [{"narration": "A paper boat drifts.", "visual_prompt": "A paper boat on a pond.", "duration": 5}],
        12.1,
        5,
    )
    assert len(scenes) == 3
    assert scenes[0]["duration"] == 5
    assert "New camera angle" in scenes[2]["visual_prompt"]


def test_second_run_does_not_submit_while_the_first_holds_the_task(ready):
    session, _state = ready
    autonomous._running.add("task-1")
    try:
        with pytest.raises(autonomous.AutonomousError):
            autonomous.generate_for_task(
                "task-1",
                _params(),
                session=session,
                sleep=lambda _seconds: None,
                clock=lambda: 0,
                analyze=_accept,
            )
    finally:
        autonomous._running.discard("task-1")
    assert not any(call[0] == "POST" for call in session.calls)
