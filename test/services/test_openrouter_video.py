"""OpenRouter video pricing and job flow. These tests never call the live API."""

import json

import pytest
import requests

from app.services import openrouter_video


HEYGEN = {
    "id": "heygen/heygen-video-1",
    "name": "HeyGen: HeyGen Video",
    "supported_durations": list(range(5, 16)),
    "supported_resolutions": ["480p", "768p", "2K"],
    "supported_aspect_ratios": ["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"],
    "pricing_skus": {
        "duration_seconds_480p": "0.02",
        "duration_seconds_768p": "0.03",
        "duration_seconds_2k": "0.09",
        "reference_duration_seconds_480p": "0.04",
    },
}


class _Response:
    def __init__(self, status_code, payload=None, content=b""):
        self.status_code = status_code
        self._payload = payload
        self.content = content

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


class _Session:
    def __init__(self, job_status="completed", post_error=None):
        self.calls = []
        self.job_status = job_status
        self.post_error = post_error

    def get(self, url, headers=None, timeout=None):
        self.calls.append(("GET", url))
        assert headers["Authorization"].startswith("Bearer ")
        assert "test-key" not in url
        if url.endswith("/videos/models"):
            return _Response(200, {"data": [HEYGEN]})
        if url.endswith("/content?index=0"):
            return _Response(200, content=b"\x00" * 2048)
        return _Response(200, {"id": "job-1", "status": self.job_status, "error": "upstream failed"})

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append(("POST", url, json))
        if self.post_error:
            raise self.post_error
        return _Response(202, {"id": "job-1", "status": "pending"})


def _params(**overrides):
    values = {
        "openrouter_video_approved": False,
        "openrouter_video_model": "heygen/heygen-video-1",
        "openrouter_video_resolution": "480p",
        "openrouter_video_duration": 5,
        "openrouter_video_budget_usd": 0.10,
        "video_aspect": "9:16",
        "video_script": "A quiet street at dawn.",
        "video_subject": "dawn",
    }
    values.update(overrides)

    class Params:
        pass

    params = Params()
    for key, value in values.items():
        setattr(params, key, value)
    return params


@pytest.fixture
def session(monkeypatch):
    monkeypatch.setattr(openrouter_video, "resolve_api_key", lambda: "test-key")
    client = _Session()
    return client


def test_quote_prices_one_heygen_clip_and_does_not_submit(session):
    priced = openrouter_video.quote(
        {"model": "heygen/heygen-video-1", "resolution": "480p", "duration": 5, "aspect_ratio": "9:16"},
        session=session,
    )

    assert priced["estimated_usd"] == "0.10"
    assert priced["budget_usd"] == "0.10"
    assert priced["within_budget"] is True
    assert priced["submitted"] is False
    assert priced["rate_usd"] == "0.02"
    assert [call[0] for call in session.calls] == ["GET"]


def test_a_higher_requested_budget_cannot_raise_the_ceiling(session):
    priced = openrouter_video.quote(
        {
            "model": "heygen/heygen-video-1",
            "resolution": "768p",
            "duration": 5,
            "aspect_ratio": "9:16",
            "budget_usd": "5.00",
        },
        session=session,
    )

    assert priced["estimated_usd"] == "0.15"
    assert priced["budget_usd"] == "0.10"
    assert priced["within_budget"] is False
    assert priced["submitted"] is False
    assert [call[0] for call in session.calls] == ["GET"]


def test_unapproved_generation_does_not_submit(session, tmp_path, monkeypatch):
    monkeypatch.setattr(openrouter_video.utils, "task_dir", lambda _task_id: str(tmp_path))
    with pytest.raises(openrouter_video.OpenRouterVideoApprovalRequired):
        openrouter_video.generate_for_task("task-1", _params(), session=session)
    assert session.calls == []


def test_over_budget_generation_does_not_submit(session, tmp_path, monkeypatch):
    monkeypatch.setattr(openrouter_video.utils, "task_dir", lambda _task_id: str(tmp_path))
    with pytest.raises(openrouter_video.OpenRouterVideoBudgetError):
        openrouter_video.generate_for_task(
            "task-1",
            _params(openrouter_video_approved=True, openrouter_video_resolution="768p"),
            session=session,
        )
    assert [call[0] for call in session.calls] == ["GET"]


def test_approved_generation_submits_once_and_saves_the_mp4(session, tmp_path, monkeypatch):
    monkeypatch.setattr(openrouter_video.utils, "task_dir", lambda _task_id: str(tmp_path))
    remembered = {}
    monkeypatch.setattr(
        openrouter_video.sm.state,
        "patch_task",
        lambda task_id, **details: remembered.update(task_id=task_id, **details),
    )

    paths = openrouter_video.generate_for_task(
        "task-1",
        _params(openrouter_video_approved=True),
        session=session,
        sleep=lambda _seconds: None,
        clock=lambda: 0,
    )

    posts = [call for call in session.calls if call[0] == "POST"]
    assert len(posts) == 1
    assert posts[0][2]["model"] == "heygen/heygen-video-1"
    assert posts[0][2]["duration"] == 5
    assert posts[0][2]["resolution"] == "480p"
    assert posts[0][2]["aspect_ratio"] == "9:16"
    assert paths == [str(tmp_path / "openrouter-video.mp4")]
    assert (tmp_path / "openrouter-video.mp4").stat().st_size == 2048
    assert remembered["openrouter_video_job_id"] == "job-1"


def test_failed_job_is_not_submitted_again(session):
    session.job_status = "failed"
    with pytest.raises(openrouter_video.OpenRouterVideoError, match="not submitted again"):
        openrouter_video.generate_for_task(
            "task-1",
            _params(openrouter_video_approved=True),
            session=session,
            sleep=lambda _seconds: None,
            clock=lambda: 0,
        )
    assert [call[0] for call in session.calls].count("POST") == 1


def test_post_timeout_is_not_retried(session):
    session.post_error = requests.Timeout("timed out")
    with pytest.raises(openrouter_video.OpenRouterVideoError, match="not sent again"):
        openrouter_video.generate_for_task(
            "task-1",
            _params(openrouter_video_approved=True),
            session=session,
        )
    assert [call[0] for call in session.calls].count("POST") == 1


def test_reference_price_is_not_used_for_text_to_video():
    with pytest.raises(openrouter_video.OpenRouterVideoError, match="no text-to-video price"):
        openrouter_video.estimate_cost(
            {"pricing_skus": {"reference_duration_seconds_480p": "0.04"}},
            5,
            "480p",
        )


def test_content_url_stays_on_openrouter():
    assert openrouter_video._content_url("job-1", ["https://storage.example.com/video.mp4"]).endswith(
        "/videos/job-1/content?index=0"
    )
    kept = openrouter_video._content_url(
        "job-1",
        ["https://openrouter.ai/api/v1/videos/job-1/content?index=0"],
    )
    assert kept == "https://openrouter.ai/api/v1/videos/job-1/content?index=0"


def test_quote_response_has_no_secret_fields(session):
    priced = openrouter_video.quote({"model": "heygen/heygen-video-1"}, session=session)
    assert "api_key" not in json.dumps(priced)
    assert "test-key" not in json.dumps(priced)
