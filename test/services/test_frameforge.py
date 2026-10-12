import json

import pytest

from app.models.schema import MaterialInfo
from app.services import frameforge


def test_split_narration_keeps_paragraphs_and_groups_sentences():
    paragraphs = frameforge.split_narration("First scene stays whole.\n\nSecond scene stays whole.")
    assert paragraphs == ["First scene stays whole.", "Second scene stays whole."]

    sentences = frameforge.split_narration("One. Two. Three. Four. Five. Six.")
    assert 2 <= len(sentences) <= 4
    assert sentences[0].startswith("One")
    assert sentences[-1].endswith("Six.")


def test_assign_terms_follows_scene_order_and_fills_gaps():
    terms = frameforge.assign_terms(
        ["Robots learn quickly.", "Data centers stay cold."],
        ["robot arm"],
    )
    assert terms[0] == "robot arm"
    assert "data" in terms[1].lower() or "centers" in terms[1].lower()


def test_project_roundtrip_and_secret_free_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(frameforge, "_storage_override", str(tmp_path))
    monkeypatch.setattr(frameforge, "_write_model_name", lambda name: None)

    project = frameforge.create_project("Three fascinating facts about artificial intelligence")
    loaded = frameforge.get_project(project["id"])
    assert loaded["topic"].startswith("Three fascinating")
    assert loaded["voice_name"] == "en-US-AriaNeural-Female"
    assert loaded["aspect"] == "9:16"

    with pytest.raises(ValueError):
        frameforge.update_settings({"openrouter_model_name": "sk-not-a-model"})

    settings = frameforge.public_settings()
    # Edge voice ids use locale prefixes such as sk-SK. Those are not secrets.
    without_voices = {key: value for key, value in settings.items() if key != "voices"}
    blob = json.dumps(without_voices)
    assert "sk-" not in blob
    assert "api_key" not in json.dumps(settings)
    assert settings["openrouter_configured"] in (True, False)
    assert settings["pexels_configured"] in (True, False)
    assert settings["product"] == "Script-to-Reality Renderer Studio"


def test_storyboard_patch_rejects_foreign_clip_and_ignores_task_id(tmp_path, monkeypatch):
    monkeypatch.setattr(frameforge, "_storage_override", str(tmp_path))
    project = frameforge.create_project("Local machines")
    saved = frameforge.apply_project_patch(project, {
        "task_id": "not-allowed",
        "script": "A small computer trains a model.",
        "scenes": [{
            "narration": "A small computer trains a model.",
            "term": "laptop desk",
        }],
    })
    assert saved["task_id"] == ""
    assert saved["scenes"][0]["term"] == "laptop desk"
    with pytest.raises(ValueError, match="Pexels"):
        frameforge.normalize_scenes([{
            "narration": "A desk.",
            "term": "desk",
            "clip": {"url": "https://example.com/secret.mp4"},
        }])


def test_render_request_keeps_scene_order_and_caption_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(frameforge, "_storage_override", str(tmp_path))
    project = frameforge.create_project("Artificial intelligence")
    project["scenes"] = frameforge.normalize_scenes([
        {"narration": "Robots learn from examples.", "term": "robot arm"},
        {"narration": "Data centers power the models.", "term": "server room"},
    ])
    request = frameforge.build_task_request(project)
    assert request.video_source == "pexels"
    assert list(request.video_terms) == ["robot arm", "server room"]
    assert request.match_materials_to_script is True
    assert getattr(request.video_concat_mode, "value", request.video_concat_mode) == "sequential"
    assert getattr(request.video_transition_mode, "value", request.video_transition_mode) == "FadeIn"
    assert "Robots learn from examples." in request.video_script
    assert request.video_script.index("Robots") < request.video_script.index("Data centers")
    assert request.subtitle_enabled is True
    assert request.subtitle_display_mode == "sentence"
    assert request.text_background_color == "#141210"
    assert request.font_name

    locked = frameforge.build_task_request(project, [MaterialInfo(provider="local", url="clip.mp4", duration=4)])
    assert locked.video_source == "local"
    assert locked.video_materials[0].url == "clip.mp4"
