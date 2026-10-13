from pathlib import Path
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

from app.config import config
from app.services import state as sm
from app.services import version_checker, voice


WEBUI_MAIN = Path(__file__).resolve().parents[2] / "webui" / "Main.py"


def _selectbox(app, key):
    """A language switch rebuilds widgets with a locale suffix; look them up by a stable key."""
    return next(
        item for item in app.selectbox
        if item.key == key or str(item.key).startswith(f"{key}_")
    )


@pytest.fixture
def catalan_app():
    # Use a credential-free config, an empty task list, and a save function that cannot write disk, so the UI test does not read
    # the user's tasks or change the real config. Keep the built-in voice catalog and check the actual Catalan voices.
    with (
        patch.object(config, "app", {"llm_provider": "openai", "video_source": "pexels"}),
        patch.object(config, "ui", {
            "language": "ca", "voice_mode": "tts", "tts_server": "azure-tts-v1",
            "voice_name": "ca-ES-JoanaNeural-Female", "video_language": "ca-ES",
        }),
        patch.object(config, "try_save_config", return_value=True),
        patch.object(sm.state, "get_all_tasks", return_value=([], 0)),
        # Pin it as already finished with no update so a real GitHub request and background refresh do not start,
        # and do not read or write the process-level version-check cache, keeping language-switch and dialog tests deterministic.
        patch.object(
            version_checker,
            "poll_available_update",
            return_value=version_checker.UpdateCheckSnapshot(complete=True),
        ),
    ):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=60)
        app.session_state["ui_language"] = "ca"
        app.run()
        assert not app.exception
        yield app


def test_catalan_language_switch_preserves_script_and_script_language(catalan_app):
    """Actually run a Streamlit rerun: one switch takes effect, and the body and generation language are not lost with the UI."""
    app = catalan_app
    assert "Català" in _selectbox(app, "top_language_code_selector").options
    assert _selectbox(app, "script_language_select").label == "Idioma del guió"
    assert _selectbox(app, "script_language_select").value == "ca-ES"
    contents = {
        "video_subject": "L'Àlex aprèn amb intel·ligència artificial",
        "video_script": "La intel·ligència artificial ens ajuda a aprendre. L'àudio també és útil.",
        "video_terms": "escola, estudiants, tecnologia",
    }
    for key, value in contents.items():
        next(item for item in app.text_area if item.key == key).set_value(value).run()
    for language, label in (("en", "Script Language"), ("ca", "Idioma del guió")):
        _selectbox(app, "top_language_code_selector").set_value(language).run()
        assert not app.exception
        assert app.session_state["ui_language"] == language
        assert _selectbox(app, "script_language_select").label == label
        assert _selectbox(app, "script_language_select").value == "ca-ES"
        for key, value in contents.items():
            assert next(item for item in app.text_area if item.key == key).value == value


def test_catalan_builtin_voices_and_settings_dialog(catalan_app):
    """Both built-in voices can be selected; the settings dialog opens normally in the new language."""
    assert set(voice.get_all_azure_voices(["ca-ES"])) >= {
        "ca-ES-EnricNeural-Male", "ca-ES-JoanaNeural-Female",
    }
    app = catalan_app
    assert any("Joana" in option for item in app.selectbox for option in item.options)
    next(item for item in app.button if item.key == "open_settings_dialog_button").click().run()
    assert not app.exception
    assert any(tab.label == "Model de llenguatge" for tab in app.tabs)
    assert any(tab.label == "Orígens del material" for tab in app.tabs)
