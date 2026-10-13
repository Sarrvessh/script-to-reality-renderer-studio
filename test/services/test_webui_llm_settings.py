from pathlib import Path
from unittest.mock import patch

import requests
from streamlit.testing.v1 import AppTest

from app.config import config
from app.services import kimi_code_oauth
from app.services import llm
from webui import kimi_login


ROOT_DIR = Path(__file__).parent.parent.parent
WEBUI_MAIN = ROOT_DIR / "webui" / "Main.py"


def _widget_by_key(elements, key):
    return next(
        item
        for item in elements
        if str(getattr(item, "key", "")) == key
        or str(getattr(item, "key", "")).startswith(f"{key}_")
    )


def _click_kimi_login_link(app, key="kimi_code_login_button"):
    # AppTest does not yet provide LinkButton.click; send the real widget the same
    # trigger_value the browser sends, and do not replace the production link_button with an ordinary button.
    link = next(item for item in app.get("link_button") if item.proto.id.endswith(key))
    states = app._tree.get_widget_states()
    states.widgets.add(id=link.proto.id, trigger_value=True)
    return app._run(states)


def _make_kimi_poll_due(app):
    flow = dict(app.session_state[kimi_login.FLOW_KEY])
    flow["next_poll_at"] = 0
    app.session_state[kimi_login.FLOW_KEY] = flow
    return app.run()


def test_fluxionai_settings_defaults_and_connection_button():
    """Actually run Streamlit widgets: check defaults, config overrides, and the connection-test entry, and do not write user config."""
    app_config = dict(
        config.app,
        llm_provider="fluxionai",
        fluxionai_api_key="",
        fluxionai_base_url="",
        fluxionai_model_name="",
    )
    with (
        patch.object(config, "app", app_config),
        patch.object(config, "ui", dict(config.ui, language="zh")),
        patch.object(config, "try_save_config", return_value=True),
        patch.object(
            llm, "test_connection", return_value=(True, "", 0.1)
        ) as test_connection,
    ):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=60)
        app.session_state["ui_language"] = "zh"
        app.session_state["settings_dialog_open"] = True
        app.run()
        assert not app.exception
        assert _widget_by_key(app.selectbox, "llm_provider_select").value == "fluxionai"
        assert (
            _widget_by_key(app.text_input, "fluxionai_base_url_custom_input").value
            == "https://fluxionai.space/v1"
        )
        assert (
            _widget_by_key(app.text_input, "fluxionai_model_name_input").value
            == "gpt-5.5"
        )
        assert app_config["fluxionai_base_url"] == ""
        assert app_config["fluxionai_model_name"] == ""
        assert any("OpenAI 接口分组" in str(item.value) for item in app.info)
        _widget_by_key(app.text_input, "fluxionai_api_key_input").set_value(
            "test-ui-key"
        ).run()
        _widget_by_key(app.text_input, "fluxionai_model_name_input").set_value(
            "custom-model"
        ).run()
        _widget_by_key(app.text_input, "fluxionai_base_url_custom_input").set_value(
            "https://gateway.example.com/v1"
        ).run()
        assert app_config["fluxionai_api_key"] == "test-ui-key"
        assert app_config["fluxionai_model_name"] == "custom-model"
        assert app_config["fluxionai_base_url"] == "https://gateway.example.com/v1"
        _widget_by_key(app.button, "test_llm_connection_button").click().run()
        test_connection.assert_called_once()
        assert not app.exception


def test_kimi_platform_selection_keeps_endpoint_configuration_consistent():
    """Switching the Kimi platform must sync the base URL and allow editing the address only in custom mode."""
    app_config = dict(
        config.app,
        llm_provider="moonshot",
        moonshot_api_key="",
        moonshot_base_url="",
        moonshot_model_name="",
    )
    ui_config = dict(config.ui, language="en")

    with (
        patch.object(config, "app", app_config),
        patch.object(config, "ui", ui_config),
        patch.object(config, "try_save_config", return_value=True),
        patch.object(
            llm,
            "test_connection",
            return_value=(False, "401 Invalid Authentication", 0.1),
        ),
    ):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=60)
        app.session_state["ui_language"] = "en"
        app.session_state["settings_dialog_open"] = True
        app.run()

        assert [str(item.value) for item in app.exception] == []
        endpoint_select = _widget_by_key(
            app.selectbox,
            "moonshot_service_endpoint_select",
        )
        global_base_url = _widget_by_key(
            app.text_input,
            "moonshot_base_url_global_input",
        )
        assert endpoint_select.value == "global"
        assert global_base_url.value == "https://api.moonshot.ai/v1"
        assert global_base_url.disabled is True
        assert app_config["moonshot_base_url"] == "https://api.moonshot.ai/v1"
        global_tips = "\n".join(str(item.value) for item in app.info)
        assert (
            "https://platform.kimi.ai?track_id=track-9e3b711aa2594e378f6fe5b8de718a76&aff=script-to-reality"
            in global_tips
        )
        assert (
            "https://platform.kimi.ai/docs/models?track_id=track-9e3b711aa2594e378f6fe5b8de718a76&aff=script-to-reality"
            in global_tips
        )
        assert "offer ends December 31, 2026" in global_tips

        endpoint_select.set_value("china").run()
        china_base_url = _widget_by_key(
            app.text_input,
            "moonshot_base_url_china_input",
        )
        assert china_base_url.value == "https://api.moonshot.cn/v1"
        assert china_base_url.disabled is True
        # The China site is the registry's compatibility default and should not be written into user config again.
        assert app_config["moonshot_base_url"] == ""
        china_tips = "\n".join(str(item.value) for item in app.info)
        assert (
            "https://platform.kimi.com?track_id=track-6eec1e56a4494e52adcaebbcbbefce59&aff=script-to-reality"
            in china_tips
        )
        assert (
            "https://platform.kimi.com/docs/models?track_id=track-6eec1e56a4494e52adcaebbcbbefce59&aff=script-to-reality"
            in china_tips
        )

        endpoint_select = _widget_by_key(
            app.selectbox,
            "moonshot_service_endpoint_select",
        )
        endpoint_select.set_value("custom").run()
        custom_base_url = _widget_by_key(
            app.text_input,
            "moonshot_base_url_custom_input",
        )
        assert custom_base_url.value == ""
        assert custom_base_url.disabled is False
        custom_base_url.set_value("https://gateway.example.com/v1").run()
        assert app_config["moonshot_base_url"] == "https://gateway.example.com/v1"

        endpoint_select = _widget_by_key(
            app.selectbox,
            "moonshot_service_endpoint_select",
        )
        endpoint_select.set_value("global").run()
        _widget_by_key(app.button, "test_llm_connection_button").click().run()
        error_messages = [str(item.value) for item in app.error]
        assert any("platform.kimi.ai" in message for message in error_messages)
        assert any("api.moonshot.ai" in message for message in error_messages)
        assert any(
            "401 Invalid Authentication" in message for message in error_messages
        )


def test_configure_llm_link_opens_settings_on_llm_tab():
    """The shortcut next to the video topic should open and land on LLM settings in one click."""
    with patch.object(config, "try_save_config", return_value=True):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=60)
        app.session_state["ui_language"] = "en"
        app.run()

        _widget_by_key(app.button, "open_llm_settings_from_subject").click().run()

        assert [str(item.value) for item in app.exception] == []
        assert app.session_state["settings_dialog_open"] is True
        assert app.session_state["settings_dialog_tabs_en"] == "LLM Settings"
        # The business target is only for one directed open. Consume it immediately after render so a later ordinary "Settings" entry
        # is not forced back to the LLM tab by a historical target.
        assert "settings_dialog_target_tab" not in app.session_state


def test_material_settings_target_uses_localized_tab_state_and_is_consumed():
    """The material shortcut stores a stable business ID and resolves the current language label at render time."""
    with patch.object(config, "try_save_config", return_value=True):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=60)
        app.session_state["ui_language"] = "zh"
        app.session_state["settings_dialog_open"] = True
        app.session_state["settings_dialog_target_tab"] = "material"
        app.run()

        assert [str(item.value) for item in app.exception] == []
        assert app.session_state["settings_dialog_tabs_zh"] == "素材来源设置"
        assert "settings_dialog_target_tab" not in app.session_state


def test_ai_video_settings_prioritize_sponsors_and_own_shengsuan_key():
    """Video providers should be shown in the agreed sponsor order, and the Shengsuanyun key is managed only in settings."""
    app_config = dict(
        config.app,
        llm_provider="openai",
        script_generation_backend="loomloom",
        video_source="pexels",
        loomloom_api_token="initial-token",
    )
    ui_config = dict(config.ui, language="en")

    with (
        patch.object(config, "app", app_config),
        patch.object(config, "ui", ui_config),
        patch.object(config, "try_save_config", return_value=True),
    ):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=60)
        app.session_state["ui_language"] = "en"
        app.session_state["settings_dialog_open"] = True
        app.session_state["settings_dialog_target_tab"] = "material"
        app.run()

        markdown_values = [str(item.value) for item in app.markdown]
        provider_titles = [
            "**Metaso · MiniMax H3**",
            "**OfoxAI**",
            "**Shengsuan Cloud AI Video**",
            "**Volcano Engine Ark · Seedance**",
            "**WaveSpeed**",
        ]
        provider_positions = [
            next(
                index
                for index, value in enumerate(markdown_values)
                if value.startswith(title)
            )
            for title in provider_titles
        ]
        assert provider_positions == sorted(provider_positions)

        settings_token = _widget_by_key(
            app.text_input,
            "loomloom_api_token_input",
        )
        assert settings_token.value == "initial-token"
        assert all(item.key != "loomloom_user_api_token" for item in app.text_input)

        settings_token.set_value("settings-token").run()
        assert app_config["loomloom_api_token"] == "settings-token"
        assert (
            _widget_by_key(
                app.text_input,
                "loomloom_api_token_input",
            ).value
            == "settings-token"
        )
        assert [str(item.value) for item in app.exception] == []


def test_kimi_code_device_authorization_network_error_is_recoverable():
    """Device-authorization request timeout: the page does not raise, it shows a network error, and the login button stays so the user can retry."""
    # The local config may hold real Kimi credentials and must be stripped, or the panel shows the authorized state immediately.
    app_config = {
        key: value
        for key, value in dict(config.app, llm_provider="kimi_code").items()
        if not key.startswith("kimi_code_")
    }
    app_config["llm_provider"] = "kimi_code"

    with (
        patch.object(config, "app", app_config),
        patch.object(config, "ui", dict(config.ui, language="en")),
        patch.object(config, "try_save_config", return_value=True),
        patch.object(
            kimi_code_oauth,
            "request_device_code",
            side_effect=requests.exceptions.ConnectTimeout("boom"),
        ),
    ):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=60)
        app.session_state["ui_language"] = "en"
        app.session_state["settings_dialog_open"] = True
        app.run()
        assert not app.exception

        _widget_by_key(app.button, "kimi_code_login_button").click().run()

        assert not app.exception
        error_messages = [str(item.value) for item in app.error]
        assert any("network error" in message.lower() for message in error_messages)
        # The flow was not established, the login button is still there, and the user can retry directly.
        assert "kimi_code_oauth_flow" not in app.session_state
        _widget_by_key(app.button, "kimi_code_login_button")


def test_kimi_code_poll_network_error_retries_then_offers_restart():
    """A network error during polling does not clear the flow: retry automatically, and after consecutive failures return to a state where login can start again."""
    app_config = {
        key: value
        for key, value in dict(config.app, llm_provider="kimi_code").items()
        if not key.startswith("kimi_code_")
    }
    app_config["llm_provider"] = "kimi_code"
    device_payload = {
        "device_code": "dc-1",
        "user_code": "ABCD-EFGH",
        "verification_uri_complete": "https://www.kimi.com/code/authorize_device?user_code=ABCD-EFGH",
        "interval": 2,
    }

    with (
        patch.object(config, "app", app_config),
        patch.object(config, "ui", dict(config.ui, language="en")),
        patch.object(config, "try_save_config", return_value=True),
        patch.object(
            kimi_code_oauth, "request_device_code", return_value=device_payload
        ),
        patch.object(
            kimi_code_oauth,
            "poll_token",
            side_effect=requests.exceptions.ReadTimeout("boom"),
        ) as poll_mock,
        patch.object(kimi_code_oauth, "store_token_bundle") as store_mock,
    ):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=60)
        app.session_state["ui_language"] = "en"
        app.session_state["settings_dialog_open"] = True
        app.run()
        assert not app.exception

        _widget_by_key(app.selectbox, "kimi_code_service_endpoint_select").set_value(
            "china"
        ).run()
        _click_kimi_login_link(app)
        assert poll_mock.call_count == 0
        for _ in range(5):
            _make_kimi_poll_due(app)

        # Each timer tick makes only one request; after consecutive failures, end the flow and show a new login link.
        assert not app.exception
        assert poll_mock.call_count == 5
        store_mock.assert_not_called()
        assert "kimi_code_oauth_flow" not in app.session_state
        error_messages = [str(item.value) for item in app.error]
        assert any("network error" in message.lower() for message in error_messages)
        assert any(
            item.proto.id.endswith("kimi_code_login_button")
            for item in app.get("link_button")
        )


def test_kimi_code_pending_then_success_completes_flow():
    """After a pending poll, keep waiting; once authorization succeeds, write the credentials and end the flow."""
    app_config = {
        key: value
        for key, value in dict(config.app, llm_provider="kimi_code").items()
        if not key.startswith("kimi_code_")
    }
    app_config["llm_provider"] = "kimi_code"
    device_payload = {
        "device_code": "dc-1",
        "user_code": "ABCD-EFGH",
        "verification_uri_complete": "https://www.kimi.com/code/authorize_device?user_code=ABCD-EFGH",
        "interval": 2,
    }
    token_payload = {"access_token": "at-1", "refresh_token": "rt-1", "expires_in": 900}

    with (
        patch.object(config, "app", app_config),
        patch.object(config, "ui", dict(config.ui, language="en")),
        patch.object(config, "try_save_config", return_value=True),
        patch.object(
            kimi_code_oauth, "request_device_code", return_value=device_payload
        ),
        patch.object(
            kimi_code_oauth,
            "poll_token",
            side_effect=[{"pending": True}, token_payload],
        ),
        patch.object(kimi_code_oauth, "store_token_bundle") as store_mock,
    ):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=60)
        app.session_state["ui_language"] = "en"
        app.session_state["settings_dialog_open"] = True
        app.run()
        assert not app.exception

        _widget_by_key(app.selectbox, "kimi_code_service_endpoint_select").set_value(
            "china"
        ).run()
        _click_kimi_login_link(app)
        _make_kimi_poll_due(app)
        assert kimi_login.FLOW_KEY in app.session_state
        store_mock.assert_not_called()
        _make_kimi_poll_due(app)

        assert not app.exception
        store_mock.assert_called_once_with(
            "china",
            token_payload,
            expected_generation=kimi_code_oauth.authorization_generation(),
        )
        assert "kimi_code_oauth_flow" not in app.session_state


def test_kimi_login_opens_selected_region_and_waits_until_clicked():
    """The button already carries the correct site link; selecting a provider does not start polling or request another authorization code."""
    app_config = {
        key: value
        for key, value in config.app.items()
        if not key.startswith("kimi_code_")
    }
    app_config["llm_provider"] = "kimi_code"

    def issue_code(region):
        return {
            "device_code": f"device-{region}",
            "user_code": "TEST-CODE",
            "verification_uri_complete": f"https://www.kimi.{'com' if region == 'china' else 'ai'}/code/authorize_device?user_code=TEST-CODE",
            "interval": 5,
            "expires_in": 600,
        }

    with (
        patch.object(config, "app", app_config),
        patch.object(config, "ui", dict(config.ui, language="en")),
        patch.object(config, "try_save_config", return_value=True),
        patch.object(
            kimi_code_oauth, "request_device_code", side_effect=issue_code
        ) as issue,
        patch.object(
            kimi_code_oauth, "poll_token", return_value={"pending": True}
        ) as poll,
    ):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=60)
        app.session_state["ui_language"] = "en"
        app.session_state["settings_dialog_open"] = True
        app.run()
        app.run()
        assert issue.call_count == 1
        poll.assert_not_called()
        assert kimi_login.FLOW_KEY not in app.session_state
        link = next(
            item
            for item in app.get("link_button")
            if item.proto.id.endswith("kimi_code_login_button")
        )
        assert link.proto.type == "primary"
        assert link.proto.ignore_rerun is False
        assert link.proto.url.startswith("https://www.kimi.ai/")

        _widget_by_key(app.selectbox, "kimi_code_service_endpoint_select").set_value(
            "china"
        ).run()
        assert issue.call_count == 2
        link = next(
            item
            for item in app.get("link_button")
            if item.proto.id.endswith("kimi_code_login_button")
        )
        assert link.proto.url.startswith("https://www.kimi.com/")
        _click_kimi_login_link(app)
        poll.assert_not_called()
        assert app.session_state[kimi_login.FLOW_KEY]["region"] == "china"


def test_kimi_poll_slow_down_and_cancel_do_not_busy_loop():
    """Honor the server poll interval; after cancel, do not request a token with the old authorization code."""
    app_config = {
        key: value
        for key, value in config.app.items()
        if not key.startswith("kimi_code_")
    }
    app_config["llm_provider"] = "kimi_code"
    payload = {
        "device_code": "review-device",
        "user_code": "TEST-CODE",
        "verification_uri_complete": "https://www.kimi.com/code/authorize_device?user_code=TEST-CODE",
        "interval": 5,
        "expires_in": 600,
    }
    with (
        patch.object(config, "app", app_config),
        patch.object(config, "ui", dict(config.ui, language="en")),
        patch.object(config, "try_save_config", return_value=True),
        patch.object(kimi_code_oauth, "request_device_code", return_value=payload),
        patch.object(
            kimi_code_oauth,
            "poll_token",
            return_value={"pending": True, "slow_down": True},
        ) as poll,
    ):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=60)
        app.session_state["ui_language"] = "en"
        app.session_state["settings_dialog_open"] = True
        app.run()
        _click_kimi_login_link(app)
        _make_kimi_poll_due(app)
        assert poll.call_count == 1
        assert app.session_state[kimi_login.FLOW_KEY]["interval"] == 10
        app.run()
        assert poll.call_count == 1
        _widget_by_key(app.button, "kimi_code_cancel_login_button").click().run()
        assert not app.exception
        assert kimi_login.FLOW_KEY not in app.session_state
        assert poll.call_count == 1


def test_kimi_login_does_not_prepare_links_on_other_settings_tabs():
    """Opening material settings does not request an unrelated authorization code."""
    app_config = {
        key: value
        for key, value in config.app.items()
        if not key.startswith("kimi_code_")
    }
    app_config["llm_provider"] = "kimi_code"
    with (
        patch.object(config, "app", app_config),
        patch.object(config, "ui", dict(config.ui, language="en")),
        patch.object(config, "try_save_config", return_value=True),
        patch.object(
            kimi_code_oauth,
            "request_device_code",
            side_effect=requests.exceptions.ConnectTimeout("test"),
        ) as issue,
    ):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=60)
        app.session_state["ui_language"] = "en"
        app.session_state["settings_dialog_open"] = True
        app.session_state["settings_dialog_target_tab"] = "material"
        app.run()
        assert not app.exception
        issue.assert_not_called()
