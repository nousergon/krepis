"""
Unit tests for ``krepis.telegram``.

Locks down the Telegram-send contract: secret resolution, markdown escape,
``disable_notification`` flag propagation, fire-and-forget failure handling
(no exceptions ever propagate to caller), and the rollup helper's
empty-list / header / default-silent semantics.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from krepis import telegram as tg
from krepis.secrets import clear_cache


@pytest.fixture(autouse=True)
def _reset_secrets_cache():
    """Every test starts with an empty secrets cache."""
    clear_cache()
    yield
    clear_cache()


@pytest.fixture
def configured_env(monkeypatch):
    """Resolve TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID via env (skip SSM)."""
    monkeypatch.setenv("ALPHA_ENGINE_SECRETS_SOURCE", "env")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token-abc123")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")


@pytest.fixture
def mock_post():
    """Patch ``requests.post`` with a 200 success by default."""
    with patch.object(tg.requests, "post") as mocked:
        mocked.return_value = MagicMock(status_code=200, text="ok")
        yield mocked


# ── _escape_markdown ────────────────────────────────────────────────────────


class TestEscapeMarkdown:
    def test_escapes_underscore_backtick_brackets(self):
        result = tg._escape_markdown("a_b `c` [d]")
        assert result == "a-b 'c' (d)"

    def test_preserves_asterisk_for_bold(self):
        assert tg._escape_markdown("*bold*") == "*bold*"

    def test_empty_string_passes_through(self):
        assert tg._escape_markdown("") == ""


# ── send_message — happy path ───────────────────────────────────────────────


class TestSendMessageHappyPath:
    def test_returns_true_on_200(self, configured_env, mock_post):
        assert tg.send_message("hello") is True

    def test_calls_correct_telegram_endpoint(self, configured_env, mock_post):
        tg.send_message("hello")
        mock_post.assert_called_once()
        url = mock_post.call_args.args[0]
        assert url == "https://api.telegram.org/bottest-token-abc123/sendMessage"

    def test_payload_shape(self, configured_env, mock_post):
        tg.send_message("hello world")
        payload = mock_post.call_args.kwargs["json"]
        assert payload == {
            "chat_id": "12345",
            "text": "hello world",
            "parse_mode": "Markdown",
            "disable_notification": False,
        }

    def test_timeout_is_5_seconds(self, configured_env, mock_post):
        tg.send_message("hello")
        assert mock_post.call_args.kwargs["timeout"] == 5

    def test_escapes_markdown_in_text(self, configured_env, mock_post):
        tg.send_message("ticker_AAPL [BUY]")
        payload = mock_post.call_args.kwargs["json"]
        assert payload["text"] == "ticker-AAPL (BUY)"

    def test_preserves_bold_markers(self, configured_env, mock_post):
        tg.send_message("*BUY AAPL*")
        payload = mock_post.call_args.kwargs["json"]
        assert payload["text"] == "*BUY AAPL*"


# ── send_message — disable_notification flag ────────────────────────────────


class TestDisableNotification:
    def test_defaults_false(self, configured_env, mock_post):
        tg.send_message("loud")
        assert mock_post.call_args.kwargs["json"]["disable_notification"] is False

    def test_true_propagates(self, configured_env, mock_post):
        tg.send_message("silent", disable_notification=True)
        assert mock_post.call_args.kwargs["json"]["disable_notification"] is True

    def test_false_propagates_explicitly(self, configured_env, mock_post):
        tg.send_message("loud", disable_notification=False)
        assert mock_post.call_args.kwargs["json"]["disable_notification"] is False

    def test_message_thread_id_propagates(self, configured_env, mock_post):
        tg.send_message("topic msg", message_thread_id=42)
        assert mock_post.call_args.kwargs["json"]["message_thread_id"] == 42

    def test_message_thread_id_omitted_when_unset(self, configured_env, mock_post):
        tg.send_message("plain")
        assert "message_thread_id" not in mock_post.call_args.kwargs["json"]

    def test_explicit_bot_token_and_chat_id_override_secrets(
        self, configured_env, mock_post, monkeypatch
    ):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        assert tg.send_message("hello", bot_token="override-token", chat_id=-10099) is True
        url = mock_post.call_args.args[0]
        assert "override-token" in url
        assert mock_post.call_args.kwargs["json"]["chat_id"] == -10099


# ── send_message — secret resolution failures ───────────────────────────────


class TestSecretResolution:
    def test_missing_token_returns_false_no_api_call(self, monkeypatch, mock_post):
        monkeypatch.setenv("ALPHA_ENGINE_SECRETS_SOURCE", "env")
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")
        assert tg.send_message("hello") is False
        mock_post.assert_not_called()

    def test_missing_chat_id_returns_false_no_api_call(self, monkeypatch, mock_post):
        monkeypatch.setenv("ALPHA_ENGINE_SECRETS_SOURCE", "env")
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        assert tg.send_message("hello") is False
        mock_post.assert_not_called()

    def test_both_missing_returns_false(self, monkeypatch, mock_post):
        monkeypatch.setenv("ALPHA_ENGINE_SECRETS_SOURCE", "env")
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        assert tg.send_message("hello") is False
        mock_post.assert_not_called()


# ── send_message — failure modes never raise ────────────────────────────────


class TestFailureSwallowing:
    def test_http_non_200_returns_false(self, configured_env, mock_post):
        mock_post.return_value = MagicMock(status_code=400, text="Bad Request")
        assert tg.send_message("hello") is False

    def test_http_500_returns_false(self, configured_env, mock_post):
        mock_post.return_value = MagicMock(status_code=500, text="Internal Server Error")
        assert tg.send_message("hello") is False

    def test_timeout_returns_false(self, configured_env, mock_post):
        mock_post.side_effect = requests.Timeout("timed out")
        assert tg.send_message("hello") is False

    def test_connection_error_returns_false(self, configured_env, mock_post):
        mock_post.side_effect = requests.ConnectionError("DNS failed")
        assert tg.send_message("hello") is False

    def test_arbitrary_request_exception_returns_false(self, configured_env, mock_post):
        mock_post.side_effect = requests.RequestException("anything")
        assert tg.send_message("hello") is False

    def test_response_with_no_text_attr_does_not_crash(self, configured_env, mock_post):
        # Some response shapes have empty text; truncation logic must not blow up.
        mock_post.return_value = MagicMock(status_code=400, text="")
        assert tg.send_message("hello") is False


# ── send_rollup ─────────────────────────────────────────────────────────────


class TestSendRollup:
    def test_empty_findings_returns_true_no_api_call(self, configured_env, mock_post):
        assert tg.send_rollup([]) is True
        mock_post.assert_not_called()

    def test_single_finding_renders_as_bullet(self, configured_env, mock_post):
        tg.send_rollup(["AMAT untouched 14 days"])
        payload = mock_post.call_args.kwargs["json"]
        assert payload["text"] == "- AMAT untouched 14 days"

    def test_multiple_findings_render_as_bullets(self, configured_env, mock_post):
        tg.send_rollup(["one", "two", "three"])
        payload = mock_post.call_args.kwargs["json"]
        assert payload["text"] == "- one\n- two\n- three"

    def test_header_prepended_as_bold(self, configured_env, mock_post):
        tg.send_rollup(["finding"], header="Surveillance Digest")
        payload = mock_post.call_args.kwargs["json"]
        assert payload["text"] == "*Surveillance Digest*\n- finding"

    def test_defaults_to_silent_delivery(self, configured_env, mock_post):
        tg.send_rollup(["finding"])
        assert mock_post.call_args.kwargs["json"]["disable_notification"] is True

    def test_disable_notification_false_propagates(self, configured_env, mock_post):
        tg.send_rollup(["urgent"], disable_notification=False)
        assert mock_post.call_args.kwargs["json"]["disable_notification"] is False

    def test_rollup_escapes_markdown_in_findings(self, configured_env, mock_post):
        tg.send_rollup(["ticker_X hit [support]"])
        payload = mock_post.call_args.kwargs["json"]
        # Escape applied at send_message layer, so '_' and '[]' are rewritten.
        assert "ticker-X hit (support)" in payload["text"]

    def test_rollup_returns_false_when_secrets_missing(self, monkeypatch, mock_post):
        monkeypatch.setenv("ALPHA_ENGINE_SECRETS_SOURCE", "env")
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        assert tg.send_rollup(["finding"]) is False
        mock_post.assert_not_called()


class TestErrorBodyLogging:
    """Non-200 bodies must never be logged raw: an HTML/proxy error page can
    echo the request URL, which embeds the bot token (CodeQL
    py/clear-text-logging-sensitive-data, krepis-PR32)."""

    def test_telegram_json_description_is_logged(self, configured_env, mock_post, caplog):
        mock_post.return_value.status_code = 400
        mock_post.return_value.text = '{"ok":false,"error_code":400,"description":"Bad Request: parse error"}'
        assert tg.send_message("x") is False
        assert "Bad Request: parse error" in caplog.text

    def test_non_json_body_is_suppressed(self, configured_env, mock_post, caplog):
        leaky = "<html>404 https://api.telegram.org/botSECRET-TOKEN/sendMessage</html>"
        mock_post.return_value.status_code = 404
        mock_post.return_value.text = leaky
        assert tg.send_message("x") is False
        assert "SECRET-TOKEN" not in caplog.text
        assert "<non-JSON body suppressed>" in caplog.text
