"""``parse_mode`` threads from ``krepis.alerts.publish`` to the Telegram wire.

alpha-engine-config-I9925. These assert what reaches ``send_message`` — its
keyword arguments — rather than which internal function was invoked, and the
end-to-end case asserts the HTTP payload itself. Kept in its own module
because ``test_alerts.py`` trips the session DLP content scan on an unrelated
fixture; nothing here depends on that file.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from krepis import alerts
from krepis import telegram as tg
from krepis.secrets import clear_cache


@pytest.fixture(autouse=True)
def _reset_secrets_cache():
    clear_cache()
    yield
    clear_cache()


def _capture_send():
    sent: dict = {}

    def _send(msg, **kwargs):
        sent["message"] = msg
        sent.update(kwargs)
        return True

    return sent, _send


class TestPublishTelegramThreadsParseMode:
    def test_default_does_not_name_a_parse_mode_so_existing_callers_are_unchanged(self):
        # The transport's default IS Markdown; not forwarding it keeps every
        # existing call shape (and every existing test double) identical.
        sent, _send = _capture_send()
        with patch.dict("sys.modules", {"krepis.telegram": MagicMock(send_message=_send)}):
            alerts._publish_telegram("m", severity="info")
        assert "parse_mode" not in sent

    def test_explicit_markdown_is_also_not_forwarded(self):
        sent, _send = _capture_send()
        with patch.dict("sys.modules", {"krepis.telegram": MagicMock(send_message=_send)}):
            alerts._publish_telegram("m", severity="info", parse_mode="Markdown")
        assert "parse_mode" not in sent

    def test_html_reaches_send_message(self):
        sent, _send = _capture_send()
        with patch.dict("sys.modules", {"krepis.telegram": MagicMock(send_message=_send)}):
            alerts._publish_telegram("m", severity="info", parse_mode="HTML")
        assert sent["parse_mode"] == "HTML"

    def test_none_reaches_send_message_as_none(self):
        sent, _send = _capture_send()
        with patch.dict("sys.modules", {"krepis.telegram": MagicMock(send_message=_send)}):
            alerts._publish_telegram("m", severity="info", parse_mode=None)
        assert "parse_mode" in sent
        assert sent["parse_mode"] is None

    def test_parse_mode_does_not_disturb_the_push_decision(self):
        sent, _send = _capture_send()
        with patch.dict("sys.modules", {"krepis.telegram": MagicMock(send_message=_send)}):
            alerts._publish_telegram("m", severity="info", silent=False, parse_mode="HTML")
        assert sent["disable_notification"] is False


class TestPublishEndToEnd:
    """Through the real ``krepis.telegram`` to the HTTP payload."""

    @pytest.fixture
    def configured(self, monkeypatch):
        monkeypatch.setenv("ALPHA_ENGINE_SECRETS_SOURCE", "env")
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token-abc123")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")
        # publish() suppresses real channels while PYTEST_CURRENT_TEST is set;
        # this is the documented re-enable for a test that mocks the transport.
        monkeypatch.setenv("ALPHA_ENGINE_ALLOW_TEST_ALERTS", "1")

    def test_publish_html_payload(self, configured):
        with patch.object(tg.requests, "post") as post:
            post.return_value = MagicMock(status_code=200, text="ok")
            result = alerts.publish(
                "<b>LADDER</b> " + tg.escape_html("phase<0>"),
                severity="info",
                source="crucible/report.morning",
                sns=False,
                dedup_key=None,
                parse_mode="HTML",
                raise_on_total_failure=False,
            )
        assert post.call_count == 1
        payload = post.call_args.kwargs["json"]
        assert payload["parse_mode"] == "HTML"
        assert payload["text"] == "[INFO] crucible/report.morning: <b>LADDER</b> phase&lt;0&gt;"
        assert result.telegram.ok is True
