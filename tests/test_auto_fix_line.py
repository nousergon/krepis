"""Every Telegram alert carries an ``Auto-fix:`` line (Brian, 2026-10-23).

Rule: nous-ergon-ops observability-policy.md § Notification — auto-fix line.
The transport guarantees the line, so a sender cannot leave it out; these
tests are the guard that keeps it that way.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from krepis import alerts
from krepis import telegram as tg
from krepis.secrets import clear_cache


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    clear_cache()
    monkeypatch.setenv("ALPHA_ENGINE_SECRETS_SOURCE", "env")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "1")
    monkeypatch.setattr(tg.fleet_events, "emit_alert_event", MagicMock(return_value=True))
    yield
    clear_cache()


@pytest.fixture
def post():
    with patch.object(tg.requests, "post") as mocked:
        mocked.return_value = MagicMock(status_code=200, text="ok")
        yield mocked


def _drop_resolution(text: str) -> str:
    """Remove the ``Resolution:`` line the transport adds beside ``Auto-fix:``.

    That line has its own guard (``tests/test_telegram.py::TestResolutionLine``);
    these tests pin the Auto-fix half only.
    """
    return "\n".join(
        line for line in text.split("\n")
        if not tg.has_resolution_line(line) and not tg._ACTION_RE.search(line)
    )


def _sent(post) -> str:
    return _drop_resolution(post.call_args.kwargs["json"]["text"])


class TestDetection:
    @pytest.mark.parametrize("text", [
        "Auto-fix: retrying at 06:10",
        "*Auto-fix:* retrying",
        "<b>Auto-fix:</b> retrying",
        "🔧 Auto-fix: retrying",
        "- auto-fix: retrying",
        "head\n  Auto-Fix : retrying",
    ])
    def test_caller_line_is_recognised(self, text):
        assert tg.has_auto_fix_line(text)

    @pytest.mark.parametrize("text", ["", "boom", "no auto-fix: here mid-line"])
    def test_absent_line_is_not_invented(self, text):
        assert not tg.has_auto_fix_line(text)


class TestSendMessage:
    def test_default_line_is_appended_when_sender_declared_nothing(self, post):
        tg.send_message("boom")
        assert _sent(post).endswith(f"*Auto-fix:* {tg.DEFAULT_AUTO_FIX}")

    def test_explicit_auto_fix_is_rendered(self, post):
        tg.send_message("boom", auto_fix="retrying at 06:10 UTC")
        assert _sent(post) == "boom\n\n*Auto-fix:* retrying at 06:10 UTC"

    def test_blank_auto_fix_still_gets_the_default(self, post):
        tg.send_message("boom", auto_fix="   ")
        assert _sent(post).endswith(tg.DEFAULT_AUTO_FIX)

    def test_callers_own_line_is_kept_and_not_duplicated(self, post):
        tg.send_message("boom\nAuto-fix: self-heals on next run", auto_fix="ignored")
        sent = _sent(post)
        assert sent.lower().count("auto-fix") == 1
        assert "self-heals on next run" in sent

    def test_html_line_is_escaped(self, post):
        tg.send_message("<b>boom</b>", parse_mode="HTML", auto_fix="a < b")
        assert _sent(post) == "<b>boom</b>\n\n<b>Auto-fix:</b> a &lt; b"

    def test_plain_text_line(self, post):
        tg.send_message("boom", parse_mode=None, auto_fix="x")
        assert _sent(post) == "boom\n\nAuto-fix: x"

    @pytest.mark.parametrize("mode", ["Markdown", "HTML", None])
    def test_line_survives_truncation(self, post, mode):
        tg.send_message("z" * 9000, parse_mode=mode, auto_fix="retrying")
        sent = _sent(post)
        assert sent.endswith("retrying")
        assert "truncated" in sent
        assert len(sent) <= tg.TELEGRAM_MESSAGE_MAX_CHARS

    def test_callers_line_at_the_tail_survives_truncation(self, post):
        tg.send_message("z" * 9000 + "\nAuto-fix: filed as aec#1", parse_mode=None)
        sent = _sent(post)
        assert sent.endswith("Auto-fix: filed as aec#1")
        assert len(sent) <= tg.TELEGRAM_MESSAGE_MAX_CHARS

    def test_bus_event_body_carries_the_line(self, post):
        tg.send_message("boom", auto_fix="retrying")
        body = tg.fleet_events.emit_alert_event.call_args.kwargs["body"]
        assert _drop_resolution(body).endswith("Auto-fix:* retrying")
        assert tg._ACTION_RE.search(body)

    def test_rollup_passes_auto_fix_through(self, post):
        tg.send_rollup(["a"], auto_fix="none needed")
        assert _sent(post).endswith("*Auto-fix:* none needed")


@pytest.fixture
def publish_env(monkeypatch):
    for name in alerts.SCHEDULED_CONTEXT_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("JOURNAL_STREAM", "8:1")  # scheduled: no provenance suffix
    monkeypatch.setenv("ALPHA_ENGINE_ALLOW_TEST_ALERTS", "1")
    monkeypatch.setattr(alerts, "_fetch_source_mutes", lambda *_a, **_k: [])
    monkeypatch.setattr(alerts.fleet_events, "emit_alert_event", MagicMock(return_value=True))
    monkeypatch.setattr(alerts, "_check_dedup_marker", lambda *_a, **_k: (False, ""))
    monkeypatch.setattr(alerts, "_write_dedup_marker", lambda *_a, **_k: None)
    # Keep the tier registry offline and uncached, so nothing leaks into
    # other test modules.
    s3 = MagicMock()
    s3.get_object.side_effect = Exception("no S3 in tests")
    fake_boto3 = MagicMock()
    fake_boto3.client.return_value = s3
    monkeypatch.setitem(__import__("sys").modules, "boto3", fake_boto3)
    monkeypatch.setattr(alerts.alert_tiers, "consecutive_count", lambda *_a, **_k: 0)
    monkeypatch.setattr(alerts.alert_tiers, "record_open", lambda *_a, **_k: None)
    monkeypatch.setattr(alerts.alert_tiers, "clear_streak", lambda *_a, **_k: None)
    alerts.alert_tiers.reset_cache()
    yield
    alerts.alert_tiers.reset_cache()


def _publish(**kwargs):
    with patch.object(alerts, "_publish_sns", return_value=alerts.ChannelResult(ok=True, detail="ok")) as sns, \
         patch.object(alerts, "_publish_telegram", return_value=alerts.ChannelResult(ok=True, detail="ok")) as tgm, \
         patch.object(alerts, "_resolve_sns_topic_arn", return_value="arn:aws:sns:us-east-1:1:t"), \
         patch.object(alerts, "_resolve_log_chat", return_value=(None, None)):
        alerts.publish(**kwargs)
    return _drop_resolution(sns.call_args.args[1]), _drop_resolution(tgm.call_args.args[0])


class TestPublish:
    def test_auto_fix_reaches_both_channels(self, publish_env):
        sns_text, tg_text = _publish(message="boom", source="x", auto_fix="retrying at 06:10")
        assert sns_text.endswith("*Auto-fix:* retrying at 06:10")
        assert tg_text.endswith("*Auto-fix:* retrying at 06:10")

    def test_error_default_is_the_sweep_line(self, publish_env):
        _sns, tg_text = _publish(message="boom", source="x", severity="error")
        assert tg_text.endswith(tg.DEFAULT_AUTO_FIX)

    def test_info_has_no_footer(self, publish_env):
        """Brian 2026-11-04: boilerplate under a report is noise."""
        sns_text, tg_text = _publish(message="report", source="x", severity="info")
        assert tg_text == "[INFO] x: report"
        assert sns_text == "[INFO] x: report"

    def test_recovery_has_no_footer(self, publish_env):
        _sns, tg_text = _publish(
            message="ok again", source="x", state=alerts.ALERT_STATE_CLEARED,
            severity="info", identity_key="k",
        )
        assert "Auto-fix:" not in tg_text

    def test_info_with_a_declared_action_keeps_its_footer(self, publish_env):
        _sns, tg_text = _publish(
            message="report", source="x", severity="info", action="read section 2",
        )
        assert "Auto-fix:" in tg_text

    def test_impact_and_action_reach_both_channels(self, publish_env):
        with patch.object(alerts, "_publish_sns", return_value=alerts.ChannelResult(ok=True, detail="ok")) as sns, \
             patch.object(alerts, "_publish_telegram", return_value=alerts.ChannelResult(ok=True, detail="ok")) as tgm, \
             patch.object(alerts, "_resolve_sns_topic_arn", return_value="arn:aws:sns:us-east-1:1:t"), \
             patch.object(alerts, "_resolve_log_chat", return_value=(None, None)):
            alerts.publish(message="boom", source="x", impact="no trades today",
                           auto_fix="retry 06:10", action="nothing")
        for text in (sns.call_args.args[1], tgm.call_args.args[0]):
            assert text.endswith(
                "*Impact:* no trades today\n*Auto-fix:* retry 06:10\n*You:* nothing"
            )

    def test_cli_impact_and_action_flags(self, publish_env):
        with patch.object(alerts, "publish") as pub:
            pub.return_value = alerts.PublishResult(
                sns=alerts.ChannelResult(ok=True, detail="ok"),
                telegram=alerts.ChannelResult(ok=True, detail="ok"),
            )
            alerts.main(["publish", "--message", "boom", "--impact", "i", "--action", "a"])
        assert pub.call_args.kwargs["impact"] == "i"
        assert pub.call_args.kwargs["action"] == "a"

    def test_cli_flag(self, publish_env):
        with patch.object(alerts, "publish") as pub:
            pub.return_value = alerts.PublishResult(
                sns=alerts.ChannelResult(ok=True, detail="ok"),
                telegram=alerts.ChannelResult(ok=True, detail="ok"),
            )
            alerts.main(["publish", "--message", "boom", "--auto-fix", "retrying"])
        assert pub.call_args.kwargs["auto_fix"] == "retrying"


class TestBriefWithDetailsUrl:
    """Brian, 2026-11-04: Telegram gets 2-3 sentences plus a console link."""

    LONG = (
        "Budget at 112% of the month. AWS is the driver. Spend jumped on 11-03. "
        "This fourth sentence is cut.\n\nPer-service table:\nEC2 $40\nS3 $12"
    )

    def _send(self, **kwargs):
        with patch.object(alerts, "_publish_sns", return_value=alerts.ChannelResult(ok=True, detail="ok")) as sns, \
             patch.object(alerts, "_publish_telegram", return_value=alerts.ChannelResult(ok=True, detail="ok")) as tgm, \
             patch.object(alerts, "_resolve_sns_topic_arn", return_value="arn:aws:sns:us-east-1:1:t"), \
             patch.object(alerts, "_resolve_log_chat", return_value=(None, None)):
            alerts.publish(**kwargs)
        return sns.call_args.args[1], tgm.call_args.args[0]

    def test_telegram_gets_brief_and_link_email_keeps_body(self, publish_env):
        sns_text, tg_text = self._send(
            message=self.LONG, source="costs", severity="warning",
            impact="the month ends ~$30 over", auto_fix="none; spend is a decision",
            action="review the costs page", details_url="https://console.nousergon.ai/page/costs",
        )
        assert tg_text.splitlines() == [
            "[WARNING] costs: Budget at 112% of the month. AWS is the driver. Spend jumped on 11-03.",
            "*Impact:* the month ends ~$30 over",
            "*Auto-fix:* none; spend is a decision",
            "*You:* review the costs page",
            "[Details on console](https://console.nousergon.ai/page/costs)",
        ]
        assert "EC2 $40" in sns_text and "EC2 $40" not in tg_text

    def test_brief_omits_you_when_no_action(self, publish_env):
        _sns, tg_text = self._send(
            message="Router fell back twice.", source="r", severity="warning",
            auto_fix="retries on its own", details_url="https://c/x",
        )
        assert "You:" not in tg_text and "Resolution:" not in tg_text

    def test_informational_brief_is_headline_and_link(self, publish_env):
        _sns, tg_text = self._send(
            message=self.LONG, source="report", severity="info", details_url="https://c/r",
        )
        assert tg_text.splitlines()[1:] == ["[Details on console](https://c/r)"]
        assert "Auto-fix" not in tg_text

    def test_long_body_without_link_is_logged(self, publish_env, caplog):
        with caplog.at_level("WARNING"):
            self._send(message="x" * 400, source="s", severity="warning")
        assert "ALERT_DETAIL_UNLINKED" in caplog.text

    def test_cli_details_url_flag(self, publish_env):
        with patch.object(alerts, "publish") as pub:
            pub.return_value = alerts.PublishResult(
                sns=alerts.ChannelResult(ok=True, detail="ok"),
                telegram=alerts.ChannelResult(ok=True, detail="ok"),
            )
            alerts.main(["publish", "--message", "boom", "--details-url", "https://c/p"])
        assert pub.call_args.kwargs["details_url"] == "https://c/p"


def test_brief_headline_cuts_long_single_sentence():
    out = tg.brief_headline("a" * 500)
    assert len(out) == tg.BRIEF_MAX_CHARS and out.endswith("…")


def test_details_line_html_escapes():
    assert tg.details_line("https://c/p?a=1&b=2", tg.PARSE_MODE_HTML) == (
        '<a href="https://c/p?a=1&amp;b=2">Details on console</a>'
    )
