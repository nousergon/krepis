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
        assert result == "a\\_b \\`c\\` \\[d\\]"

    def test_the_identifier_survives_escaping(self):
        """alpha-engine-config-I7168: this SUBSTITUTED until 2026-08-13, so a
        box-health alert named `/home/ec2-user/flow-doctor/flow-doctor.db` — a
        path that does not exist, while two OTHER files on that box really are
        named `flow-doctor.db`. An identifier the reader is expected to copy
        into a command must arrive intact."""
        escaped = tg._escape_markdown("/home/ec2-user/flow-doctor/flow_doctor.db")
        assert escaped.replace("\\", "") == "/home/ec2-user/flow-doctor/flow_doctor.db"
        assert "flow-doctor.db" not in escaped.replace("\\", "").rsplit("/", 1)[-1]

    def test_a_backslash_in_the_input_cannot_neutralise_an_escape(self):
        """Escaping `_` after a caller-supplied backslash would emit `\\_`
        meaning a literal backslash followed by an unescaped underscore."""
        assert tg._escape_markdown("a\\_b") == "a\\\\\\_b"

    def test_preserves_asterisk_for_bold(self):
        assert tg._escape_markdown("*bold*") == "*bold*"

    def test_empty_string_passes_through(self):
        assert tg._escape_markdown("") == ""


# ── _truncate_for_telegram (config-I3301) ───────────────────────────────────


class TestTruncateForTelegram:
    def test_short_text_passes_through_unchanged(self):
        assert tg._truncate_for_telegram("hello") == "hello"

    def test_text_at_exact_limit_passes_through_unchanged(self):
        text = "x" * tg.TELEGRAM_MESSAGE_MAX_CHARS
        assert tg._truncate_for_telegram(text) == text

    def test_text_over_limit_is_truncated_to_max_chars(self):
        text = "x" * (tg.TELEGRAM_MESSAGE_MAX_CHARS + 500)
        result = tg._truncate_for_telegram(text)
        assert len(result) == tg.TELEGRAM_MESSAGE_MAX_CHARS

    def test_truncated_result_keeps_the_head(self):
        text = "HEAD-IDENTIFYING-SUMMARY " + ("x" * (tg.TELEGRAM_MESSAGE_MAX_CHARS + 500))
        result = tg._truncate_for_telegram(text)
        assert result.startswith("HEAD-IDENTIFYING-SUMMARY")

    def test_truncated_result_notes_original_length(self):
        original_len = tg.TELEGRAM_MESSAGE_MAX_CHARS + 500
        text = "x" * original_len
        result = tg._truncate_for_telegram(text)
        assert str(original_len) in result
        assert "truncated" in result


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
        assert payload["text"] == "ticker\\_AAPL \\[BUY\\]"
        assert payload["text"].replace("\\", "") == "ticker_AAPL [BUY]"

    def test_preserves_bold_markers(self, configured_env, mock_post):
        tg.send_message("*BUY AAPL*")
        payload = mock_post.call_args.kwargs["json"]
        assert payload["text"] == "*BUY AAPL*"

    def test_oversized_text_is_truncated_before_send(self, configured_env, mock_post):
        # config-I3301: alert_on_failure.sh dumped 30 raw journal lines with
        # no length guard — Telegram rejected it (400) while a parallel SNS
        # publish succeeded, masking the failure. send_message must never
        # hand the API a payload over the hard limit.
        oversized = "🚨 substrate-health-daily.service failed. Last 30 journal lines:\n" + (
            "line of journal output\n" * 400
        )
        assert len(oversized) > tg.TELEGRAM_MESSAGE_MAX_CHARS
        tg.send_message(oversized)
        payload = mock_post.call_args.kwargs["json"]
        assert len(payload["text"]) <= tg.TELEGRAM_MESSAGE_MAX_CHARS
        assert payload["text"].startswith("🚨 substrate-health-daily.service failed")


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
        # Escape applied at the send_message layer; the finding survives it.
        assert "ticker\\_X hit \\[support\\]" in payload["text"]

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

    def test_token_in_json_description_is_redacted(self, configured_env, mock_post, caplog):
        mock_post.return_value.status_code = 502
        mock_post.return_value.text = '{"description":"upstream error for https://api.telegram.org/bottest-token-abc123/sendMessage"}'
        assert tg.send_message("x") is False
        assert "test-token-abc123" not in caplog.text
        assert "[REDACTED]" in caplog.text


# ── plain-text fallback on a Markdown entity parse failure ──────────────────


_ENTITY_400 = (
    '{"ok":false,"error_code":400,"description":"Bad Request: '
    "can't parse entities: Can't find end of the entity starting at byte "
    'offset 355"}'
)

# The live message shape that was dropped six times in 21 days: a caller
# template with intentional bold, into which a Step Functions failure cause was
# interpolated — and that cause carried git's fetch summary line. The lone `*`
# in ` * branch` makes the asterisk count odd, leaving a bold entity open.
# `_escape_markdown` cannot fix this without breaking every intentional bold.
_REAL_DROPPED_MESSAGE = (
    "*Fleet-SF Watch — AUTO-FIX*\n"
    "Weekly Freshness SF: FAILED\n"
    "Cause: `From https://github.com/nousergon/crucible-backtester\n"
    " * branch            main       -> FETCH_HEAD\n"
    "ssm_log_capture: ERROR: [backtester] failed (rc=75)`"
)


def _responses(*specs):
    """Sequence of mock responses, one per successive ``requests.post`` call."""
    return [MagicMock(status_code=code, text=body) for code, body in specs]


# ── parse_mode (alpha-engine-config-I9925) ──────────────────────────────────
#
# Every assertion here is about the WIRE PAYLOAD (`requests.post(..., json=)`),
# never about which Python function was called: a mocked transport proves the
# call, the payload proves the behaviour (krepis-PR199's `silent=False` was a
# no-op for a week because its tests asserted the call).


class TestParseModeHtml:
    def test_html_mode_sends_parse_mode_html(self, configured_env, mock_post):
        tg.send_message("<b>LADDER</b>", parse_mode="HTML")
        payload = mock_post.call_args.kwargs["json"]
        assert payload["parse_mode"] == "HTML"

    def test_html_mode_does_not_escape_the_callers_markup(self, configured_env, mock_post):
        # The caller OWNS the tags under HTML; escaping `<` here would destroy
        # the heading the caller built. Interpolated content is the caller's
        # to escape (see TestEscapeHtml).
        tg.send_message("<b>LADDER</b> phase_0 [x] `ok`", parse_mode="HTML")
        payload = mock_post.call_args.kwargs["json"]
        assert payload["text"] == "<b>LADDER</b> phase_0 [x] `ok`"

    def test_html_mode_applies_no_markdown_escaping(self, configured_env, mock_post):
        tg.send_message("ticker_AAPL [BUY]", parse_mode="HTML")
        payload = mock_post.call_args.kwargs["json"]
        assert "\\" not in payload["text"]

    def test_html_entity_error_is_retried_as_plain_text(self, configured_env, mock_post):
        # Telegram's HTML-mode parse failure carries the same marker as v1's,
        # so the plain-text redelivery covers both (gotcha 2 of I9925).
        mock_post.side_effect = _responses(
            (400, '{"ok":false,"error_code":400,"description":"Bad Request: '
                  'can\'t parse entities: Unsupported start tag \\"x\\" at byte offset 3"}'),
            (200, "ok"),
        )
        assert tg.send_message("<x>bad</x>", parse_mode="HTML") is True
        first = mock_post.call_args_list[0].kwargs["json"]
        retry = mock_post.call_args_list[1].kwargs["json"]
        assert first["parse_mode"] == "HTML"
        assert "parse_mode" not in retry
        assert retry["text"] == first["text"]


class TestParseModeNone:
    def test_none_omits_the_parse_mode_key_entirely(self, configured_env, mock_post):
        tg.send_message("plain *text* _here_", parse_mode=None)
        payload = mock_post.call_args.kwargs["json"]
        assert "parse_mode" not in payload
        # Absence, not `null`: Telegram rejects an explicit null.
        assert None not in payload.values()

    def test_none_applies_no_escaping(self, configured_env, mock_post):
        tg.send_message("a_b [c] `d` <e>", parse_mode=None)
        payload = mock_post.call_args.kwargs["json"]
        assert payload["text"] == "a_b [c] `d` <e>"

    def test_none_is_not_retried_on_a_parse_error(self, configured_env, mock_post):
        # Plain text was never sent with a parse mode, so there is nothing to
        # drop on retry; a 400 here is some other failure and is not repeated.
        mock_post.side_effect = _responses(
            (400, '{"ok":false,"description":"Bad Request: can\'t parse entities"}'),
        )
        assert tg.send_message("x", parse_mode=None) is False
        assert mock_post.call_count == 1

    def test_none_still_truncates(self, configured_env, mock_post):
        tg.send_message("y" * 5000, parse_mode=None)
        payload = mock_post.call_args.kwargs["json"]
        assert len(payload["text"]) <= tg.TELEGRAM_MESSAGE_MAX_CHARS
        assert "truncated" in payload["text"]


class TestParseModeMarkdownIsUnchanged:
    """The default is byte-identical to the pre-I9925 behaviour.

    `TestSendMessageHappyPath.test_payload_shape` already pins the exact
    payload for a default call; this class pins that passing the default
    EXPLICITLY produces the same bytes, so the new parameter is provably a
    no-op for every existing call site.
    """

    def test_explicit_markdown_equals_the_default_payload(self, configured_env, mock_post):
        tg.send_message("ticker_AAPL [BUY] *bold*")
        default_payload = mock_post.call_args.kwargs["json"]
        mock_post.reset_mock()
        tg.send_message("ticker_AAPL [BUY] *bold*", parse_mode="Markdown")
        explicit_payload = mock_post.call_args.kwargs["json"]
        assert explicit_payload == default_payload
        assert explicit_payload["parse_mode"] == "Markdown"
        assert explicit_payload["text"] == "ticker\\_AAPL \\[BUY\\] *bold*"

    def test_the_wire_body_is_byte_identical_including_key_order(self, configured_env, mock_post):
        # "Byte-identical" means the serialised JSON, not a dict comparison
        # (review A1): `requests` serialises in insertion order, so the order
        # the payload is BUILT in is the order on the wire.
        import json

        tg.send_message("hello world")
        payload = mock_post.call_args.kwargs["json"]
        assert list(payload) == ["chat_id", "text", "parse_mode", "disable_notification"]
        assert json.dumps(payload) == (
            '{"chat_id": "12345", "text": "hello world", "parse_mode": "Markdown", '
            '"disable_notification": false}'
        )

    def test_the_module_default_constant_is_still_markdown(self):
        assert tg.PARSE_MODE == "Markdown"


class TestParseModeValidation:
    def test_unknown_mode_raises_before_any_request(self, configured_env, mock_post):
        with pytest.raises(ValueError, match="parse_mode"):
            tg.send_message("x", parse_mode="markdown")
        mock_post.assert_not_called()

    def test_markdown_v2_is_not_supported(self, configured_env, mock_post):
        # Not implemented here: v2 needs a different escaper. Refusing is
        # honest; sending it would 400 and silently downgrade on the retry.
        with pytest.raises(ValueError):
            tg.send_message("x", parse_mode="MarkdownV2")


class TestEscapeHtml:
    def test_escapes_the_three_html_syntax_characters(self):
        assert tg.escape_html("a < b & c > d") == "a &lt; b &amp; c &gt; d"

    def test_ampersand_first_so_entities_are_not_double_escaped(self):
        assert tg.escape_html("&lt;") == "&amp;lt;"

    def test_leaves_quotes_and_markdown_characters_alone(self):
        assert tg.escape_html("it's *fine* _here_ [ok]") == "it's *fine* _here_ [ok]"

    def test_escaped_text_survives_an_html_send_verbatim(self, configured_env, mock_post):
        body = "<b>DETAIL</b> " + tg.escape_html("gates/<phase>/gate.json & more")
        tg.send_message(body, parse_mode="HTML")
        assert mock_post.call_args.kwargs["json"]["text"] == body


class TestHtmlTruncation:
    def test_within_limit_passes_through(self):
        assert tg._truncate_html("<b>x</b>") == "<b>x</b>"

    def test_result_fits_the_limit(self):
        text = "<pre>" + "z" * 6000 + "</pre>"
        out = tg._truncate_html(text)
        assert len(out) <= tg.TELEGRAM_MESSAGE_MAX_CHARS

    def test_an_open_tag_at_the_cut_is_closed_before_the_marker(self):
        text = "<pre>" + "z" * 6000 + "</pre>"
        out = tg._truncate_html(text)
        assert "</pre>" in out
        assert out.index("</pre>") < out.index("…(truncated")

    @pytest.mark.parametrize("head_len", range(4000, 4060))
    def test_a_partial_tag_at_the_cut_is_dropped(self, head_len):
        # Sweep the cut point across a `<code>` opener: wherever the budget
        # lands, no `<` is ever left without its `>` and the result balances.
        text = "a" * head_len + "<code>" + "b" * 3000 + "</code>"
        out = tg._truncate_html(text)
        assert out.count("<") == out.count(">")
        assert "<code>" not in out or "</code>" in out
        assert len(out) <= tg.TELEGRAM_MESSAGE_MAX_CHARS

    def test_nested_tags_close_innermost_first(self):
        text = "<b><i>" + "n" * 6000 + "</i></b>"
        out = tg._truncate_html(text)
        assert "</i></b>" in out

    def test_unknown_tags_are_not_treated_as_structure(self):
        # `<x>` is not a Telegram tag; it neither opens nor needs closing.
        text = "<x>" + "n" * 6000
        out = tg._truncate_html(text)
        assert "</x>" not in out

    def test_send_under_html_uses_tag_aware_truncation(self, configured_env, mock_post):
        tg.send_message("<pre>" + "z" * 6000 + "</pre>", parse_mode="HTML")
        payload = mock_post.call_args.kwargs["json"]
        assert payload["text"].endswith(
            f"</pre>\n…(truncated, showing {tg.TELEGRAM_MESSAGE_MAX_CHARS} of 6011 chars)"
        )
        assert len(payload["text"]) <= tg.TELEGRAM_MESSAGE_MAX_CHARS


class TestMarkdownEntityFallback:
    def test_unpaired_asterisk_message_is_redelivered_as_plain_text(
        self, configured_env, mock_post
    ):
        mock_post.side_effect = _responses((400, _ENTITY_400), (200, "ok"))
        assert tg.send_message(_REAL_DROPPED_MESSAGE) is True
        assert mock_post.call_count == 2

    def test_retry_drops_parse_mode_and_changes_nothing_else(
        self, configured_env, mock_post
    ):
        mock_post.side_effect = _responses((400, _ENTITY_400), (200, "ok"))
        tg.send_message(_REAL_DROPPED_MESSAGE, message_thread_id=77)
        first = mock_post.call_args_list[0].kwargs["json"]
        retry = mock_post.call_args_list[1].kwargs["json"]
        assert first["parse_mode"] == tg.PARSE_MODE
        assert "parse_mode" not in retry
        # Everything the operator needs must survive the degradation: only the
        # rendering is allowed to change, never the content or the routing.
        assert retry["text"] == first["text"]
        assert retry["chat_id"] == first["chat_id"]
        assert retry["message_thread_id"] == 77
        assert retry["disable_notification"] == first["disable_notification"]

    def test_non_entity_400_is_not_retried(self, configured_env, mock_post):
        # "chat not found" is not fixable by dropping parse_mode; retrying would
        # double the request and delay the failure log for no gain.
        mock_post.side_effect = _responses(
            (400, '{"description":"Bad Request: chat not found"}'), (200, "ok")
        )
        assert tg.send_message("hello") is False
        assert mock_post.call_count == 1

    def test_non_json_400_is_not_retried(self, configured_env, mock_post):
        mock_post.side_effect = _responses((400, "<html>gateway</html>"), (200, "ok"))
        assert tg.send_message("hello") is False
        assert mock_post.call_count == 1

    def test_plain_text_retry_that_also_fails_returns_false(
        self, configured_env, mock_post
    ):
        mock_post.side_effect = _responses(
            (400, _ENTITY_400), (500, '{"description":"Internal Server Error"}')
        )
        assert tg.send_message(_REAL_DROPPED_MESSAGE) is False
        assert mock_post.call_count == 2

    def test_retry_outcome_is_logged(self, configured_env, mock_post, caplog):
        mock_post.side_effect = _responses((400, _ENTITY_400), (200, "ok"))
        tg.send_message(_REAL_DROPPED_MESSAGE)
        assert "redelivered" in caplog.text

    def test_network_exception_is_not_treated_as_a_parse_failure(
        self, configured_env, mock_post
    ):
        # No description at all must not fall through into the retry branch.
        mock_post.side_effect = requests.Timeout("timed out")
        assert tg.send_message(_REAL_DROPPED_MESSAGE) is False
        assert mock_post.call_count == 1


class TestIsEntityParseError:
    def test_matches_the_live_telegram_description(self):
        assert tg._is_entity_parse_error(
            "Bad Request: can't parse entities: Can't find end of the entity "
            "starting at byte offset 355"
        )

    def test_is_case_insensitive(self):
        assert tg._is_entity_parse_error("BAD REQUEST: CAN'T PARSE ENTITIES: x")

    def test_does_not_match_unrelated_400s(self):
        for other in (
            "Bad Request: chat not found",
            "Bad Request: message is too long",
            "Forbidden: bot was blocked by the user",
            "<non-JSON body suppressed>",
            "",
        ):
            assert not tg._is_entity_parse_error(other)


# ── Destination overrides (alpha-engine-config-I7857) ───────────────────────
# `krepis.alerts` routes a non-incident finding to a SECOND chat by handing
# this transport a `chat_id` (and optionally a forum-topic
# `message_thread_id`). These pin the payload the transport actually builds,
# because the routing tier above is only as real as the field it sets here.


class TestDestinationOverrides:
    def test_default_uses_the_operator_chat_from_secrets(
        self, configured_env, mock_post
    ):
        tg.send_message("m")
        assert mock_post.call_args.kwargs["json"]["chat_id"] == "12345"

    def test_explicit_chat_id_replaces_the_secret(self, configured_env, mock_post):
        tg.send_message("m", chat_id="-1001234")
        payload = mock_post.call_args.kwargs["json"]
        assert payload["chat_id"] == "-1001234"

    def test_chat_id_none_is_not_an_override(self, configured_env, mock_post):
        # `alerts` passes `chat_id=None` for the operator destination; that
        # must resolve the secret, never send to a chat named "None".
        tg.send_message("m", chat_id=None)
        assert mock_post.call_args.kwargs["json"]["chat_id"] == "12345"

    def test_message_thread_id_is_sent_only_when_given(
        self, configured_env, mock_post
    ):
        tg.send_message("m", chat_id="-1001234")
        assert "message_thread_id" not in mock_post.call_args.kwargs["json"]
        tg.send_message("m", chat_id="-1001234", message_thread_id=42)
        assert mock_post.call_args.kwargs["json"]["message_thread_id"] == 42

    def test_override_does_not_disturb_the_buzz_flag(self, configured_env, mock_post):
        tg.send_message("m", chat_id="-1001234", disable_notification=True)
        assert mock_post.call_args.kwargs["json"]["disable_notification"] is True
        tg.send_message("m", chat_id="-1001234", disable_notification=False)
        assert mock_post.call_args.kwargs["json"]["disable_notification"] is False

    def test_override_survives_the_plain_text_retry(self, configured_env, mock_post):
        # A Markdown parse failure retries the identical body without
        # `parse_mode`; the destination must not silently revert to the
        # operator chat on the retry.
        mock_post.side_effect = [
            MagicMock(
                status_code=400,
                text='{"description": "Bad Request: can\'t parse entities"}',
            ),
            MagicMock(status_code=200, text="ok"),
        ]
        assert tg.send_message("m *x", chat_id="-1001234", message_thread_id=7) is True
        retry_payload = mock_post.call_args.kwargs["json"]
        assert retry_payload["chat_id"] == "-1001234"
        assert retry_payload["message_thread_id"] == 7
        assert "parse_mode" not in retry_payload
