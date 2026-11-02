"""DLP on outbound alert text (alpha-engine-config-I13042).

On 2026-10-02 a live Polygon key reached a Telegram alert inside an exception
string (``...?apiKey=<key>``; nous-ergon-ops-I2002). The point fix stopped one
client from putting the key in its errors; this module is the check that any
credential-shaped text handed to an outbound alert sender is gone by the time
it reaches the transport.

Two halves, and both are needed:

1. **Behaviour.** Every sender in :data:`OUTBOUND_SENDERS` is driven with a
   body carrying one canary of each credential class, the transport is
   replaced by a capturing double, and the bytes that would have left the
   process are asserted to carry no canary and to match no
   ``_SECRET_REDACTION_PATTERNS`` entry.
2. **Inventory.** Every ``src/krepis`` module that reaches an outbound alert
   transport (Telegram Bot API, SNS, SES, SMTP) must be listed in
   :data:`OUTBOUND_SENDERS`. A new sender module fails here until it routes
   through an existing sender or applies ``redact_secrets`` and gains a
   behaviour test above. Without this half the behaviour test only proves the
   senders somebody remembered to list.
"""

from __future__ import annotations

import email
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from krepis import alerts, email_sender
from krepis import telegram as tg
from krepis.logging import _SECRET_REDACTION_PATTERNS
from krepis.secrets import clear_cache

# Fake but format-valid credentials, one per redaction class. None is real.
_POLYGON_KEY = "PgFAKEk3yNotReal0000xyzABC"
CANARIES: dict[str, str] = {
    "url_query_camel": _POLYGON_KEY,
    "aws_access_key_id": "AKIAFAKEFAKEFAKE1234",
    "anthropic": "sk-ant-api03-FAKEfakeFAKEfake0000",
    "github_pat": "ghp_FAKEfakeFAKEfakeFAKEfake00000000",
    "bearer": "eyFAKEbearerTOKEN.value",
}
ALERT_BODY = (
    "Dividend window failed: 429 for url: "
    f"https://api.polygon.io/v3/reference/dividends?cursor=1&apiKey={_POLYGON_KEY}&limit=50\n"
    f"aws key {CANARIES['aws_access_key_id']} anthropic {CANARIES['anthropic']}\n"
    f"gh {CANARIES['github_pat']}\n"
    f"Authorization: Bearer {CANARIES['bearer']}"
)

# Module (relative to src/krepis) -> the behaviour test below that covers it.
OUTBOUND_SENDERS: dict[str, str] = {
    "telegram.py": "test_telegram_send_message_redacts_on_the_wire",
    "alerts.py": "test_alerts_sns_leg_redacts_on_the_wire",
    "email_sender.py": "test_email_gmail_leg_redacts_on_the_wire / test_email_ses_leg_redacts_on_the_wire",
}

# What "reaches an outbound alert transport" means, as source text.
_TRANSPORT_MARKERS = re.compile(
    r"api\.telegram\.org"
    r"|client\(\s*['\"](?:sns|ses|sesv2)['\"]"
    r"|smtplib\.SMTP"
)

_SRC = Path(__file__).resolve().parents[1] / "src" / "krepis"


def _assert_clean(payload: str) -> None:
    leaked = [name for name, value in CANARIES.items() if value in payload]
    assert not leaked, f"credential canaries reached the transport: {leaked}\n{payload}"
    hits = [p.pattern for p, _ in _SECRET_REDACTION_PATTERNS if p.search(payload)]
    assert not hits, f"outbound payload still matches redaction patterns {hits}\n{payload}"


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("ALPHA_ENGINE_SECRETS_SOURCE", "env")
    monkeypatch.delenv("ALPHA_ENGINE_DISABLE_LOG_REDACTION", raising=False)
    clear_cache()
    yield
    clear_cache()


def test_canary_body_is_dirty_before_redaction() -> None:
    """Guards the test itself: every canary must be one the patterns catch,
    or the clean-payload assertions below pass vacuously."""
    for name, value in CANARIES.items():
        assert value in ALERT_BODY, name
    with pytest.raises(AssertionError):
        _assert_clean(ALERT_BODY)


def test_telegram_send_message_redacts_on_the_wire(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token-abc123")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")
    with patch.object(tg.requests, "post") as post:
        post.return_value.status_code = 200
        post.return_value.json.return_value = {"ok": True, "result": {"message_id": 1}}
        tg.send_message(ALERT_BODY, parse_mode=None)
    assert post.called
    _assert_clean(post.call_args.kwargs["json"]["text"])


def test_alerts_sns_leg_redacts_on_the_wire() -> None:
    client = MagicMock()
    client.publish.return_value = {"MessageId": "m-1"}
    with patch("boto3.client", return_value=client):
        result = alerts._publish_sns(
            "arn:aws:sns:us-east-1:000000000000:alpha-engine-alerts",
            ALERT_BODY,
            subject=f"failed: ?apiKey={_POLYGON_KEY}",
        )
    assert result.ok
    sent: dict[str, Any] = client.publish.call_args.kwargs
    _assert_clean(sent["Message"])
    _assert_clean(sent["Subject"])


def _decoded_parts(raw: str) -> str:
    msg = email.message_from_string(raw)
    out = [str(msg["Subject"])]
    for part in msg.walk():
        payload = part.get_payload(decode=True)
        if isinstance(payload, bytes):
            out.append(payload.decode("utf-8"))
    return "\n".join(out)


def test_email_gmail_leg_redacts_on_the_wire(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "fake-app-password")
    server = MagicMock()
    smtp = MagicMock()
    smtp.return_value.__enter__.return_value = server
    with patch.object(email_sender.smtplib, "SMTP", smtp):
        ok = email_sender.send_email(
            f"failed: ?apiKey={_POLYGON_KEY}",
            ALERT_BODY,
            recipients=["ops@example.invalid"],
            sender="alerts@example.invalid",
            html=f"<pre>{ALERT_BODY}</pre>",
        )
    assert ok
    raw = server.sendmail.call_args.args[2]
    _assert_clean(_decoded_parts(raw))


def test_email_ses_leg_redacts_on_the_wire(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GMAIL_APP_PASSWORD", raising=False)
    ses = MagicMock()
    with patch("boto3.client", return_value=ses):
        ok = email_sender.send_email(
            f"failed: ?apiKey={_POLYGON_KEY}",
            ALERT_BODY,
            recipients=["ops@example.invalid"],
            sender="alerts@example.invalid",
            region="us-east-1",
            html=f"<pre>{ALERT_BODY}</pre>",
        )
    assert ok
    message = ses.send_email.call_args.kwargs["Message"]
    _assert_clean(message["Subject"]["Data"])
    _assert_clean(message["Body"]["Text"]["Data"])
    _assert_clean(message["Body"]["Html"]["Data"])


def test_every_outbound_alert_transport_is_covered() -> None:
    found = sorted(
        str(path.relative_to(_SRC))
        for path in _SRC.rglob("*.py")
        if _TRANSPORT_MARKERS.search(path.read_text(encoding="utf-8"))
    )
    uncovered = [mod for mod in found if mod not in OUTBOUND_SENDERS]
    assert not uncovered, (
        "these krepis modules reach an outbound alert transport but have no "
        f"DLP egress test: {uncovered}. Route them through telegram.send_message "
        "/ alerts.publish / email_sender.send_email, or apply "
        "krepis.logging.redact_secrets before the send and add a behaviour "
        "test here (alpha-engine-config-I13042)."
    )
    stale = [mod for mod in OUTBOUND_SENDERS if mod not in found]
    assert not stale, f"OUTBOUND_SENDERS lists modules with no transport any more: {stale}"
