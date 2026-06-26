"""
Email-send client for Alpha Engine modules.

Consolidation substrate for transactional email across consumer repos.
Before this module, every repo independently implemented the
"Gmail SMTP primary + AWS SES fallback" send path against the same
``EMAIL_SENDER`` / ``EMAIL_RECIPIENTS`` / ``GMAIL_APP_PASSWORD`` /
``AWS_REGION`` env-var convention (``~/Development/CLAUDE.md`` "Email"):
``alpha-engine/executor/eod_emailer.py``, ``alpha-engine-research``'s
sender, plus the predictor morning briefing, backtester evaluator email,
and data-collector failure alerts. Email is the higher-cardinality
producer (>=4 modules), so the "two writers diverged silently"
antipattern — drift on retry semantics, MIME shape, fallback ordering —
carries more risk here than it did for the Telegram consolidation
(``krepis.telegram``, lib v0.14.0) that surfaced this gap.

**Public API:**

- :func:`send_email` — primitive single-message send. Returns ``bool``,
  never raises. Missing/misconfigured secrets resolve to a logged warning
  + ``False``, not an exception, so every caller can be fire-and-forget.

**Transport.** Gmail SMTP (``smtp.gmail.com:587`` STARTTLS) is the primary
path when ``GMAIL_APP_PASSWORD`` is set — mail originates from Gmail's
servers and passes SPF/DKIM. When the app password is absent the send
falls back to AWS SES in ``AWS_REGION``. (SES delivers reliably only with
a verified custom-domain sender; an ``@gmail.com`` SES sender may be
silently dropped — the legacy per-repo behavior, preserved here.)

**Secret resolution.** ``EMAIL_SENDER``, ``EMAIL_RECIPIENTS``,
``GMAIL_APP_PASSWORD`` and ``AWS_REGION`` are loaded via
:func:`krepis.secrets.get_secret` with ``required=False``.
Explicit ``sender`` / ``recipients`` / ``region`` arguments override the
resolved secrets. If no sender or no recipients can be determined the call
logs a warning and returns ``False`` — configured-or-no-op without
conditional branching at the call site, matching
:func:`krepis.telegram.send_message`.

**Failure behavior.** SMTP auth failures, network errors, SES
``ClientError``, and any other exception are logged at ERROR and returned
as ``False``. No exceptions propagate. By design — a failed notification
must never block trade execution, EOD reconcile, or a Saturday-SF stage.

**Migration arc**: ``alpha-engine-config/private-docs/ROADMAP.md`` L3204
("Consolidate ``email_sender`` into ``krepis``"), PR 1 of the
~5-PR sequence (this PR = lib substrate + tests + version bump; PRs 2-N
migrate each consumer with a lockstep requirements pin bump).
"""

from __future__ import annotations

import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Final

from krepis.secrets import get_secret

logger = logging.getLogger(__name__)

GMAIL_SMTP_HOST: Final[str] = "smtp.gmail.com"
GMAIL_SMTP_PORT: Final[int] = 587
SMTP_TIMEOUT_SEC: Final[int] = 30
DEFAULT_AWS_REGION: Final[str] = "us-east-1"


def _resolve_recipients(recipients: list[str] | None) -> list[str]:
    """Return the recipient list, preferring the explicit argument.

    Falls back to the comma-separated ``EMAIL_RECIPIENTS`` secret. Blank
    entries are stripped so a trailing comma in the env value doesn't
    produce an empty-string recipient.
    """
    if recipients:
        return [r.strip() for r in recipients if r and r.strip()]
    raw = get_secret("EMAIL_RECIPIENTS", required=False, default="") or ""
    return [r.strip() for r in raw.split(",") if r.strip()]


def _send_via_gmail(
    *, sender: str, recipients: list[str], subject: str,
    plain_body: str, html: str | None, app_password: str,
) -> bool:
    """Send through Gmail SMTP with STARTTLS. Returns success bool."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg.attach(MIMEText(plain_body, "plain", "utf-8"))
    if html:
        msg.attach(MIMEText(html, "html", "utf-8"))
    try:
        with smtplib.SMTP(
            GMAIL_SMTP_HOST, GMAIL_SMTP_PORT, timeout=SMTP_TIMEOUT_SEC
        ) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(sender, app_password)
            server.sendmail(sender, recipients, msg.as_string())
        logger.info("Email sent via Gmail SMTP: %r -> %s", subject, recipients)
        return True
    except smtplib.SMTPAuthenticationError as e:
        logger.error(
            "Gmail SMTP auth failed: %s. Check GMAIL_APP_PASSWORD and 2FA.", e
        )
        return False
    except Exception as e:
        logger.error("Gmail SMTP send error: %s", e)
        return False


def _send_via_ses(
    *, sender: str, recipients: list[str], subject: str,
    plain_body: str, html: str | None, region: str,
) -> bool:
    """Send through AWS SES. Returns success bool."""
    logger.warning(
        "GMAIL_APP_PASSWORD not set — falling back to SES. "
        "If sender is @gmail.com, email may be silently dropped."
    )
    try:
        import boto3
        from botocore.exceptions import ClientError

        ses = boto3.client("ses", region_name=region)
        message: dict = {
            "Subject": {"Data": subject, "Charset": "UTF-8"},
            "Body": {"Text": {"Data": plain_body, "Charset": "UTF-8"}},
        }
        if html:
            message["Body"]["Html"] = {"Data": html, "Charset": "UTF-8"}
        ses.send_email(
            Source=sender,
            Destination={"ToAddresses": recipients},
            Message=message,
        )
        logger.info("Email sent via SES: %r -> %s", subject, recipients)
        return True
    except ClientError as e:
        logger.error("SES send failed: %s", e.response["Error"]["Message"])
        return False
    except Exception as e:
        logger.error("SES email error: %s", e)
        return False


def send_email(
    subject: str,
    body: str,
    *,
    recipients: list[str] | None = None,
    html: str | None = None,
    sender: str | None = None,
    region: str | None = None,
) -> bool:
    """Send a single email via Gmail SMTP (primary) or AWS SES (fallback).

    Resolves ``EMAIL_SENDER`` / ``EMAIL_RECIPIENTS`` / ``GMAIL_APP_PASSWORD``
    / ``AWS_REGION`` via :func:`krepis.secrets.get_secret`
    (``required=False``); explicit arguments override the resolved secrets.
    Gmail SMTP is used when an app password is available, otherwise SES.
    Returns ``True`` only on a confirmed successful send; ``False`` on
    missing config, auth failure, network error, or any other outcome
    (logged). **Never raises** — callers are fire-and-forget.

    :param subject: Email subject line.
    :param body: Plain-text body (always sent as the ``text/plain`` part).
    :param recipients: Explicit recipient list. Overrides
        ``EMAIL_RECIPIENTS`` when truthy.
    :param html: Optional HTML body. When provided the message is
        ``multipart/alternative`` (plain + html); otherwise plain only.
    :param sender: Explicit From address. Overrides ``EMAIL_SENDER``.
    :param region: Explicit AWS region for the SES fallback. Overrides
        ``AWS_REGION`` (default ``us-east-1``).
    :returns: ``True`` if the email was sent, ``False`` otherwise.
    """
    sender = sender or get_secret("EMAIL_SENDER", required=False, default="") or ""
    to = _resolve_recipients(recipients)
    if not sender or not to:
        logger.warning(
            "Email not configured — EMAIL_SENDER=%s recipients=%s",
            "set" if sender else "MISSING",
            "set" if to else "MISSING",
        )
        return False

    app_password = (
        get_secret("GMAIL_APP_PASSWORD", required=False, default="") or ""
    ).replace(" ", "")

    if app_password:
        return _send_via_gmail(
            sender=sender, recipients=to, subject=subject,
            plain_body=body, html=html, app_password=app_password,
        )
    region = (
        region
        or get_secret("AWS_REGION", required=False, default="")
        or DEFAULT_AWS_REGION
    )
    return _send_via_ses(
        sender=sender, recipients=to, subject=subject,
        plain_body=body, html=html, region=region,
    )
