"""
Telegram push-notification client for Alpha Engine modules.

Consolidation substrate for Telegram sends across consumer repos. Before this
module, ``alpha-engine/executor/notifier.py`` was the only Telegram producer
and duplicated token/chat_id resolution, markdown escaping, and the
fire-and-forget request shape inline. With the executor surveillance Lambda
arc (ROADMAP L1067, 2026-05-13), a second producer (``alpha-engine-research``)
needs the same send path — consolidating here prevents the
"two writers diverged silently" antipattern.

**Public API:**

- :func:`send_message` — primitive single-message send. Returns ``bool``,
  never raises. Misconfigured secrets resolve to a logged warning + ``False``,
  not an exception, so caller code can be fire-and-forget at every site.
- :func:`send_rollup` — convenience wrapper that joins a list of findings
  into a single bulleted message, defaulting to ``disable_notification=True``
  (in-channel surveillance digest without push buzz).

**Severity tiering via ``disable_notification``.** Telegram's
``disable_notification`` flag delivers the message into the chat silently —
visible in-channel but no phone-buzz notification. Use this to send a single
channel both loud (critical alerts: daemon-down, position drawdown) and
silent (surveillance digests: untouched buy-candidates). Critical alerts:
``send_message(text)`` (defaults to push). Informational digests:
``send_rollup(findings)`` (defaults to silent).

**Secret resolution.** Both ``TELEGRAM_BOT_TOKEN`` and ``TELEGRAM_CHAT_ID``
are loaded via :func:`krepis.secrets.get_secret` with
``required=False``. If either is absent, the call logs a warning and returns
``False`` — matches the legacy ``notifier.py`` behavior so callers can be
configured-or-no-op without conditional branching.

**Failure behavior.** Network errors, HTTP non-200 responses, and timeouts
are logged at WARNING and returned as ``False``. No exceptions propagate.
This is by design — a failed Telegram notification must never block trade
execution or surveillance Lambda completion.

**Migration arc**: ``alpha-engine-config/private-docs/ROADMAP.md`` L1067
("Intraday data store → executor surveillance Lambda"), PR 1 of the 3-PR
sequence.
"""

from __future__ import annotations

import logging
from typing import Final

import requests

from krepis.secrets import get_secret

logger = logging.getLogger(__name__)

TELEGRAM_API_URL: Final[str] = "https://api.telegram.org/bot{token}/sendMessage"
TELEGRAM_TIMEOUT_SEC: Final[int] = 5
PARSE_MODE: Final[str] = "Markdown"


def _escape_markdown(text: str) -> str:
    """Escape Telegram Markdown v1 special characters.

    Replaces characters that Telegram interprets as formatting markers
    (``_``, `````, ``[``, ``]``) to prevent 400 Bad Request parse errors.
    Preserves ``*`` for bold markers which callers control via message
    templates.
    """
    return (
        text
        .replace("_", "-")
        .replace("`", "'")
        .replace("[", "(")
        .replace("]", ")")
    )


def send_message(text: str, *, disable_notification: bool = False) -> bool:
    """Send a single Telegram message to the channel resolved from secrets.

    Loads ``TELEGRAM_BOT_TOKEN`` + ``TELEGRAM_CHAT_ID`` via
    :func:`krepis.secrets.get_secret` (required=False). Applies
    Markdown v1 escaping, ``POST``s with a 5-second timeout. Returns ``True``
    on HTTP 200, ``False`` on any other outcome (logged at WARNING). Never
    raises.

    :param text: The message body. Markdown v1 formatting (``*bold*``) is
        respected; other special characters are escaped automatically.
    :param disable_notification: If ``True``, the message is delivered into
        the chat silently (no phone push). Use for informational/digest
        traffic that should be visible but not buzz.
    :returns: ``True`` if the Telegram API returned HTTP 200, ``False``
        otherwise (missing secrets, network error, non-200 response).
    """
    token = get_secret("TELEGRAM_BOT_TOKEN", required=False)
    chat_id = get_secret("TELEGRAM_CHAT_ID", required=False)
    if not token or not chat_id:
        logger.warning(
            "Telegram not configured — TELEGRAM_BOT_TOKEN=%s TELEGRAM_CHAT_ID=%s",
            "set" if token else "MISSING",
            "set" if chat_id else "MISSING",
        )
        return False

    payload = {
        "chat_id": chat_id,
        "text": _escape_markdown(text),
        "parse_mode": PARSE_MODE,
        "disable_notification": disable_notification,
    }

    try:
        resp = requests.post(
            TELEGRAM_API_URL.format(token=token),
            json=payload,
            timeout=TELEGRAM_TIMEOUT_SEC,
        )
    except requests.RequestException:
        logger.warning("Telegram send failed (request exception)", exc_info=True)
        return False

    if resp.status_code == 200:
        return True
    logger.warning(
        "Telegram API returned %d: %s",
        resp.status_code,
        resp.text[:200] if resp.text else "",
    )
    return False


def send_rollup(
    findings: list[str],
    *,
    header: str | None = None,
    disable_notification: bool = True,
) -> bool:
    """Send a bulleted rollup of N findings as a single message.

    Convenience wrapper for surveillance digest traffic — a list of findings
    becomes a single message with each finding rendered as a ``-``-prefixed
    bullet. Defaults to ``disable_notification=True`` (silent in-channel) so
    digests don't buzz the phone; pass ``False`` to override for high-severity
    rollups.

    Empty ``findings`` is a no-op that returns ``True`` without an API call —
    callers can pass output of a filter directly without an emptiness check.

    :param findings: List of finding strings (one per bullet).
    :param header: Optional bold header rendered above the bullets.
    :param disable_notification: Default ``True`` (silent). Pass ``False`` to
        push.
    :returns: ``True`` if no findings (no-op) or Telegram returned 200,
        ``False`` on send failure.
    """
    if not findings:
        return True

    lines = []
    if header:
        lines.append(f"*{header}*")
    lines.extend(f"- {item}" for item in findings)
    text = "\n".join(lines)

    return send_message(text, disable_notification=disable_notification)
