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

**Message length.** ``text`` over ``TELEGRAM_MESSAGE_MAX_CHARS`` (4096, the
Bot API's hard limit) is truncated rather than sent as-is and rejected —
config-I3301: an oversized message used to fail Telegram silently (HTTP 400)
while a parallel SNS publish in the same ``krepis.alerts.publish`` call
succeeded, masking the failure entirely from the caller.

**Migration arc**: ``alpha-engine-config/private-docs/ROADMAP.md`` L1067
("Intraday data store → executor surveillance Lambda"), PR 1 of the 3-PR
sequence.
"""

from __future__ import annotations

import json
import logging
from typing import Final

import requests

from krepis import fleet_events
from krepis.secrets import get_secret

logger = logging.getLogger(__name__)

TELEGRAM_API_URL: Final[str] = "https://api.telegram.org/bot{token}/sendMessage"
TELEGRAM_TIMEOUT_SEC: Final[int] = 5
PARSE_MODE: Final[str] = "Markdown"
# Telegram's hard per-message limit (Bot API `sendMessage.text`). A message
# over this returns HTTP 400 with no partial delivery — and until
# config-I3301, that failure was silent from the caller's perspective
# whenever a parallel channel (e.g. `krepis.alerts.publish`'s SNS leg)
# succeeded, masking it. First observed live 2026-07-22:
# `alpha-engine-dashboard/infrastructure/alert_on_failure.sh` builds its
# message from up to 30 raw `journalctl` lines with no length guard.
TELEGRAM_MESSAGE_MAX_CHARS: Final[int] = 4096
# Telegram's 400 `description` when the Markdown parser reaches the end of the
# body with an entity still open. See `_is_entity_parse_error` for why this is
# matched on text rather than a status code.
_ENTITY_PARSE_ERROR_MARKER: Final[str] = "can't parse entities"


def _is_entity_parse_error(description: str) -> bool:
    """True when a Telegram 400 was caused by unparseable Markdown entities.

    Telegram returns HTTP 400 for many unrelated conditions (chat not found,
    bot blocked, message too long), and the status code alone cannot tell them
    apart — only the ``description`` string can. Matching on it is therefore
    not a shortcut; it is the only signal the API provides.

    Deliberately narrow: a formatting failure is the one 400 that a plain-text
    retry can fix. Retrying the others would send the same doomed request
    twice and delay the failure log.
    """
    return _ENTITY_PARSE_ERROR_MARKER in description.lower()


def _truncate_for_telegram(text: str) -> str:
    """Truncate ``text`` to fit Telegram's ``TELEGRAM_MESSAGE_MAX_CHARS``
    hard limit, appending a marker noting how much was cut. No-op when
    already within the limit.

    Keeps the HEAD of the message and truncates the TAIL: callers
    front-load the identifying summary (severity/source/what-failed) per
    ``krepis.alerts._format_message``'s convention and append supplementary
    detail (journal excerpts, stack traces, findings lists) after — so
    trimming the tail preserves the part an operator needs to triage at a
    glance, at the cost of the least-essential detail.
    """
    if len(text) <= TELEGRAM_MESSAGE_MAX_CHARS:
        return text
    suffix = f"\n…(truncated, showing {TELEGRAM_MESSAGE_MAX_CHARS} of {len(text)} chars)"
    keep = TELEGRAM_MESSAGE_MAX_CHARS - len(suffix)
    return text[:keep] + suffix


def _escape_markdown(text: str) -> str:
    """Escape Telegram Markdown v1 special characters.

    Replaces characters that Telegram interprets as formatting markers
    (``_``, `````, ``[``, ``]``) to prevent 400 Bad Request parse errors.
    Preserves ``*`` for bold markers which callers control via message
    templates.

    **``*`` is knowingly left unescaped, and that is not sufficient on its
    own.** The premise — that callers control every asterisk — holds for a
    literal template and fails the moment arbitrary text is interpolated into
    one. Live instance (2026-08-01, `alpha-engine-config-I5995` arc): the
    Fleet-SF Watch receipt embeds a Step Functions failure cause, that cause
    embedded git's ``* branch main -> FETCH_HEAD`` line, and the resulting odd
    asterisk count left a bold entity open. Telegram returned
    ``can't parse entities`` at byte offset 355 — the unpaired ``*`` — and the
    receipt was dropped. Six such drops in 21 days across two dates, each one
    an alert the operator never saw.

    Escaping ``*`` here would fix that and break every intentional ``*bold*``
    in the fleet, so the residual is handled at the transport instead:
    :func:`send_message` retries once as plain text on this specific 400.
    Formatting is what degrades; delivery is not.
    """
    return (
        text
        .replace("_", "-")
        .replace("`", "'")
        .replace("[", "(")
        .replace("]", ")")
    )


def send_message(
    text: str,
    *,
    disable_notification: bool = False,
    bot_token: str | None = None,
    chat_id: str | int | None = None,
    message_thread_id: int | None = None,
) -> bool:
    """Send a single Telegram message to the channel resolved from secrets.

    Loads ``TELEGRAM_BOT_TOKEN`` + ``TELEGRAM_CHAT_ID`` via
    :func:`krepis.secrets.get_secret` (required=False) when ``bot_token`` /
    ``chat_id`` are not passed explicitly. Truncates ``text`` to
    ``TELEGRAM_MESSAGE_MAX_CHARS`` (config-I3301 — see
    :func:`_truncate_for_telegram`), applies Markdown v1 escaping, ``POST``s
    with a 5-second timeout. Returns ``True`` on HTTP 200, ``False`` on any
    other outcome (logged at WARNING). Never raises.

    Explicit ``bot_token`` / ``chat_id`` overrides allow flow-doctor (and other
    multi-bot consumers) to route through this transport without clobbering the
    process-global secret resolution path.

    :param text: The message body. Markdown v1 formatting (``*bold*``) is
        respected; other special characters are escaped automatically. Bodies
        over ``TELEGRAM_MESSAGE_MAX_CHARS`` are truncated (tail-trimmed) with
        a marker noting how much was cut, rather than failing the send
        outright.
    :param disable_notification: If ``True``, the message is delivered into
        the chat silently (no phone push). Use for informational/digest
        traffic that should be visible but not buzz.
    :param bot_token: Optional explicit bot token (skips secret lookup).
    :param chat_id: Optional explicit chat id (skips secret lookup).
    :param message_thread_id: Optional forum-topic id for supergroup routing.
    :returns: ``True`` if the Telegram API returned HTTP 200, ``False``
        otherwise (missing secrets, network error, non-200 response). A 400
        caused by unparseable Markdown is retried once as plain text before
        ``False`` is returned — see the fallback comment in the body.
    """
    token = bot_token or get_secret("TELEGRAM_BOT_TOKEN", required=False)
    resolved_chat = chat_id if chat_id is not None else get_secret("TELEGRAM_CHAT_ID", required=False)
    if not token or resolved_chat in (None, ""):
        logger.warning(
            "Telegram not configured — TELEGRAM_BOT_TOKEN=%s TELEGRAM_CHAT_ID=%s",
            "set" if token else "MISSING",
            "set" if resolved_chat not in (None, "") else "MISSING",
        )
        return False

    payload = {
        "chat_id": resolved_chat,
        "text": _escape_markdown(_truncate_for_telegram(text)),
        "parse_mode": PARSE_MODE,
        "disable_notification": disable_notification,
    }
    if message_thread_id is not None:
        payload["message_thread_id"] = message_thread_id

    def _post(body: dict) -> "tuple[bool, str | None]":
        """POST ``body``; return ``(delivered, description_on_failure)``.

        ``description`` is ``None`` when the request never produced a parseable
        Telegram error — a network exception or a non-JSON body — so a caller
        cannot mistake "no description" for "an empty description".
        """
        try:
            resp = requests.post(
                TELEGRAM_API_URL.format(token=token),
                json=body,
                timeout=TELEGRAM_TIMEOUT_SEC,
            )
        except requests.RequestException:
            logger.warning("Telegram send failed (request exception)", exc_info=True)
            return False, None
        if resp.status_code == 200:
            return True, None
        # Log only the parsed Telegram `description` field, never the raw
        # body: the request URL embeds the bot token, and a non-Telegram
        # error page (proxy 502, HTML 404) can echo the full URL — logging
        # raw resp.text would leak the token in clear text. Telegram's own
        # JSON error bodies never contain the token, so `description` is
        # safe and carries the operationally useful part.
        try:
            detail = str(json.loads(resp.text).get("description", ""))[:200]
        except Exception:
            detail = "<non-JSON body suppressed>"
        # Defense in depth: even a hostile/MITM JSON body that echoes the
        # request URL cannot leak the token past this replace.
        detail = detail.replace(token, "[REDACTED]")
        logger.warning("Telegram API returned %d: %s", resp.status_code, detail)
        return False, detail

    ok, description = _post(payload)

    # ── Plain-text fallback (alpha-engine-config-I5995 arc) ───────────────
    # A Markdown parse failure is the one 400 whose cause is the FORMATTING
    # rather than the message, the recipient or the bot — so it is the one
    # that a retry can fix rather than merely repeat.
    #
    # `_escape_markdown` neutralises every v1 delimiter except `*`, which it
    # must preserve for intentional bold. Any caller interpolating arbitrary
    # text — a Step Functions cause, a traceback, journalctl output — can
    # therefore emit an odd asterisk count and lose the whole message. That is
    # not a hypothetical: six Fleet-SF Watch dispatch receipts were dropped
    # this way over 21 days, and because the failure is a logged WARNING on a
    # best-effort path, nothing downstream noticed.
    #
    # Retrying WITHOUT `parse_mode` sends the identical body as plain text.
    # The operator loses bold; they do not lose the alert. Degrading the
    # presentation to preserve the delivery is the correct direction for an
    # alerting transport — the inverse, which is what shipped, silently
    # converts a formatting defect into a missed incident.
    if not ok and description is not None and _is_entity_parse_error(description):
        retry_payload = {k: v for k, v in payload.items() if k != "parse_mode"}
        ok, _ = _post(retry_payload)
        logger.warning(
            "Telegram Markdown parse failed; %s as plain text "
            "(krepis._escape_markdown preserves '*', which arbitrary "
            "interpolated text can leave unpaired)",
            "redelivered" if ok else "plain-text retry ALSO failed",
        )

    # ── Overseer intake event (side-channel; best-effort, never raises) ──
    # Direct sends (Lambdas, flow-doctor notifiers) get structured intake
    # coverage here with zero caller changes; alerts.publish suppresses
    # this hook and emits its own richer event. The not-configured early
    # return above deliberately does NOT emit — no Telegram config means a
    # non-production context. Severity is proxied from the silent flag.
    if not fleet_events.emission_suppressed():
        fleet_events.emit_alert_event(
            origin="telegram.send_message",
            body=text,
            severity_raw=None,
            dedup_key=None,
            channels={"sns": None, "telegram": ok},
            disable_notification=disable_notification,
        )

    return ok


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
