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
#: The parse modes a caller may ask for (alpha-engine-config-I9925). ``None``
#: is also accepted and means "send with NO ``parse_mode`` key" — plain text —
#: which is distinct from the default: the payload omits the key rather than
#: sending ``null``, because Telegram rejects an explicit null.
PARSE_MODE_HTML: Final[str] = "HTML"
PARSE_MODES: Final[tuple] = (PARSE_MODE, PARSE_MODE_HTML)
# Telegram's 400 `description` when the entity parser reaches the end of the
# body with an entity still open, or meets a tag it does not know. The same
# marker opens BOTH the Markdown-v1 and the HTML variants (measured against the
# Bot API: `can't parse entities: Can't find end of the entity starting at
# byte offset 355` for v1, `can't parse entities: Unsupported start tag "x" at
# byte offset 3` and `... Unclosed start tag at byte offset 12` for HTML), so
# one predicate covers every mode this module can send. See
# `_is_entity_parse_error` for why this is matched on text rather than a
# status code.
_ENTITY_PARSE_ERROR_MARKER: Final[str] = "can't parse entities"
#: Tags Telegram's HTML mode accepts. Anything else is an "Unsupported start
#: tag" 400 and, on our side, a truncation boundary that must not be crossed.
_HTML_TAGS: Final[frozenset] = frozenset({
    "b", "strong", "i", "em", "u", "ins", "s", "strike", "del", "span",
    "tg-spoiler", "tg-emoji", "a", "code", "pre", "blockquote",
})


def _is_entity_parse_error(description: str) -> bool:
    """True when a Telegram 400 was caused by unparseable entities, in any mode.

    Telegram returns HTTP 400 for many unrelated conditions (chat not found,
    bot blocked, message too long), and the status code alone cannot tell them
    apart — only the ``description`` string can. Matching on it is therefore
    not a shortcut; it is the only signal the API provides.

    Deliberately narrow: a formatting failure is the one 400 that a plain-text
    retry can fix. Retrying the others would send the same doomed request
    twice and delay the failure log. Markdown v1 and HTML share the marker (see
    the constant), so a caller switching modes does not switch off the retry.
    """
    return _ENTITY_PARSE_ERROR_MARKER in description.lower()


def escape_html(text: str) -> str:
    """Escape ``& < >`` so ``text`` renders literally under ``parse_mode="HTML"``.

    Under HTML mode the CALLER owns the markup — a heading is ``<b>…</b>`` in
    the text it passes — so :func:`send_message` escapes nothing for that mode
    (escaping the whole body would destroy the caller's own tags). Every piece
    of interpolated content therefore goes through this first. It is public
    because it is the caller's job; keeping it private would make every caller
    write its own, which is how three escapers diverge.

    Only the three characters Telegram's HTML parser treats as syntax are
    escaped. Quotes are not: they are only special inside an attribute value,
    which this helper is not for (a caller building ``<a href="…">`` escapes
    the URL as an attribute itself).
    """
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _validate_parse_mode(parse_mode: "str | None") -> None:
    """Fail loud on a mode this module cannot escape or truncate for.

    A typo like ``"markdown"`` (Telegram is case-sensitive: it accepts
    ``Markdown`` and ``MarkdownV2``, not ``markdown``) would otherwise be sent
    as-is, rejected with a 400 that looks like a formatting error, and then
    retried as plain text — a silent downgrade of every message from that
    caller. Raising at the call site is the honest outcome: the caller asked
    for something this transport does not do.
    """
    if parse_mode is not None and parse_mode not in PARSE_MODES:
        raise ValueError(
            f"parse_mode must be one of {PARSE_MODES!r} or None (plain text), "
            f"got {parse_mode!r}"
        )


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


def _truncate_html(text: str) -> str:
    """Truncate an HTML-mode body to a TAG BOUNDARY, then close what is open.

    :func:`_truncate_for_telegram` cuts at a character count. Under
    ``parse_mode="HTML"`` a cut can land inside ``<co`` (Telegram: "Unsupported
    start tag"), inside ``<a href="…`` (attribute never closed), or after a
    ``<pre>`` whose ``</pre>`` was in the discarded tail ("Unclosed start
    tag") — each of which rejects the WHOLE message, which is the outcome the
    truncation existed to prevent (alpha-engine-config-I9925, gotcha 3). So:

    1. cut at the same character budget as plain truncation;
    2. drop a trailing partial tag (a ``<`` with no ``>`` after it);
    3. re-close, innermost first, every tag still open at the cut, so the
       marker itself renders as plain text rather than inside a code block.

    The budget for step 3's closers is reserved up front from the worst case
    (the deepest nesting the body actually has), so the result never exceeds
    ``TELEGRAM_MESSAGE_MAX_CHARS``. Entities (``&lt;``) are never split: a cut
    inside ``&am`` renders those characters literally, which is ugly and
    accepted, not a rejection.
    """
    if len(text) <= TELEGRAM_MESSAGE_MAX_CHARS:
        return text
    suffix = f"\n…(truncated, showing {TELEGRAM_MESSAGE_MAX_CHARS} of {len(text)} chars)"
    keep = TELEGRAM_MESSAGE_MAX_CHARS - len(suffix)
    # Reserve room for the closers of the deepest nesting in the whole body:
    # cheaper than iterating, and the over-reservation is at most a few tags.
    keep -= _max_open_tag_closer_length(text)
    keep = max(keep, 0)
    head = text[:keep]
    lt = head.rfind("<")
    if lt != -1 and head.find(">", lt) == -1:
        head = head[:lt]
    closers = "".join(f"</{tag}>" for tag in reversed(_open_tags(head)))
    return head + closers + suffix


def _open_tags(fragment: str) -> list:
    """The stack of Telegram-HTML tags still open at the end of ``fragment``."""
    stack: list = []
    i = 0
    while True:
        i = fragment.find("<", i)
        if i == -1:
            break
        j = fragment.find(">", i)
        if j == -1:
            break
        inner = fragment[i + 1 : j].strip()
        i = j + 1
        if not inner:
            continue
        closing = inner.startswith("/")
        name = inner.lstrip("/").split()[0].lower() if inner.lstrip("/").split() else ""
        if name not in _HTML_TAGS:
            continue
        if closing:
            if name in stack:
                # Pop to and including the matching opener; Telegram nests
                # strictly, so anything above it was already malformed.
                while stack and stack[-1] != name:
                    stack.pop()
                if stack:
                    stack.pop()
        else:
            stack.append(name)
    return stack


def _max_open_tag_closer_length(text: str) -> int:
    """Characters the closers would need at the deepest nesting in ``text``."""
    deepest = 0
    stack: list = []
    i = 0
    while True:
        i = text.find("<", i)
        if i == -1:
            break
        j = text.find(">", i)
        if j == -1:
            break
        inner = text[i + 1 : j].strip()
        i = j + 1
        parts = inner.lstrip("/").split()
        name = parts[0].lower() if parts else ""
        if name not in _HTML_TAGS:
            continue
        if inner.startswith("/"):
            if name in stack:
                while stack and stack[-1] != name:
                    stack.pop()
                if stack:
                    stack.pop()
        else:
            stack.append(name)
            deepest = max(deepest, sum(len(f"</{t}>") for t in stack))
    return deepest


def _escape_markdown(text: str) -> str:
    """Escape Telegram Markdown v1 special characters.

    Escapes characters Telegram interprets as formatting markers
    (``_``, `````, ``[``, ``]``) to prevent 400 Bad Request parse errors.
    Preserves ``*`` for bold markers which callers control via message
    templates.

    **This SUBSTITUTED rather than escaped until 2026-08-13, and a substitution
    emits a fact that is false** (`alpha-engine-config-I7168`). ``_`` became
    ``-``, ``[`` became ``(``, and a backtick became an apostrophe — so every
    identifier in every fleet alert was rewritten on the way to the operator.

    Live instance: a box-health WARNING named an undeclared database as
    ``/home/ec2-user/flow-doctor/flow-doctor.db``. That path does not exist. The
    real file is ``flow_doctor.db``, and the box carries two OTHER files
    genuinely named ``flow-doctor.db`` under ``morning-signal/``, already
    declared, with different dispositions. So the alert did not merely look
    wrong — it named a real, different file than the one it had found. The
    detector's own output was correct throughout.

    A path, a unit name, an S3 key and an ArcticDB symbol are identifiers: the
    reader is expected to copy them into a command. Escaping preserves them;
    substitution silently hands over something else. ``\\_`` renders as a
    literal underscore under Markdown v1, so nothing about the original reason
    for this function is given up.

    The residual risk is a v1 parse error from an escape sequence Telegram
    dislikes, and it is already covered: :func:`send_message` retries once
    without ``parse_mode`` on exactly that 400, which sends the identical body
    as plain text. Formatting degrades; the identifier survives either way.

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
    # Backslash FIRST, or an escape this function adds could be neutralised by
    # a backslash the caller's text already carried.
    return (
        text
        .replace("\\", "\\\\")
        .replace("_", "\\_")
        .replace("`", "\\`")
        .replace("[", "\\[")
        .replace("]", "\\]")
    )


def send_message(
    text: str,
    *,
    disable_notification: bool = False,
    bot_token: str | None = None,
    chat_id: str | int | None = None,
    message_thread_id: int | None = None,
    parse_mode: str | None = PARSE_MODE,
) -> bool:
    """Send a single Telegram message to the channel resolved from secrets.

    ``parse_mode`` (alpha-engine-config-I9925) selects how the body is
    interpreted, and with it how this function prepares the body:

    - ``"Markdown"`` (the default, unchanged for every existing caller): v1
      escaping via :func:`_escape_markdown`; ``*bold*`` is the caller's.
    - ``"HTML"``: **no escaping** — the caller owns the markup and has already
      run every interpolated string through :func:`escape_html`. Truncation
      respects tag boundaries (:func:`_truncate_html`).
    - ``None``: plain text. No escaping, and the payload carries **no**
      ``parse_mode`` key at all (Telegram rejects an explicit null).

    Any other value raises ``ValueError`` at the call site rather than being
    sent, rejected and silently downgraded on the retry path.

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
    :param parse_mode: ``"Markdown"`` (default), ``"HTML"`` or ``None`` for
        plain text — see above.
    :returns: ``True`` if the Telegram API returned HTTP 200, ``False``
        otherwise (missing secrets, network error, non-200 response). A 400
        caused by unparseable entities (Markdown or HTML) is retried once as
        plain text before ``False`` is returned — see the fallback comment in
        the body.
    """
    _validate_parse_mode(parse_mode)
    token = bot_token or get_secret("TELEGRAM_BOT_TOKEN", required=False)
    resolved_chat = chat_id if chat_id is not None else get_secret("TELEGRAM_CHAT_ID", required=False)
    if not token or resolved_chat in (None, ""):
        logger.warning(
            "Telegram not configured — TELEGRAM_BOT_TOKEN=%s TELEGRAM_CHAT_ID=%s",
            "set" if token else "MISSING",
            "set" if resolved_chat not in (None, "") else "MISSING",
        )
        return False

    # Escaping is per mode, and truncation runs BEFORE escaping (so the escape
    # sequences the transport adds are never what pushes a body over the
    # limit, and never what gets cut in half). HTML is the one mode where the
    # cut itself must respect structure — see `_truncate_html`.
    if parse_mode == PARSE_MODE:
        body_text = _escape_markdown(_truncate_for_telegram(text))
    elif parse_mode == PARSE_MODE_HTML:
        body_text = _truncate_html(text)
    else:
        body_text = _truncate_for_telegram(text)

    payload = {
        "chat_id": resolved_chat,
        "text": body_text,
        "disable_notification": disable_notification,
    }
    if parse_mode is not None:
        # Plain text is the ABSENCE of the key, not `parse_mode: null`.
        payload["parse_mode"] = parse_mode
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
        # request URL cannot leak the token into the log.
        #
        # Drop the whole description rather than splicing the secret out of it.
        # A surgical `detail.replace(token, ...)` reads as safer but puts the
        # credential on a live dataflow path from the secret into the log sink —
        # correct only for as long as nobody reorders these two lines, and
        # `py/clear-text-logging-sensitive-data` flags it for exactly that
        # reason. There is nothing to preserve anyway: a body echoing our bot
        # token is not a Telegram error body, so its content is not the
        # operationally useful field this branch exists to log.
        if token in detail:
            detail = "<description echoing the bot token suppressed> [REDACTED]"
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
    #
    # The same retry covers HTML mode (I9925): an unbalanced or unsupported
    # tag in caller-owned markup is the HTML analogue of the unpaired `*`, and
    # Telegram names it with the same `can't parse entities` marker. A
    # plain-text body was never sent with `parse_mode`, so it has nothing to
    # retry — `parse_mode` is not in its payload and the branch is inert.
    if (
        not ok
        and description is not None
        and "parse_mode" in payload
        and _is_entity_parse_error(description)
    ):
        retry_payload = {k: v for k, v in payload.items() if k != "parse_mode"}
        ok, _ = _post(retry_payload)
        logger.warning(
            "Telegram %s parse failed; %s as plain text "
            "(under Markdown krepis._escape_markdown preserves '*', which "
            "arbitrary interpolated text can leave unpaired; under HTML the "
            "caller owns the tags)",
            parse_mode,
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
