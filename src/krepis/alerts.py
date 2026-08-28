"""
Unified failure-surveillance fan-out for Alpha Engine modules.

Consolidation substrate for the **"fire an operator alert from a failure
site"** pattern that has appeared inline across the fleet:

* :file:`alpha-engine/infrastructure/health_checker.sh` — raw ``curl`` to
  Telegram bot API
* :file:`alpha-engine-data/infrastructure/lambdas/changelog-incident-mirror/deploy.sh`
  — raw ``aws sns publish`` to ``alpha-engine-alerts``
* ROADMAP L116/L117 — names 5 more Lambda-deploying repos that need the
  same canary-rollback alert primitive ("Mirror in all 5 Lambda-deploying
  repos … same recurrence class as ``feedback_env_regression_recurs_per_repo_spot_script``
  — fix forward across all repos in one pass, not per-repo at incident time")

Per the ``~/Development/CLAUDE.md`` SOTA / institutional-approach rule
(sub-sub-rule: lift to lib when ≥2 consumers exist), this module is the
canonical Python primitive backing all consumers. Bash callers reach it
via the CLI entry (``python -m krepis.alerts publish ...``) —
mirrors the the transparency CLI ``--cadence daily/weekly``
CLI convention.

**Public API:**

- :func:`publish` — fan-out to both SNS (``alpha-engine-alerts`` topic →
  email) and Telegram (``@nous_ergon_alerts_bot`` channel) by default.
  Each channel is independently best-effort — failure in one does not
  block the other. Returns a :class:`PublishResult` dataclass with the
  per-channel outcome for caller observability.
- :func:`publish_clear` — the OTHER half of the pair (I8105). A publisher
  that tracks a set of currently-true conditions calls this for each
  condition that has left the set, passing the same ``identity_key`` its
  page carried. Delivered at ``info`` severity, silent (no phone push),
  ``state="cleared"``, no dedup key. Before this existed every alert in the
  fleet was WRITE-ONCE: a condition that cleared emitted nothing, so no page
  could be told from a live outage.
- :func:`diff_conditions` — the set arithmetic a publisher runs between two
  observations to get ``(opened, still_open, cleared)``.
- :func:`resolve_destination` — the pure severity/config → destination
  decision, exposed so a call site (or a test) can assert on where a finding
  WOULD go without publishing it. Destinations are
  :data:`DESTINATION_OPERATOR_CHAT`, :data:`DESTINATION_LOG_CHAT` and
  :data:`DESTINATION_CONSOLE_ONLY` (:data:`ALERT_DESTINATIONS`), reported
  back on :attr:`PublishResult.telegram_destination`.
- CLI: ``python -m krepis.alerts publish --message "..."
  --severity error --source "..."`` and ``python -m krepis.alerts clear
  --message "..." --identity-key "..."``. Designed for Bash failure-trap
  callers (``cleanup()`` in spot dispatchers, ``deploy.sh`` rollback
  branches). Exit code is ``0`` if *either* channel succeeded, ``1`` if
  *both* failed.

**Dry-run** (``--dry-run`` CLI flag / ``dry_run=True`` kwarg, config-I6759).
Runs argument parsing + ``_format_message`` and reports a synthetic
``ok=True, detail="dry-run: would send"`` :class:`ChannelResult` per
channel — no SNS publish, no Telegram call, no dedup marker write, no
Overseer intake event, and no boto3 client construction. The
short-circuit fires before the dedup check and before the
``PYTEST_CURRENT_TEST`` guard, so it is deterministic in every caller
environment. Use it to verify a delivery call site's argument shape
without paging the operator (PR165 paged Brian with a synthetic ERROR
because ``publish()`` previously only suppressed fan-out under
``PYTEST_CURRENT_TEST``).

**Severity gates DESTINATION and BUZZ. It never gates DELIVERY.**
``severity`` is a free-form string prepended to the message (``[ERROR]
...`` / ``[WARNING] ...``) for both channels. It decides two things and
only two:

1. **Which Telegram destination the message goes to.** Severities in
   :data:`SEVERITY_PHONE_PUSH` go to the operator incident channel
   (:data:`DESTINATION_OPERATOR_CHAT`). Every other severity goes to a
   NON-operator destination — the log channel
   (:data:`DESTINATION_LOG_CHAT`) when ``TELEGRAM_LOG_CHAT_ID`` is
   configured, or nowhere on Telegram at all
   (:data:`DESTINATION_CONSOLE_ONLY`) when the caller passes
   ``console_artifact=`` naming the durable surface the finding is ALSO
   published to.
2. **Whether the send buzzes the phone** (``disable_notification``).
   ``error``/``critical`` buzz; nothing else does.

**SNS delivery is byte-identical at every severity.** Routing is a
Telegram-only concept; the durable record on ``alpha-engine-alerts``
does not move, whatever the destination resolves to.

**The fallback rule — a finding is NEVER dropped.** A non-pushing
severity with NEITHER a configured log chat NOR a ``console_artifact``
falls back to the operator chat exactly as it did before this routing
existed, and logs at WARNING that it did so. There is no code path in
this module that discards a finding because of its severity: the worst
case is that it lands in the incident channel and the operator sees a
WARNING in the log explaining that the routing was unconfigured. Do not
choose ``info``/``warning`` believing the finding will be invisible, and
do not pass ``console_artifact`` for a surface you have not actually
published to — that argument is the EVIDENCE that makes skipping the
operator chat safe, and it is the only thing that does.

This is alpha-engine-config-I7857 expressed in code rather than prose.
That issue was filed after a fleet of publishers assumed a non-``error``
severity would be unseen, when the message was landing in the operator's
chat the whole time. The pre-2026-08-28 module said, correctly for its
behaviour, that severity was not a delivery gate at all: every severity
reached SNS and the one Telegram chat, and the only tier was the buzz.
Severity now moves a message to a different destination — it still never
removes it from one.

**SNS topic resolution.** Defaults to
``arn:aws:sns:{region}:{account_id}:alpha-engine-alerts``, with
``region`` from ``AWS_REGION``/``AWS_DEFAULT_REGION`` (fallback
``us-east-1``) and ``account_id`` resolved via ``sts:GetCallerIdentity``.
Override with the ``--sns-topic-arn`` CLI flag or ``sns_topic_arn``
kwarg.

**Failure behavior.** Never raises. SNS errors (boto3 ``ClientError``,
network) and Telegram errors both log at WARNING and return a
:class:`PublishResult` with the failed channel marked ``ok=False``. This
is by design — the caller is already in a failure path; secondary
surveillance failure must not mask the primary error.

**Source-keyed suppression** (v0.57.0, alpha-engine-config mute-arc).
Distinct from dedup: an operator can mute an entire alert *source*
(e.g. ``"metron"``) for a stated, expiring window via a JSON list in
SSM (:data:`DEFAULT_MUTE_SSM_PARAM`) — see the :func:`publish`
``mute_ssm_param`` parameter and :attr:`PublishResult.muted`. Checked
before dedup; fails open (never suppresses) on any fetch/parse error
or on a missing/expired ``expires_at``.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Final

from krepis import _dedup, fleet_events

logger = logging.getLogger(__name__)

DEFAULT_SNS_TOPIC_NAME: Final[str] = "alpha-engine-alerts"
DEFAULT_REGION: Final[str] = "us-east-1"

# Severities that trigger a Telegram PHONE-PUSH notification
# (``disable_notification=False``) AND route to the operator incident chat.
# Membership here is still NOT a delivery gate: every severity is published
# to SNS byte-identically, and every severity is delivered on Telegram — a
# non-member goes to the LOG chat when one is configured, to the console
# alone when the caller names the artifact it is published to, and otherwise
# to this same operator chat with a WARNING logged. Read the module
# docstring's "Severity gates DESTINATION and BUZZ" paragraph before writing
# a new call site that picks a severity believing it will be unseen — it
# will not be. See :func:`resolve_destination`.
SEVERITY_PHONE_PUSH: Final[frozenset[str]] = frozenset({"error", "critical"})

# Deprecated alias — kept additive-only per CONTRIBUTING.md ("renaming or
# removing an exported name breaks consumers at import time"). No known
# fleet call site imports this name directly (alpha-engine-config-I7857
# audit, 2026-08-20 — every fleet reference was a comment, not an import);
# it is kept anyway in case an external consumer does. The OLD name is
# exactly the naming defect this rename fixes: a reader can misparse
# "SEVERITY_PUSH" as "severities that get pushed [and everything else is
# not]" rather than its true meaning, "severities that get a phone buzz on
# top of the delivery every severity already gets." Prefer
# :data:`SEVERITY_PHONE_PUSH` in new code. Remove once a migration window
# has passed with no external usage confirmed.
SEVERITY_PUSH: Final[frozenset[str]] = SEVERITY_PHONE_PUSH

# ── Delivery DESTINATION (alpha-engine-config-I7857, 2026-08-28) ────────────
# Severity tiering was "fixed" twice (2026-07-29 tier split, 2026-08-20
# cadence 60→1440) without either change touching the thing that actually
# reaches the operator: WHICH CHAT the message lands in. Both changes altered
# whether a message buzzed or how often it repeated; neither could keep a
# non-incident out of the incident channel, because there was exactly one
# Telegram destination and every severity was posted into it.
#
# A destination is therefore a first-class, named concept — not an implicit
# consequence of a boolean. Three of them, and every publish resolves to
# exactly one:
#
#   operator_chat  the incident channel Brian reads (TELEGRAM_CHAT_ID). Where
#                  error/critical go, and where EVERYTHING goes when nothing
#                  else is configured.
#   log_chat       a second Telegram destination (TELEGRAM_LOG_CHAT_ID, plus
#                  an optional forum-topic thread id). Delivered, readable,
#                  out of the incident channel.
#   console_only   no Telegram send at all — legal ONLY when the caller names
#                  the durable surface the finding is also published to.
#
# `console_only` is the one that could become a silent drop, so it is the one
# that requires evidence: `console_artifact` must name the artifact / envelope
# URI. The pattern it encodes already exists downstream in
# `crucible-dashboard/infrastructure/emit_box_health_hygiene.py`, which routes
# its `notice` tier to the console alone and is safe for reasons this argument
# is asserting on the caller's behalf: the envelope is published on EVERY run
# including clean ones, `ran_at` + `cadence_minutes` make a dead emitter read
# STALE rather than green, and a missing artifact renders `unreadable`, never
# `ok`. A destination that can go dark without saying so is a drop with extra
# steps.
DESTINATION_OPERATOR_CHAT: Final[str] = "operator_chat"
DESTINATION_LOG_CHAT: Final[str] = "log_chat"
DESTINATION_CONSOLE_ONLY: Final[str] = "console_only"
ALERT_DESTINATIONS: Final[tuple[str, ...]] = (
    DESTINATION_OPERATOR_CHAT,
    DESTINATION_LOG_CHAT,
    DESTINATION_CONSOLE_ONLY,
)

# ── Naming, and why it is not cosmetic ─────────────────────────────────────
# These two constants hold the NAME of a parameter, never its value. They are
# suffixed ``_PARAM`` rather than ``_SECRET`` for that reason, and the
# distinction is load-bearing twice over:
#
#   * A reader (and CodeQL's `py/clear-text-logging-sensitive-data`
#     name heuristic) treats an identifier ending ``_SECRET`` as holding
#     secret material. Logging one is then indistinguishable from logging a
#     credential, whether or not it is. Naming the thing for what it holds —
#     a parameter name — is the honest fix; suppressing the rule would not
#     have made the code any easier to read correctly.
#   * The VALUES these name are resolved by :func:`_resolve_log_chat` and are
#     credential-adjacent: a chat id addresses a channel the bot token can
#     post to. Nothing in this module may log a resolved chat or thread id,
#     put one in :attr:`PublishResult.destination_reason`, or return one in a
#     :class:`ChannelResult` detail — those surfaces are shipped to journald,
#     CloudWatch and S3 run logs (``~/Development/CLAUDE.md``, "CLI output
#     safety"). Routing facts are reported as a DESTINATION name plus a
#     configured/not-configured boolean, which is everything an operator
#     needs to diagnose routing and nothing an attacker needs to reach the
#     chat. `TestDestinationReasonLeaksNoChatId` pins that.
#
#: Name of the parameter holding the non-operator Telegram chat id. Resolved
#: through :func:`krepis.secrets.get_secret` with ``required=False``, exactly
#: like ``TELEGRAM_CHAT_ID`` — absent is a normal, supported state (it is what
#: every fleet box looks like until an operator configures one), and the
#: fallback rule covers it.
TELEGRAM_LOG_CHAT_PARAM: Final[str] = "TELEGRAM_LOG_CHAT_ID"
#: Name of the parameter holding an optional forum-topic id within the log
#: chat, for a supergroup that files alert traffic into a topic rather than a
#: whole separate chat. Ignored unless :data:`TELEGRAM_LOG_CHAT_PARAM` also
#: resolves.
TELEGRAM_LOG_THREAD_PARAM: Final[str] = "TELEGRAM_LOG_MESSAGE_THREAD_ID"

# ── Dedup (v0.24.0; marker mechanism lifted to krepis._dedup in v0.NEXT) ────
# When the caller passes a ``dedup_key``, ``publish`` writes a marker at
# ``s3://{dedup_bucket}/{DEDUP_MARKER_PREFIX}/{sha1(dedup_key)[:16]}.json``
# after the first successful publish. Subsequent calls with the same
# ``dedup_key`` within ``dedup_window_min`` minutes find the marker and
# skip the publish. See the :func:`publish` docstring. The underlying
# S3-marker check/write mechanism now lives in :mod:`krepis._dedup`
# (shared with :mod:`krepis.email_sender` — config#2291); this module keeps
# its own ``DEDUP_MARKER_PREFIX`` namespace so the two dedup domains never
# collide.
DEFAULT_DEDUP_BUCKET: Final[str] = _dedup.DEFAULT_DEDUP_BUCKET
DEDUP_MARKER_PREFIX: Final[str] = "_alerts/_dedup"
DEFAULT_DEDUP_WINDOW_MIN: Final[int] = 60

# ── Source-keyed suppression (v0.57.0; alpha-engine-config-I<mute-issue>) ───
# Distinct from dedup: dedup rate-limits a *recurring* alert (resets on any
# new finding text — a systemd-drift finding for a different unit resets the
# window even while metron is "muted"); a mute is an operator-declared,
# time-boxed "stop paging me about this SOURCE" that holds regardless of
# message content. Config lives in SSM (not a file in alpha-engine-config)
# because ``publish`` callers span EC2 boxes, Lambdas and CI runners that do
# not reliably have that private repo checked out, but already reach SSM for
# every other runtime secret/toggle (mirrors ``krepis.secrets`` /
# ``krepis.router``'s ``ssm.get_parameter`` pattern). One parameter holds a
# JSON list so an operator edits one value instead of one-param-per-mute.
DEFAULT_MUTE_SSM_PARAM: Final[str] = "/alpha-engine/alerts/source_mutes"

# ── Condition lifecycle: the open/clear pair (alpha-engine-config-I8105) ────
# Every alert this module publishes used to be WRITE-ONCE: a publisher emitted
# on detection and emitted nothing when the condition ended, so a page and a
# live outage were indistinguishable to every downstream consumer — a human
# reading a digest, the Overseer alert-drain (which ingests events as evidence
# of a CURRENT condition), and the console (whose last known state stayed the
# failure forever). `principles.md` §2.7 asks what the ABSENCE of a signal
# looks like on the surface; the absence of a terminator rendered as an
# ongoing outage, indefinitely.
#
# The fix is a first-class LIFECYCLE on the record rather than a second prose
# email. A publisher that tracks a condition set declares which of three
# things this emission is:
#
#   opened      the condition was not present at the previous observation
#   still_open  the condition was present then and is present now
#   cleared     the condition was present then and is ABSENT now
#
# `state` rides on the `nousergon.alert.v1` event (additive, optional) so
# `alert_drain_ingest.py` and the console adapter can pair a page to its clear
# WITHOUT string-matching prose. `identity_key` is what they pair ON: the
# publisher's own stable identity for the condition (for box-health's timer
# findings, unit name + the failing run's InactiveExitTimestamp — see
# alpha-engine-config-I7677), deliberately NOT a computed relative age and not
# the message text.
#
# WHY identity_key IS SEPARATE FROM dedup_key EVEN WHEN THEY CARRY THE SAME
# STRING. dedup_key is a SUPPRESSION input: passing it makes `publish` check
# the S3 marker and skip. A clear that reused the page's dedup_key would be
# suppressed by the page's own still-live marker — the terminator swallowed by
# the mechanism that rate-limits the thing it terminates. So the clear carries
# `identity_key` (correlation, never suppression) and no `dedup_key`, and the
# drain's `compute_corr_key` reads `identity_key` first so both land on one
# incident anyway.
ALERT_STATE_OPENED: Final[str] = "opened"
ALERT_STATE_STILL_OPEN: Final[str] = "still_open"
ALERT_STATE_CLEARED: Final[str] = "cleared"
ALERT_STATES: Final[tuple[str, ...]] = (
    ALERT_STATE_OPENED,
    ALERT_STATE_STILL_OPEN,
    ALERT_STATE_CLEARED,
)

# Severity a recovery is published at. `info`, never `error`/`critical`:
# alpha-engine-config-I7857 established that severity does NOT gate delivery
# here (every severity reaches SNS, and reaches a Telegram destination) — it
# gates the phone push and, since 2026-08-28, WHICH destination. A clear that buzzes Brian's phone at 8pm is a second alert, not
# a resolution, so the recovery is delivered and silent. `publish_clear` also
# forces `silent=True` explicitly rather than leaning on `info` being outside
# SEVERITY_PHONE_PUSH, so a future widening of that set cannot turn every
# all-clear in the fleet into a push.
CLEAR_SEVERITY: Final[str] = "info"

# Human-facing prefix on a cleared emission. A reader scanning a channel must
# be able to tell a resolution from a fresh page in the first two words,
# without parsing the JSON that machines read.
CLEAR_MESSAGE_PREFIX: Final[str] = "RESOLVED"


@dataclass
class ChannelResult:
    """Per-channel outcome from a :func:`publish` call."""

    ok: bool
    detail: str = ""


@dataclass
class PublishResult:
    """Aggregated outcome from a :func:`publish` call.

    ``sns`` and ``telegram`` are independent — a publish may succeed in
    one channel and fail in the other. :attr:`any_ok` is the typical
    caller gate (success = at least one channel delivered the alert);
    :attr:`all_ok` is the strict variant for callers that want both.

    When the caller passes ``dedup_key`` and an earlier publish for the
    same key is still within window, :attr:`dedup_skipped` is True and
    neither channel is attempted; :attr:`any_ok` still reports True
    (the alert is logically in the operator's hands by virtue of the
    earlier successful publish).

    When the alert's ``source`` matches a live entry in the source-mute
    list, :attr:`muted` is True and neither channel is attempted;
    :attr:`any_ok` reports True (an intentional operator-declared skip,
    not a delivery failure — a Bash caller's ``|| echo '...failed'``
    fallback must not fire for a mute).
    """

    sns: ChannelResult = field(default_factory=lambda: ChannelResult(ok=False, detail="not attempted"))
    telegram: ChannelResult = field(default_factory=lambda: ChannelResult(ok=False, detail="not attempted"))
    dedup_skipped: bool = False
    dedup_reason: str = ""
    muted: bool = False
    mute_reason: str = ""
    #: Condition lifecycle this publish carried (I8105). Echoed back so a
    #: caller that computed it from a set difference can assert on it without
    #: re-deriving, and so a test can tell an ``opened`` from a ``cleared``.
    state: str = "opened"
    #: The condition identity a consumer pairs page-to-clear on; falls back to
    #: ``dedup_key`` when the caller passed one and no explicit identity.
    identity_key: str | None = None
    #: Which Telegram destination this publish resolved to — one of
    #: :data:`ALERT_DESTINATIONS`, or ``None`` when the Telegram leg was not
    #: reached at all (``telegram=False``, dry-run, mute, dedup-skip, test-env
    #: guard). Machine-readable on purpose: a caller or a test asserting on
    #: routing must not have to parse :attr:`ChannelResult.detail` prose, and
    #: a test that asserts the message TEXT cannot catch a wrong TIER.
    telegram_destination: str | None = None
    #: Why the destination resolved the way it did — in particular, whether a
    #: non-pushing severity reached the operator chat because it was ROUTED
    #: there or because nothing else was configured (the I7857 fallback).
    destination_reason: str = ""

    @property
    def any_ok(self) -> bool:
        if self.dedup_skipped or self.muted:
            return True
        return self.sns.ok or self.telegram.ok

    @property
    def all_ok(self) -> bool:
        if self.dedup_skipped or self.muted:
            return True
        return self.sns.ok and self.telegram.ok


def _resolve_sns_topic_arn(explicit: str | None) -> str | None:
    """Return the SNS topic ARN, resolving from env + STS if not explicit."""
    if explicit:
        return explicit
    from krepis.aws_region import resolve_region

    region = resolve_region()
    try:
        import boto3

        account_id = boto3.client("sts", region_name=region).get_caller_identity()["Account"]
    except Exception as exc:  # boto3 missing, STS unreachable, creds bad
        logger.warning("alerts.publish: SNS topic ARN resolution failed: %s", exc)
        return None
    return f"arn:aws:sns:{region}:{account_id}:{DEFAULT_SNS_TOPIC_NAME}"


def _format_message(
    message: str,
    severity: str,
    source: str | None,
    state: str = ALERT_STATE_OPENED,
) -> str:
    """Prepend severity tag + source prefix (+ RESOLVED marker) to the body.

    ``opened`` and ``still_open`` render exactly as before — the lifecycle is
    carried on the event, not spelled into every page. Only ``cleared`` adds a
    visible marker, because a human scanning the channel has to tell a
    resolution from a fresh page without reading the JSON.
    """
    tag = f"[{severity.upper()}]"
    body = message
    if state == ALERT_STATE_CLEARED:
        body = f"{CLEAR_MESSAGE_PREFIX} — {message}"
    if source:
        return f"{tag} {source}: {body}"
    return f"{tag} {body}"


def _publish_sns(arn: str, message: str, subject: str | None = None) -> ChannelResult:
    try:
        import boto3

        region = arn.split(":")[3] if ":" in arn else DEFAULT_REGION
        client = boto3.client("sns", region_name=region)
        kwargs: dict = {"TopicArn": arn, "Message": message}
        if subject:
            # SNS subject is limited to 100 chars + ASCII + no newlines.
            cleaned = subject.replace("\n", " ").replace("\r", " ")[:100]
            kwargs["Subject"] = cleaned
        resp = client.publish(**kwargs)
        return ChannelResult(ok=True, detail=resp.get("MessageId", "<no id>"))
    except Exception as exc:
        logger.warning("alerts.publish: SNS publish failed: %s", exc)
        return ChannelResult(ok=False, detail=f"sns error: {exc!r}")


def _resolve_log_chat() -> tuple[str | None, int | None]:
    """Resolve the non-operator Telegram destination from secrets.

    Returns ``(chat_id, message_thread_id)``; ``(None, None)`` when no log
    chat is configured. Both are optional secrets — an unconfigured fleet is
    the normal state, and :func:`resolve_destination` handles it by falling
    back rather than by dropping.

    A thread id that will not parse as an int is dropped with a WARNING and
    the chat is still used: a malformed topic id must degrade to "the log
    chat, no topic", never to "no delivery".

    **Neither resolved value is ever logged or returned in a message.** Both
    come from :func:`krepis.secrets.get_secret`, and a chat id addresses a
    channel this bot token can post into — it is credential-adjacent, and
    every surface that would carry it (WARNING logs, ``destination_reason``,
    ``ChannelResult.detail``) is shipped off the box. The WARNING below
    therefore reports that the topic id is unparseable and names the
    parameter to fix; it does not echo what was read.
    """
    from krepis.secrets import get_secret

    try:
        chat_id = get_secret(TELEGRAM_LOG_CHAT_PARAM, required=False)
    except Exception as exc:  # secrets backend unreachable — never fatal here
        logger.warning(
            "alerts: log-chat secret lookup failed (%s); routing falls back", exc
        )
        return None, None
    if chat_id in (None, ""):
        return None, None

    thread_id: int | None = None
    try:
        raw_thread = get_secret(TELEGRAM_LOG_THREAD_PARAM, required=False)
    except Exception:
        raw_thread = None
    if raw_thread not in (None, ""):
        try:
            thread_id = int(str(raw_thread).strip())
        except (TypeError, ValueError):
            # The VALUE is deliberately absent from this line. "which
            # parameter, and what is wrong with it" is the whole of what an
            # operator needs to fix it; the value itself would put a
            # secrets-resolved string into journald.
            logger.warning(
                "alerts: the configured Telegram log topic id is not an "
                "integer — delivering to the log chat without a thread. Fix "
                "the value of the %s parameter (value not logged).",
                TELEGRAM_LOG_THREAD_PARAM,
            )
    return str(chat_id), thread_id


def resolve_destination(
    severity: str,
    *,
    destination: str | None = None,
    console_artifact: str | None = None,
    log_chat_id: str | None = None,
) -> tuple[str, str]:
    """Decide which destination an emission goes to, and say why.

    Pure given its inputs — the secret lookup happens in
    :func:`_resolve_log_chat` and is passed in as ``log_chat_id`` — so the
    decision itself is testable without a secrets backend.

    Order, and the ONLY order (see the module docstring):

    1. An explicit ``destination`` is the caller's intent and is honoured
       where it can be. It is still subject to steps 3 and 4: asking for a
       log chat that is not configured, or for ``console_only`` with no
       artifact named, cannot delete a finding.
    2. A severity in :data:`SEVERITY_PHONE_PUSH` goes to the operator chat.
       Nothing overrides this away except an explicit ``destination``.
    3. Otherwise: the log chat if one is configured, else console-only if the
       caller supplied evidence of a durable surface.
    4. Otherwise: the operator chat, logged at WARNING. This is the
       alpha-engine-config-I7857 invariant — there is no fourth branch, and
       no branch that returns "delivered nowhere".

    :returns: ``(destination, reason)`` — ``reason`` is recorded on
        :attr:`PublishResult.destination_reason` so an operator reading a
        result can tell a ROUTED operator-chat delivery from a fallback one.
        It names the DESTINATION and, where it matters, whether a log chat
        was configured — a boolean. It NEVER carries a resolved chat or
        thread id: the reason is serialized into run logs, and the id is
        credential-adjacent (see the ``_PARAM`` naming note above).
        ``log_chat_id`` is read here only for its truthiness.
    """
    if destination is not None and destination not in ALERT_DESTINATIONS:
        raise ValueError(
            f"alerts: unknown destination {destination!r}; expected one of "
            f"{ALERT_DESTINATIONS}"
        )

    if destination == DESTINATION_OPERATOR_CHAT:
        return DESTINATION_OPERATOR_CHAT, "explicit destination=operator_chat"

    if destination == DESTINATION_LOG_CHAT:
        if log_chat_id:
            return (
                DESTINATION_LOG_CHAT,
                "explicit destination=log_chat; log_chat configured=True",
            )
        logger.warning(
            "alerts: destination=log_chat requested but %s is not configured — "
            "delivering to the OPERATOR chat instead. A finding is never "
            "dropped for want of routing config (alpha-engine-config-I7857).",
            TELEGRAM_LOG_CHAT_PARAM,
        )
        return (
            DESTINATION_OPERATOR_CHAT,
            "fallback: destination=log_chat requested but log_chat "
            "configured=False",
        )

    if destination == DESTINATION_CONSOLE_ONLY:
        if console_artifact:
            return (
                DESTINATION_CONSOLE_ONLY,
                f"explicit destination=console_only, published to {console_artifact}",
            )
        logger.warning(
            "alerts: destination=console_only requested with no "
            "console_artifact naming where the finding IS published — "
            "delivering to the OPERATOR chat instead. console_artifact is the "
            "evidence that makes skipping the chat safe "
            "(alpha-engine-config-I7857).",
        )
        return (
            DESTINATION_OPERATOR_CHAT,
            "fallback: destination=console_only requested with no console_artifact",
        )

    # ── Severity-derived routing (no explicit destination) ───────────────
    if severity.lower() in SEVERITY_PHONE_PUSH:
        return (
            DESTINATION_OPERATOR_CHAT,
            f"severity={severity!r} is an incident tier",
        )

    if log_chat_id:
        return (
            DESTINATION_LOG_CHAT,
            f"severity={severity!r} is not an incident tier; "
            f"log_chat configured=True",
        )

    if console_artifact:
        return (
            DESTINATION_CONSOLE_ONLY,
            f"severity={severity!r} is not an incident tier; also published to "
            f"{console_artifact}",
        )

    logger.warning(
        "alerts: severity=%r is not an incident tier, but neither %s nor a "
        "console_artifact is configured for this call — delivering to the "
        "OPERATOR chat. This is the fallback, not the intent: configure the "
        "log chat, or pass console_artifact naming the durable surface this "
        "finding is also published to. A finding is never dropped "
        "(alpha-engine-config-I7857).",
        severity, TELEGRAM_LOG_CHAT_PARAM,
    )
    return (
        DESTINATION_OPERATOR_CHAT,
        f"fallback: severity={severity!r} is non-pushing; log_chat "
        f"configured=False and no console_artifact was supplied",
    )


def _publish_telegram(
    message: str,
    severity: str,
    silent: bool | None = None,
    *,
    destination: str = DESTINATION_OPERATOR_CHAT,
    chat_id: str | None = None,
    message_thread_id: int | None = None,
) -> ChannelResult:
    """Send one message to the resolved Telegram ``destination``.

    ``chat_id`` / ``message_thread_id`` are the log-chat overrides;
    ``None`` means "the operator chat resolved from ``TELEGRAM_CHAT_ID``",
    which is what :func:`krepis.telegram.send_message` does by default.
    ``destination`` is carried through only to name itself in the returned
    :attr:`ChannelResult.detail` — the routing decision was already made by
    :func:`resolve_destination`.
    """
    try:
        from krepis.telegram import send_message

        # Every severity reaches the chat. `disable_push` only controls
        # whether THIS send also triggers a phone buzz (error/critical do;
        # everything else still posts, just without the buzz) — it is never
        # a delivery gate. See SEVERITY_PHONE_PUSH and the module docstring.
        #
        # `silent` is an explicit caller override of that severity-derived
        # decision, in ONE direction only: True forces the silent delivery,
        # None keeps the severity default. It exists so `publish_clear` does
        # not depend on `info` staying outside SEVERITY_PHONE_PUSH forever
        # (I8105) — a recovery must never push, whatever that set becomes.
        disable_push = severity.lower() not in SEVERITY_PHONE_PUSH
        if silent:
            disable_push = True
        ok = send_message(
            message,
            disable_notification=disable_push,
            chat_id=chat_id,
            message_thread_id=message_thread_id,
        )
        # The destination is named in `detail` as well as on
        # `PublishResult.telegram_destination` so a human reading a CLI
        # stderr line (which prints only the detail) can tell the operator
        # channel from the log channel without re-running anything.
        detail = (
            f"sent (destination={destination})"
            if ok
            else f"send_message returned False (destination={destination})"
        )
        return ChannelResult(ok=bool(ok), detail=detail)
    except Exception as exc:  # send_message itself never raises, but defensive
        logger.warning("alerts.publish: Telegram fan-out failed: %s", exc)
        return ChannelResult(
            ok=False, detail=f"telegram error (destination={destination}): {exc!r}"
        )


def _dedup_marker_key(dedup_key: str) -> str:
    """Stable S3 key for a dedup_key marker under this module's namespace.

    Thin wrapper over :func:`krepis._dedup.marker_key` — kept as a
    module-level function (rather than inlining the call at each call
    site) so existing tests / callers that reach into
    ``alerts._dedup_marker_key`` keep working unchanged.
    """
    return _dedup.marker_key(dedup_key, marker_prefix=DEDUP_MARKER_PREFIX)


def _check_dedup_marker(
    bucket: str,
    marker_key: str,
    *,
    dedup_window_min: int | None,
) -> tuple[bool, str]:
    """Check whether a recent publish for this dedup_key is still in window.

    Thin wrapper over :func:`krepis._dedup.check_marker` — see that
    function's docstring for the fail-safe contract.
    """
    return _dedup.check_marker(bucket, marker_key, dedup_window_min=dedup_window_min)


def _write_dedup_marker(
    bucket: str,
    marker_key: str,
    *,
    dedup_key: str,
    formatted_message: str,
) -> None:
    """Persist (or refresh) the dedup marker after a successful publish.

    Thin wrapper over :func:`krepis._dedup.write_marker`.
    """
    _dedup.write_marker(
        bucket, marker_key,
        dedup_key=dedup_key, message_preview=formatted_message,
    )


def _fetch_source_mutes(ssm_param: str) -> list[dict]:
    """Fetch + parse the source-mute list from SSM.

    Fail-*open* throughout: boto3 unavailable, the parameter missing,
    an SSM error, or unparseable/malformed JSON all resolve to "no
    mutes" (``[]``) so a fetch failure suppresses nothing — an alert
    that should have been muted but wasn't is a nuisance page; an alert
    that should have fired but was silently swallowed by a broken mute
    fetch is a missed incident. Never raises.
    """
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError as exc:
        logger.debug(
            "alerts.publish: mute check skipped — boto3 unavailable: %s", exc,
        )
        return []
    from krepis.aws_region import resolve_region

    region = resolve_region()
    try:
        client = boto3.client("ssm", region_name=region)
        resp = client.get_parameter(Name=ssm_param)
        raw = resp["Parameter"]["Value"]
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code != "ParameterNotFound":
            logger.debug(
                "alerts.publish: mute list fetch errored (fail-open, no "
                "mutes applied): %s", exc,
            )
        return []
    except Exception as exc:  # boto3 missing at call time, network, etc.
        logger.debug(
            "alerts.publish: mute list fetch errored (fail-open, no mutes "
            "applied): %s", exc,
        )
        return []

    try:
        import json

        entries = json.loads(raw)
    except Exception as exc:
        logger.warning(
            "alerts.publish: mute list at %s is not valid JSON (fail-open, "
            "no mutes applied): %s", ssm_param, exc,
        )
        return []
    if not isinstance(entries, list):
        logger.warning(
            "alerts.publish: mute list at %s is not a JSON list (fail-open, "
            "no mutes applied)", ssm_param,
        )
        return []
    return entries


def _find_live_mute(source: str | None, entries: list[dict]) -> dict | None:
    """Return the first live (non-expired) mute entry matching ``source``.

    An entry is a ``{source_prefix, expires_at, reason}`` dict. Matches
    when ``source`` starts with ``source_prefix``. "Live" requires an
    ``expires_at`` that parses as ISO8601 AND is still in the future —
    a missing, unparseable, or already-past ``expires_at`` does NOT
    suppress (fail toward alerting: a typo'd or omitted expiry must
    never become an accidental permanent mute).
    """
    if not source:
        return None
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        prefix = entry.get("source_prefix")
        expires_at = entry.get("expires_at")
        if not prefix or not expires_at:
            continue
        if not source.startswith(prefix):
            continue
        try:
            expires = datetime.fromisoformat(str(expires_at).replace("Z", "+00:00"))
        except ValueError:
            continue
        if expires <= now:
            continue
        return entry
    return None


def publish(
    message: str,
    *,
    severity: str = "error",
    source: str | None = None,
    sns: bool = True,
    telegram: bool = True,
    sns_topic_arn: str | None = None,
    dedup_key: str | None = None,
    dedup_window_min: int | None = DEFAULT_DEDUP_WINDOW_MIN,
    dedup_bucket: str | None = None,
    mute_ssm_param: str | None = None,
    dry_run: bool = False,
    state: str = ALERT_STATE_OPENED,
    identity_key: str | None = None,
    silent: bool | None = None,
    destination: str | None = None,
    console_artifact: str | None = None,
) -> PublishResult:
    """Fan out a failure alert to the operator-surveillance channels.

    Default: publish to both ``alpha-engine-alerts`` SNS (→ email) AND
    Telegram (``@nous_ergon_alerts_bot``). Pass ``sns=False`` /
    ``telegram=False`` to suppress individual channels (useful for
    tests, or for callers that have a narrower target).

    **Dedup** (v0.24.0). When ``dedup_key`` is provided, the call
    checks an S3 marker at
    ``s3://{dedup_bucket}/_alerts/_dedup/{sha1(dedup_key)[:16]}.json``.
    If the marker exists and the last publish for that key is within
    ``dedup_window_min`` minutes (default ``60``; ``None`` = forever),
    the publish is suppressed and :attr:`PublishResult.dedup_skipped`
    is True. After a successful fresh publish, the marker is written
    (or refreshed) with an incremented ``publish_count``. Use cases:

    - **One email per cost anomaly** even when ``evaluate.py`` runs
      multiple times for the same date — pass a deterministic
      ``dedup_key`` derived from the anomaly inputs.
    - **One alert per Lambda canary rollback episode** even when 8
      Lambda repos cascade-fail from one shared lib regression — pass
      ``dedup_key=f"canary-rollback-{lib_pin_sha}"`` so the cascading
      deploys all collapse to one operator email.
    - **Once-per-hour throttling** on noisy WARN paths — pass any
      stable key + leave the default 60min window.

    Dedup is best-effort: any S3 error during the check falls through
    to publish (better an extra alert than a silent drop). Marker
    write failure after a successful publish is logged but does NOT
    propagate (worst case is one duplicate next call within window).

    :param message: The alert body. Severity tag + source prefix are
        prepended automatically (e.g. ``"[ERROR] spot_backtest.sh: <body>"``).
    :param severity: Free-form severity string. Delivered to both channels
        at every severity — ``error`` / ``critical`` additionally trigger a
        Telegram phone push; everything else still posts to the chat, only
        without the buzz (NOT suppressed; see :data:`SEVERITY_PHONE_PUSH`).
        The tag is uppercased in the rendered message.
    :param source: Optional source identifier (script path, repo, Lambda
        name) inserted between the tag and the message body. Helps the
        operator triage at a glance.
    :param sns: When ``False``, skip the SNS publish entirely.
    :param telegram: When ``False``, skip the Telegram fan-out entirely.
    :param sns_topic_arn: Explicit topic ARN. Defaults to
        ``arn:aws:sns:{region}:{account_id}:alpha-engine-alerts`` resolved
        from env + STS.
    :param dedup_key: Opaque caller-chosen string. Same key + same
        window ⇒ at most one publish per window. ``None`` (default)
        disables dedup entirely; legacy callers behave unchanged.
    :param dedup_window_min: Window in minutes after which a fresh
        publish is allowed for the same ``dedup_key``. Default
        ``60``. Pass ``None`` for "forever" (publish once per
        ``dedup_key`` for the lifetime of the marker bucket).
    :param dedup_bucket: S3 bucket holding the markers. Defaults to
        ``alpha-engine-research`` (the shared corpus bucket).
    :param mute_ssm_param: Override the SSM parameter holding the
        source-mute list. Defaults to :data:`DEFAULT_MUTE_SSM_PARAM`
        (``/alpha-engine/alerts/source_mutes``), a JSON list of
        ``{source_prefix, expires_at, reason}`` objects maintained by an
        operator (e.g. via ``aws ssm put-parameter``). Checked BEFORE
        the dedup step: if ``source`` starts with any entry's
        ``source_prefix`` and that entry's ``expires_at`` (ISO8601) is
        still in the future, the alert is suppressed on both channels
        — logged at DEBUG (never silent), never raised to WARNING/ERROR
        so a live mute doesn't itself look like a problem. Missing,
        expired, or malformed entries never suppress (fail toward
        alerting). See :attr:`PublishResult.muted`.
    :param dry_run: When ``True``, short-circuits before the dedup
        check and before any boto3 client construction. Argument
        parsing, ``_format_message``, and (if ``sns`` and an explicit
        ``sns_topic_arn`` were given) topic-ARN echoing still run; no
        SNS publish, no Telegram call, no dedup marker write, and no
        Overseer intake event fire. Returns a :class:`PublishResult`
        with ``ok=True, detail="dry-run: would send"`` per attempted
        channel so callers verifying a delivery call site's shape exit
        ``0`` without paging the operator (config-I6759).
    :param state: Condition lifecycle for this emission — one of
        :data:`ALERT_STATES` (``opened`` / ``still_open`` / ``cleared``).
        Defaults to ``opened``, which is what every pre-I8105 call site
        means. Rides on the ``nousergon.alert.v1`` event so a consumer can
        pair a page to its terminator without string-matching prose. An
        unrecognised value raises :class:`ValueError` — a lifecycle field
        nobody validates is a field consumers cannot trust, and the
        fail-loud default applies (``~/Development/CLAUDE.md``).
    :param identity_key: The publisher's own stable identity for the
        CONDITION (not for this emission): a page and its later clear carry
        the same one. Correlation only — unlike ``dedup_key`` it never
        suppresses, which is what lets a clear reference a page whose dedup
        marker is still live. Defaults to ``dedup_key`` when that is set and
        this is not, so existing dedup-keyed publishers become pairable
        without touching their call sites.
    :param silent: Force silent Telegram delivery (delivered to the chat, no
        phone push) regardless of severity. ``None`` (default) keeps the
        severity-derived behaviour. ``False`` is NOT an escalation — it is
        treated as "no override".
    :param destination: Explicit override of the severity-derived Telegram
        destination — one of :data:`ALERT_DESTINATIONS`. ``None`` (default)
        derives it from ``severity``. An override is honoured where it can
        be, and is still subject to the fallback rule: asking for
        ``log_chat`` with no ``TELEGRAM_LOG_CHAT_ID`` configured, or for
        ``console_only`` with no ``console_artifact``, delivers to the
        operator chat and logs at WARNING rather than dropping the finding.
        An unrecognised value raises :class:`ValueError`.
    :param console_artifact: URI/identifier of the durable non-channel
        surface this finding is ALSO published to — a console artifact, a
        ``nousergon_lib.fleet_check_result`` envelope, an S3 key. Supplying
        it is what makes ``console_only`` legal: it is the evidence that the
        finding is readable somewhere the operator can find it without the
        chat. Do not pass it for a surface you did not actually write. It
        does not override an incident-tier severity — ``error``/``critical``
        still page.
    :param sns: SNS delivery is unaffected by all of the above. It is
        byte-identical at every severity and every destination; routing is a
        Telegram-only concept and the durable record never moves.
    :returns: :class:`PublishResult` — caller can inspect per-channel
        outcomes. :attr:`PublishResult.any_ok` is the typical success
        gate; :attr:`PublishResult.all_ok` is the strict variant.
        On dedup-skip, :attr:`PublishResult.dedup_skipped` is True and
        :attr:`PublishResult.dedup_reason` explains why.
    """
    if state not in ALERT_STATES:
        raise ValueError(
            f"alerts.publish: unknown state {state!r}; expected one of {ALERT_STATES}"
        )
    # An emission that carries a dedup_key already has a stable condition
    # identity; reuse it so pre-existing publishers pair for free. The reverse
    # is NEVER done — identity_key is not fed back into the dedup check, or a
    # clear would suppress itself against its own page's marker.
    effective_identity = identity_key or dedup_key

    result = PublishResult()
    result.state = state
    result.identity_key = effective_identity
    formatted = _format_message(message, severity, source, state)

    # ── Dry-run short-circuit (config-I6759) ─────────────────────────────
    # Fires before the dedup check and before any boto3 client construction
    # (SNS ARN resolution's ``sts.get_caller_identity`` call included) so a
    # caller verifying a delivery call site's argument shape never pages
    # the operator and never depends on AWS credentials being present.
    # Deliberately does NOT attempt live SNS topic-ARN resolution — only an
    # already-explicit ``sns_topic_arn`` is echoed — so this path never
    # imports boto3. No dedup marker write, no Overseer intake event
    # (full suppression; no ``dry_run`` field added to the event schema).
    if dry_run:
        detail = "dry-run: would send"
        if sns:
            sns_detail = f"{detail} to {sns_topic_arn}" if sns_topic_arn else detail
            result.sns = ChannelResult(ok=True, detail=sns_detail)
        else:
            result.sns = ChannelResult(ok=True, detail="dry-run: sns disabled (sns=False)")
        if telegram:
            result.telegram = ChannelResult(ok=True, detail=detail)
        else:
            result.telegram = ChannelResult(ok=True, detail="dry-run: telegram disabled (telegram=False)")
        return result

    # ── Test-environment guard (defense-in-depth) ────────────────────────
    # NEVER fan out a real SNS / Telegram alert from inside a test process.
    # pytest sets ``PYTEST_CURRENT_TEST`` for the duration of each test; when
    # it is present we short-circuit to a no-op result so any consumer test
    # that exercises a ``publish`` call site without stubbing it cannot page
    # the operator for real. This is the cross-repo chokepoint — one guard
    # protects all 8 suites; consumer repos SHOULD also stub ``publish`` in
    # their own conftest, but this catches the case where they forget (which
    # is exactly how the optimizer turnover-governor large-move WARN leaked
    # from alpha-engine's suite on 2026-06-07). Escape hatch:
    # ``ALPHA_ENGINE_ALLOW_TEST_ALERTS=1`` re-enables the real path — used
    # ONLY by this lib's own ``test_alerts`` suite, which deliberately
    # exercises the fan-out logic against mocked boto3 / Telegram transports.
    if os.environ.get("PYTEST_CURRENT_TEST") and not os.environ.get(
        "ALPHA_ENGINE_ALLOW_TEST_ALERTS"
    ):
        detail = "suppressed in test env (PYTEST_CURRENT_TEST set)"
        result.sns = ChannelResult(ok=False, detail=detail)
        result.telegram = ChannelResult(ok=False, detail=detail)
        return result

    # ── Source-mute check (pre-dedup) ────────────────────────────────────
    # Deliberately runs BEFORE the dedup step: a mute is a coarser,
    # operator-declared "stop paging about this source" that must hold
    # regardless of how the message text varies (unlike dedup, which keys
    # on message content and resets whenever a new finding appears).
    mute_entries = _fetch_source_mutes(mute_ssm_param or DEFAULT_MUTE_SSM_PARAM)
    live_mute = _find_live_mute(source, mute_entries)
    if live_mute is not None:
        reason = (
            f"source={source!r} matches muted prefix "
            f"{live_mute.get('source_prefix')!r} "
            f"(expires {live_mute.get('expires_at')}, "
            f"reason: {live_mute.get('reason', '')!r})"
        )
        logger.debug("alerts.publish: suppressed alert — %s", reason)
        result.muted = True
        result.mute_reason = reason
        result.sns = ChannelResult(ok=False, detail="suppressed by source mute")
        result.telegram = ChannelResult(ok=False, detail="suppressed by source mute")
        return result

    # ── Dedup check (pre-publish) ────────────────────────────────────────
    marker_key: str | None = None
    bucket = dedup_bucket or DEFAULT_DEDUP_BUCKET
    if dedup_key:
        marker_key = _dedup_marker_key(dedup_key)
        within_window, reason = _check_dedup_marker(
            bucket, marker_key, dedup_window_min=dedup_window_min,
        )
        if within_window:
            result.dedup_skipped = True
            result.dedup_reason = reason
            result.sns = ChannelResult(ok=False, detail="suppressed by dedup")
            result.telegram = ChannelResult(ok=False, detail="suppressed by dedup")
            logger.info(
                "alerts.publish: skipped publish for dedup_key=%r (%s)",
                dedup_key, reason,
            )
            return result

    # ── Publish ──────────────────────────────────────────────────────────
    if sns:
        arn = _resolve_sns_topic_arn(sns_topic_arn)
        if arn is None:
            result.sns = ChannelResult(ok=False, detail="topic ARN resolution failed")
        else:
            # SNS subject — concise header, falls back to severity tag.
            subject = f"Alpha Engine alert [{severity.upper()}]"
            if source:
                subject += f" — {source}"
            result.sns = _publish_sns(arn, formatted, subject=subject)

    if telegram:
        # ── Destination resolution (alpha-engine-config-I7857) ───────────
        # Runs here rather than at the top of `publish` so the dry-run
        # short-circuit and the test-env guard keep their promise of
        # constructing no client and reading no secret. Every path out of
        # `resolve_destination` names a destination that DELIVERS: the
        # unconfigured case is the operator chat plus a WARNING, never a
        # drop.
        log_chat_id, log_thread_id = (None, None)
        if destination != DESTINATION_OPERATOR_CHAT:
            # Skip the secret read entirely when the caller (or the severity,
            # checked below) can only mean the operator chat — an incident
            # must not wait on a secrets round-trip for a destination it was
            # never going to use.
            if destination is not None or severity.lower() not in SEVERITY_PHONE_PUSH:
                log_chat_id, log_thread_id = _resolve_log_chat()
        resolved, reason = resolve_destination(
            severity,
            destination=destination,
            console_artifact=console_artifact,
            log_chat_id=log_chat_id,
        )
        result.telegram_destination = resolved
        result.destination_reason = reason

        if resolved == DESTINATION_CONSOLE_ONLY:
            # Delivered, not dropped: the caller named the durable surface it
            # is published to, so `ok=True` — a Bash caller's `|| echo
            # 'alert failed'` fallback must not fire for a finding that IS
            # readable, and the SNS leg above already wrote the durable
            # record regardless.
            result.telegram = ChannelResult(
                ok=True,
                detail=(
                    f"not sent (destination={DESTINATION_CONSOLE_ONLY}); "
                    f"published to {console_artifact}"
                ),
            )
        else:
            # Suppress send_message's auto-emit hook — this publish call emits
            # one rich event itself below; without the guard every publish
            # would land twice on the Overseer intake bus.
            with fleet_events.suppress_emission():
                result.telegram = _publish_telegram(
                    formatted,
                    severity=severity,
                    silent=silent,
                    destination=resolved,
                    chat_id=log_chat_id if resolved == DESTINATION_LOG_CHAT else None,
                    message_thread_id=(
                        log_thread_id if resolved == DESTINATION_LOG_CHAT else None
                    ),
                )

    # ── Dedup marker write (post-publish, only if any channel succeeded) ─
    if marker_key and (result.sns.ok or result.telegram.ok):
        _write_dedup_marker(
            bucket, marker_key,
            dedup_key=dedup_key, formatted_message=formatted,
        )

    # ── Overseer intake event (side-channel; best-effort, never raises) ──
    # Emitted only when channels were actually attempted: the test-env
    # guard and dedup-skip paths return earlier, so suppressed repeats and
    # test runs never reach the bus.
    fleet_events.emit_alert_event(
        origin="alerts.publish",
        body=message,
        severity_raw=severity,
        source=source,
        dedup_key=dedup_key,
        channels={
            "sns": result.sns.ok if sns else None,
            "telegram": result.telegram.ok if telegram else None,
        },
        state=state,
        identity_key=effective_identity,
    )

    return result


def publish_clear(
    message: str,
    *,
    identity_key: str,
    source: str | None = None,
    sns: bool = True,
    telegram: bool = True,
    sns_topic_arn: str | None = None,
    mute_ssm_param: str | None = None,
    dry_run: bool = False,
    destination: str | None = None,
    console_artifact: str | None = None,
) -> PublishResult:
    """Publish the terminator for a condition previously alerted on.

    The other half of the open/clear pair (alpha-engine-config-I8105). A
    publisher that keeps a set of currently-true conditions calls this for
    each condition that has left the set, passing the SAME ``identity_key``
    the page carried.

    Three things are fixed here rather than left to the caller, because each
    of them was got wrong at least once by a call site that tried:

    - **``severity="info"``.** A recovery is not an incident. It is still
      delivered on both channels — severity has never been a delivery gate
      here (alpha-engine-config-I7857).
    - **``silent=True``.** Explicitly, not by relying on ``info`` sitting
      outside :data:`SEVERITY_PHONE_PUSH`. A clear that buzzes a phone is a
      second alert.
    - **no ``dedup_key``.** The identity travels as ``identity_key``, which
      correlates without suppressing. Passing the page's dedup key here would
      let the page's own live marker swallow its terminator.

    **Routing is inherited, not special-cased.** A clear is published at
    ``info``, so it resolves to the same non-operator destination any other
    non-incident severity does — the log chat when one is configured, the
    console when the caller names an artifact, and the operator chat (with a
    WARNING) when neither exists. ``destination`` / ``console_artifact``
    pass straight through to :func:`publish`. A recovery landing in the
    incident channel was the most-reported instance of the noise this tier
    exists to remove, and it is fixed by the general rule rather than by a
    branch that only clears take.

    :param message: What cleared, in the publisher's own words. The
        ``RESOLVED — `` marker is prepended by :func:`_format_message`.
    :param identity_key: The originating page's condition identity. Required
        — a clear nobody can pair to a page is prose, which is the thing this
        primitive exists to stop emitting.
    :returns: :class:`PublishResult`, same contract as :func:`publish`.
    """
    if not identity_key:
        raise ValueError(
            "alerts.publish_clear: identity_key is required — an unpairable "
            "clear is prose, not a terminator (alpha-engine-config-I8105)"
        )
    return publish(
        message,
        severity=CLEAR_SEVERITY,
        source=source,
        sns=sns,
        telegram=telegram,
        sns_topic_arn=sns_topic_arn,
        dedup_key=None,
        mute_ssm_param=mute_ssm_param,
        dry_run=dry_run,
        state=ALERT_STATE_CLEARED,
        identity_key=identity_key,
        silent=True,
        destination=destination,
        console_artifact=console_artifact,
    )


def diff_conditions(
    previous: "Iterable[str]", current: "Iterable[str]"
) -> tuple[list[str], list[str], list[str]]:
    """Classify two observations of a condition set into the three states.

    The set arithmetic behind the open/clear pair, in one place so every
    publisher that grows a lifecycle computes it identically rather than
    re-deriving the direction of the difference (the ``cleared`` set is the
    one call sites get backwards, because the alerting path only ever needed
    the other direction).

    :param previous: Conditions true at the previous observation.
    :param current: Conditions true now.
    :returns: ``(opened, still_open, cleared)``, each sorted for a stable
        emission order — an unordered emission makes two identical ticks
        produce different journals and defeats diffing them.
    """
    prev_set = set(previous)
    cur_set = set(current)
    return (
        sorted(cur_set - prev_set),
        sorted(cur_set & prev_set),
        sorted(prev_set - cur_set),
    )


# ─── CLI entry ──────────────────────────────────────────────────────────────
# Designed for Bash callers that need failure surveillance from a script
# (spot dispatcher `cleanup` traps, deploy.sh rollback branches, etc.).
# Mirrors the the transparency CLI ``python -m`` pattern so
# Bash callers reach this primitive without bootstrapping a full Python
# project. Exit code is 0 if *any* channel succeeded, 1 if both failed.


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m krepis.alerts",
        description=(
            "Publish a failure alert to alpha-engine's operator-surveillance "
            "channels (SNS topic alpha-engine-alerts + Telegram). Designed "
            "for Bash callers — exit code 0 if any channel succeeded, 1 if "
            "both failed. Never raises."
        ),
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    pub = subparsers.add_parser("publish", help="Publish an alert message.")
    pub.add_argument("--message", required=True, help="Alert body text.")
    pub.add_argument(
        "--severity",
        default="error",
        help=(
            "Severity tag (default: error). Delivered at every severity — "
            "'error' and 'critical' additionally trigger a Telegram phone "
            "push; all others still post to the channel, just without the "
            "buzz (never suppressed)."
        ),
    )
    pub.add_argument(
        "--source",
        default=None,
        help=(
            "Optional source identifier (script path, repo, Lambda name) "
            "rendered between the severity tag and the message body."
        ),
    )
    pub.add_argument("--no-sns", action="store_true", help="Skip SNS publish.")
    pub.add_argument("--no-telegram", action="store_true", help="Skip Telegram fan-out.")
    pub.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Verify the delivery call site's argument shape without "
            "sending anything: runs arg parsing + message formatting, "
            "attempts no SNS publish, no Telegram call, writes no dedup "
            "marker, and never constructs a boto3 client. Exits 0. "
            "Use this to smoke-test a call site's flags instead of "
            "issuing a real (or synthetic-ERROR) alert (config-I6759)."
        ),
    )
    pub.add_argument(
        "--sns-topic-arn",
        default=None,
        help=(
            "Override the SNS topic ARN. Defaults to "
            "arn:aws:sns:{region}:{account_id}:alpha-engine-alerts."
        ),
    )
    pub.add_argument(
        "--dedup-key",
        default=None,
        help=(
            "Optional opaque dedup key. When set, ``publish`` checks an "
            "S3 marker first and suppresses the alert if an earlier "
            "publish for the same key is within --dedup-window-min. "
            "Use for cost anomalies / canary rollback episodes / any "
            "noisy WARN path that benefits from rate-limiting. Bash "
            "callers typically pass a bucketed timestamp, e.g. "
            "--dedup-key \"canary-rollback-$(date -u +%%Y%%m%%d%%H)\"."
        ),
    )
    pub.add_argument(
        "--dedup-window-min",
        type=int,
        default=DEFAULT_DEDUP_WINDOW_MIN,
        help=(
            f"Window in minutes after which a fresh publish is allowed for "
            f"the same --dedup-key (default: {DEFAULT_DEDUP_WINDOW_MIN}). "
            "Pass 0 for 'forever' (publish once per --dedup-key for the "
            "lifetime of the marker bucket)."
        ),
    )
    pub.add_argument(
        "--dedup-bucket",
        default=None,
        help=(
            f"S3 bucket holding the dedup markers. Defaults to "
            f"{DEFAULT_DEDUP_BUCKET!r}."
        ),
    )
    pub.add_argument(
        "--state",
        default=ALERT_STATE_OPENED,
        choices=list(ALERT_STATES),
        help=(
            "Condition lifecycle for this emission (default: opened). "
            "Rides on the nousergon.alert.v1 event so a consumer can pair a "
            "page to its terminator. Prefer the `clear` subcommand over "
            "`publish --state cleared`: it also forces info severity and "
            "silent delivery."
        ),
    )
    pub.add_argument(
        "--identity-key",
        default=None,
        help=(
            "Stable identity of the CONDITION (not this emission), carried "
            "by both the page and its later clear. Correlation only — never "
            "suppression. Defaults to --dedup-key when that is given."
        ),
    )
    pub.add_argument(
        "--destination",
        default=None,
        choices=list(ALERT_DESTINATIONS),
        help=(
            "Override the severity-derived Telegram destination: "
            "operator_chat (the incident channel), log_chat "
            "(TELEGRAM_LOG_CHAT_ID), or console_only (no Telegram send, "
            "legal only with --console-artifact). Never drops the finding: "
            "an unconfigured log chat or a console_only with no artifact "
            "falls back to the operator chat and logs a WARNING."
        ),
    )
    pub.add_argument(
        "--console-artifact",
        default=None,
        help=(
            "URI of the durable surface this finding is ALSO published to "
            "(console artifact, fleet_check_result envelope, S3 key). This "
            "is the evidence that makes console_only delivery safe — pass it "
            "only for a surface you actually wrote."
        ),
    )

    clr = subparsers.add_parser(
        "clear",
        help=(
            "Publish the terminator for a condition previously alerted on: "
            "info severity, silent delivery (no phone push), state=cleared, "
            "no dedup. Bash callers that track a condition set call this for "
            "each condition that has LEFT the set."
        ),
    )
    clr.add_argument(
        "--message", required=True,
        help="What cleared, in the publisher's own words. 'RESOLVED — ' is prepended.",
    )
    clr.add_argument(
        "--identity-key", required=True,
        help=(
            "The originating page's condition identity. Required: an "
            "unpairable clear is prose, not a terminator."
        ),
    )
    clr.add_argument(
        "--source", default=None,
        help="Source identifier, matching the page's --source.",
    )
    clr.add_argument("--no-sns", action="store_true", help="Skip SNS publish.")
    clr.add_argument("--no-telegram", action="store_true", help="Skip Telegram fan-out.")
    clr.add_argument(
        "--sns-topic-arn", default=None, help="Override the SNS topic ARN.",
    )
    clr.add_argument(
        "--dry-run", action="store_true",
        help="Verify the call site's argument shape without sending anything.",
    )
    clr.add_argument(
        "--destination",
        default=None,
        choices=list(ALERT_DESTINATIONS),
        help=(
            "Override the severity-derived Telegram destination: "
            "operator_chat (the incident channel), log_chat "
            "(TELEGRAM_LOG_CHAT_ID), or console_only (no Telegram send, "
            "legal only with --console-artifact). Never drops the finding: "
            "an unconfigured log chat or a console_only with no artifact "
            "falls back to the operator chat and logs a WARNING."
        ),
    )
    clr.add_argument(
        "--console-artifact",
        default=None,
        help=(
            "URI of the durable surface this finding is ALSO published to "
            "(console artifact, fleet_check_result envelope, S3 key). This "
            "is the evidence that makes console_only delivery safe — pass it "
            "only for a surface you actually wrote."
        ),
    )

    args = parser.parse_args(argv)

    if args.cmd == "clear":
        logging.basicConfig(level=logging.WARNING)
        clear_result = publish_clear(
            args.message,
            identity_key=args.identity_key,
            source=args.source,
            sns=not args.no_sns,
            telegram=not args.no_telegram,
            sns_topic_arn=args.sns_topic_arn,
            dry_run=args.dry_run,
            destination=args.destination,
            console_artifact=args.console_artifact,
        )
        print(
            f"alerts.clear: identity_key={args.identity_key!r} "
            f"sns.ok={clear_result.sns.ok} ({clear_result.sns.detail}); "
            f"telegram.ok={clear_result.telegram.ok} ({clear_result.telegram.detail})",
            file=sys.stderr,
        )
        return 0 if clear_result.any_ok else 1

    logging.basicConfig(level=logging.WARNING)

    # CLI convention: --dedup-window-min 0 = forever; map to None for the
    # Python API (whose default is 60 + None=forever).
    window_min: int | None
    if args.dedup_window_min == 0:
        window_min = None
    else:
        window_min = args.dedup_window_min

    result = publish(
        args.message,
        severity=args.severity,
        source=args.source,
        sns=not args.no_sns,
        telegram=not args.no_telegram,
        sns_topic_arn=args.sns_topic_arn,
        dedup_key=args.dedup_key,
        dedup_window_min=window_min,
        dedup_bucket=args.dedup_bucket,
        dry_run=args.dry_run,
        state=args.state,
        identity_key=args.identity_key,
        destination=args.destination,
        console_artifact=args.console_artifact,
    )

    # One-line status to stderr (stdout reserved for structured output if
    # any caller starts parsing it). Bash callers can ignore.
    if result.muted:
        print(
            f"alerts.publish: muted=True ({result.mute_reason})",
            file=sys.stderr,
        )
    elif result.dedup_skipped:
        print(
            f"alerts.publish: dedup_skipped=True ({result.dedup_reason})",
            file=sys.stderr,
        )
    else:
        print(
            f"alerts.publish: sns.ok={result.sns.ok} ({result.sns.detail}); "
            f"telegram.ok={result.telegram.ok} ({result.telegram.detail}); "
            f"destination={result.telegram_destination}",
            file=sys.stderr,
        )

    return 0 if result.any_ok else 1


if __name__ == "__main__":
    sys.exit(main())
