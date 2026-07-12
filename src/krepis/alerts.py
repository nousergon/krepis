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
- CLI: ``python -m krepis.alerts publish --message "..."
  --severity error --source "..."``. Designed for Bash failure-trap
  callers (``cleanup()`` in spot dispatchers, ``deploy.sh`` rollback
  branches). Exit code is ``0`` if *either* channel succeeded, ``1`` if
  *both* failed.

**Severity tiering.** ``severity`` is a free-form string that is
prepended to the message (``[ERROR] ...`` / ``[WARNING] ...``) for both
channels. Telegram pushes (``disable_notification=False``) for
``error``/``critical``; in-channel silent for ``info``/``warning``. SNS
delivery is identical regardless of severity — downstream subscribers
choose how to fan out.

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
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Final

from krepis import _dedup

logger = logging.getLogger(__name__)

DEFAULT_SNS_TOPIC_NAME: Final[str] = "alpha-engine-alerts"
DEFAULT_REGION: Final[str] = "us-east-1"
SEVERITY_PUSH: Final[frozenset[str]] = frozenset({"error", "critical"})

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
    """

    sns: ChannelResult = field(default_factory=lambda: ChannelResult(ok=False, detail="not attempted"))
    telegram: ChannelResult = field(default_factory=lambda: ChannelResult(ok=False, detail="not attempted"))
    dedup_skipped: bool = False
    dedup_reason: str = ""

    @property
    def any_ok(self) -> bool:
        if self.dedup_skipped:
            return True
        return self.sns.ok or self.telegram.ok

    @property
    def all_ok(self) -> bool:
        if self.dedup_skipped:
            return True
        return self.sns.ok and self.telegram.ok


def _resolve_sns_topic_arn(explicit: str | None) -> str | None:
    """Return the SNS topic ARN, resolving from env + STS if not explicit."""
    if explicit:
        return explicit
    region = (
        os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or DEFAULT_REGION
    )
    try:
        import boto3

        account_id = boto3.client("sts", region_name=region).get_caller_identity()["Account"]
    except Exception as exc:  # boto3 missing, STS unreachable, creds bad
        logger.warning("alerts.publish: SNS topic ARN resolution failed: %s", exc)
        return None
    return f"arn:aws:sns:{region}:{account_id}:{DEFAULT_SNS_TOPIC_NAME}"


def _format_message(message: str, severity: str, source: str | None) -> str:
    """Prepend severity tag + source prefix to the message body."""
    tag = f"[{severity.upper()}]"
    if source:
        return f"{tag} {source}: {message}"
    return f"{tag} {message}"


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


def _publish_telegram(message: str, severity: str) -> ChannelResult:
    try:
        from krepis.telegram import send_message

        # Push for error/critical, silent in-channel for info/warning.
        silent = severity.lower() not in SEVERITY_PUSH
        ok = send_message(message, disable_notification=silent)
        return ChannelResult(ok=bool(ok), detail="sent" if ok else "send_message returned False")
    except Exception as exc:  # send_message itself never raises, but defensive
        logger.warning("alerts.publish: Telegram fan-out failed: %s", exc)
        return ChannelResult(ok=False, detail=f"telegram error: {exc!r}")


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
    :param severity: Free-form severity string (``error`` / ``critical``
        push on Telegram; everything else is silent in-channel). The tag
        is uppercased in the rendered message.
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
    :returns: :class:`PublishResult` — caller can inspect per-channel
        outcomes. :attr:`PublishResult.any_ok` is the typical success
        gate; :attr:`PublishResult.all_ok` is the strict variant.
        On dedup-skip, :attr:`PublishResult.dedup_skipped` is True and
        :attr:`PublishResult.dedup_reason` explains why.
    """
    result = PublishResult()
    formatted = _format_message(message, severity, source)

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
        result.telegram = _publish_telegram(formatted, severity=severity)

    # ── Dedup marker write (post-publish, only if any channel succeeded) ─
    if marker_key and (result.sns.ok or result.telegram.ok):
        _write_dedup_marker(
            bucket, marker_key,
            dedup_key=dedup_key, formatted_message=formatted,
        )

    return result


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
            "Severity tag (default: error). 'error' and 'critical' push on "
            "Telegram; all others are silent in-channel."
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

    args = parser.parse_args(argv)

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
    )

    # One-line status to stderr (stdout reserved for structured output if
    # any caller starts parsing it). Bash callers can ignore.
    if result.dedup_skipped:
        print(
            f"alerts.publish: dedup_skipped=True ({result.dedup_reason})",
            file=sys.stderr,
        )
    else:
        print(
            f"alerts.publish: sns.ok={result.sns.ok} ({result.sns.detail}); "
            f"telegram.ok={result.telegram.ok} ({result.telegram.detail})",
            file=sys.stderr,
        )

    return 0 if result.any_ok else 1


if __name__ == "__main__":
    sys.exit(main())
