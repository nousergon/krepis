"""Per-stage output assertion — the ONE implementation, called from every stage.

**What this asserts.** A Step Functions run establishes that a stage did not
THROW. Nothing in it establishes that the stage WROTE. Those are different
claims and only the second is what the trading week depends on: on 2026-08-08
five weekly stages produced no output while the run terminated ``SUCCEEDED``,
and every one of them exited zero (``alpha-engine-config-I7167``).

**Where the assertion belongs — and why it is here rather than at the end of
the run.** `sf-pipeline-policy.md` §2.1 (atomicity: one stage, one script, one
failure route) puts the assertion at the boundary where the fact becomes
knowable — the moment the stage's own launcher or handler is about to exit.
Two prior designs placed it at the pipeline's convergence point instead, as a
single end-of-run sweep. Both were ruled non-SOTA on the same two axes
(Brian, 2026-08-13):

1. **Detection latency.** The weekly pipeline runs ~4h. A sweep at the
   convergence point learns that ``MorningEnrich`` wrote nothing roughly three
   hours after the fact — after every downstream stage has already consumed
   the absence.
2. **Attribution.** A sweep cannot attribute a miss to a stage that is still
   in flight, and it must reconstruct the entered-stage set from execution
   history to avoid crying wolf on gated-out runs. The stage itself needs
   neither: it knows it ran, and it knows what it declared.

The corollary that makes this shippable: asserting inside the launcher costs
**no topology change** — no new SF state, no new Lambda, no new IAM role, no
operator-gated bootstrap. Launcher scripts already run under an instance
profile holding ``s3:GetObject``/``s3:ListBucket`` on the research bucket, and
every Lambda-backed stage already writes to it.

**One implementation, two front doors.** Per `policy-shared-code`, this is at
least the third adoption of the same logic, which is the lift trigger:

- **Bash launchers** (``spot_*.sh`` in nousergon-data, crucible-backtester,
  crucible-predictor) call the CLI::

      "$LIB_PYTHON" -m krepis.stage_coverage assert --stage MorningEnrich

- **Lambda handlers** call :func:`assert_stage_coverage` directly, immediately
  before returning their payload, and merge the returned verdict into it.

Neither carries a copy of the rules, a copy of the stage list, or a copy of
the registry. A per-repo copy is the fork `policy-shared-code` forbids.

**The declaration is data, and it is NOT redefined here.** Stage → artifact
comes from ``ARTIFACT_REGISTRY.yaml``'s ``pipeline_stages:`` section, read
from the S3 mirror ``alpha-engine-config``'s ``sync-artifact-registry.yml``
refreshes on every merge that touches it — so the consumer is refreshed by the
same event that changes the declaration. Nothing here carries its own list of
stages: a hand-maintained denominator drifts, and its drift is invisible
because the missing rows produce no signal.

**Observe mode is the default and the assertion cannot fail the run.** Every
error path returns a verdict; no path raises to the caller. ``--enforce``
flips one boolean, and only a real ``MISSING`` (not ``STALE``, not
``UNMEASURED``) is allowed to signal failure — enforcing a threshold whose
distribution nobody has seen is how a correct detector gets switched off in
week one (Brian ruling 2026-08-11,
``ruling_detect_before_enforcing_when_the_floor_is_unmeasured``).

**Could-not-measure is never a pass, and never a finding either.** An
unreadable registry, a non-404 probe failure, an unresolvable cycle date, or a
stage absent from the registry are all :data:`STATUS_UNMEASURED` — never
:data:`STATUS_COVERED` (`sf-pipeline-policy.md` §2.3a rule 2: a missing verdict
propagates as UNKNOWN, never as pass) and never :data:`STATUS_MISSING` (a
detector that reports its own harness fault AS the defect is this fleet's
most-repeated detector bug, and it always errs in the alarming direction).

**A stage declaring ``output: none`` is COVERED by having said so.** That is
the entire point of the registry's stage section: it is what distinguishes a
stage that writes nothing from a stage nobody ever considered. Eleven of the
43 weekly stages are in this class; they assert nothing and still record that
they declared nothing.

**The cycle date is not the run date.** ``{date}`` / ``{trading_day}`` in a key
template resolve to the *cycle* tick — the last closed trading day — not to
the execution's ``run_date``. Measured 2026-08-13: the 08-08 execution carries
``run_date=2026-08-08`` while every artifact it produced landed under
``2026-08-07``; resolving with ``run_date`` produced 28 confident and entirely
false ``missing`` verdicts. :func:`resolve_cycle_date` defers to
:func:`krepis.trading_calendar.last_closed_trading_day` — the same notion the
fleet's artifact-freshness probe substitutes — and returns ``None`` rather
than guessing when it cannot resolve.

**Public surface:**

- :data:`STATUS_COVERED` / :data:`STATUS_COVERED_NO_OUTPUT` /
  :data:`STATUS_MISSING` / :data:`STATUS_STALE` / :data:`STATUS_UNMEASURED`
- :class:`StageVerdict` — the record written to S3 and returned to callers.
- :func:`assert_stage_coverage` — the Lambda-handler front door; never raises.
- :func:`evaluate_stage` — the pure ``(registry, stage, …) → StageVerdict``
  core, injected clients only, no side effects.
- :func:`record_verdict` — the S3 + CloudWatch recording surface.
- :func:`main` — the ``python -m krepis.stage_coverage`` CLI.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Final

from krepis.trading_calendar import last_closed_trading_day

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

#: Bucket holding both the registry mirror and the verdict records. A default,
#: not a constant: krepis is the fleet's provider-neutral toolkit, so the
#: registry's LOCATION is configuration (``STAGE_COVERAGE_BUCKET``) and only
#: the *shape* of a registry row is knowledge this module carries.
DEFAULT_BUCKET: Final[str] = os.environ.get(
    "STAGE_COVERAGE_BUCKET", "alpha-engine-research"
)

#: The registry mirror ``alpha-engine-config``'s ``sync-artifact-registry.yml``
#: refreshes on every merge touching ``private-docs/ARTIFACT_REGISTRY.yaml``.
REGISTRY_KEY: Final[str] = os.environ.get(
    "STAGE_COVERAGE_REGISTRY_KEY", "_freshness_monitor/ARTIFACT_REGISTRY.yaml"
)

#: Prefix under which per-stage verdicts land, one object per stage per run.
#: Per-stage rather than per-run because the writers are concurrent: the
#: weekly pipeline's Parallel branches assert simultaneously, and a single
#: per-run object would be a lost-update race between them.
VERDICT_PREFIX: Final[str] = "_stage_coverage"

#: CloudWatch namespace/metric the verdict is published on. Value is 1 for a
#: covered stage and 0 otherwise, so an alarm on the minimum catches a miss
#: and the *absence* of the datapoint is itself visible as a gap — `no data`
#: is never rendered as green (`principles.md` §2.7).
METRIC_NAMESPACE: Final[str] = "AlphaEngine"
METRIC_NAME: Final[str] = "StageCoverage"

#: The stage wrote every artifact it declared, inside this run's window.
STATUS_COVERED: Final[str] = "COVERED"
#: The stage positively declared it produces no durable artifact.
STATUS_COVERED_NO_OUTPUT: Final[str] = "COVERED_NO_OUTPUT"
#: At least one declared artifact does not exist. The real finding.
STATUS_MISSING: Final[str] = "MISSING"
#: Every declared artifact exists, but at least one predates this run's
#: window — a leftover from a previous cycle satisfying an existence-only
#: probe while the consumer reads last week's belief.
STATUS_STALE: Final[str] = "STALE"
#: The assertion could not establish an answer. Never a pass, never a finding.
STATUS_UNMEASURED: Final[str] = "UNMEASURED"

#: Process exit code used by the CLI when ``--enforce`` is set and the verdict
#: is a real MISSING. Distinct from 1 so a shell cannot confuse a coverage
#: finding with an interpreter error or an unhandled traceback.
EXIT_COVERAGE_FAILURE: Final[int] = 3


# ── Verdict record ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StageVerdict:
    """The outcome of one stage's own output assertion.

    Attributes:
        stage: SF Task-state name the assertion was made for.
        status: One of the five ``STATUS_*`` constants.
        stage_class: ``product`` / ``infrastructure`` from the registry,
            or ``""`` when the stage could not be resolved.
        declared_output: The registry row's ``output`` value
            (``registered`` / ``none`` / ``variable_cardinality``).
        expected: Artifact ids the stage declares it writes.
        missing: Declared artifacts whose key does not exist.
        stale: Declared artifacts present but older than ``window_start``.
        covered: Declared artifacts present and inside the window.
        unmeasured: ``[(artifact_id, reason)]`` — probes that could not be
            evaluated. Kept apart from ``missing`` deliberately: a harness
            fault reported as a producer finding is always wrong in the
            alarming direction.
        reason: Human-readable explanation, always populated for
            ``UNMEASURED``.
        run_date: The execution's ``run_date``, as supplied.
        cycle_date: The date substituted into ``{date}`` / ``{trading_day}``.
        window_start: ISO-8601 UTC instant before which an artifact counts
            as ``stale``. ``None`` disables the staleness half — absence is
            still measurable without a window, so ``MISSING`` still fires.
        recorded_at: ISO-8601 UTC instant the verdict was produced.
        enforce: Whether the caller asked for enforcement.
    """

    stage: str
    status: str
    stage_class: str = ""
    declared_output: str = ""
    expected: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)
    covered: list[str] = field(default_factory=list)
    unmeasured: list[tuple[str, str]] = field(default_factory=list)
    reason: str = ""
    run_date: str = ""
    cycle_date: str = ""
    window_start: str | None = None
    recorded_at: str = ""
    enforce: bool = False

    @property
    def is_finding(self) -> bool:
        """True only for a real MISSING — the one status enforcement acts on.

        ``STALE`` is deliberately excluded: staleness is a genuine defect but
        its distribution has never been measured on a live run, and the first
        enforcement of an unmeasured floor is how a correct detector gets
        switched off in week one. ``UNMEASURED`` is excluded because it is not
        a claim about the producer at all.
        """
        return self.status == STATUS_MISSING

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form (tuples flattened to lists)."""
        raw = asdict(self)
        raw["unmeasured"] = [list(pair) for pair in self.unmeasured]
        raw["is_finding"] = self.is_finding
        return raw


# ── Registry access ──────────────────────────────────────────────────────────


def load_registry(
    s3_client: Any,
    *,
    bucket: str = DEFAULT_BUCKET,
    key: str = REGISTRY_KEY,
    local_path: str | None = None,
) -> dict[str, Any]:
    """Return the parsed registry.

    ``local_path`` reads from disk instead of S3 — used by tests and by any
    caller that already has the file. Raises on failure; callers convert that
    to :data:`STATUS_UNMEASURED` rather than letting it escape.
    """
    import yaml  # imported lazily: not every consumer of this module needs it

    if local_path:
        with open(local_path, encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    body = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()
    return yaml.safe_load(body) or {}


def resolve_stage_row(registry: dict[str, Any], stage: str) -> dict[str, Any] | None:
    """Return the ``pipeline_stages:`` row for ``stage``, or ``None``.

    ``None`` is an UNMEASURED condition, never a pass: a stage absent from the
    registry has not declared anything, and treating silence as "produces
    nothing" is exactly the ambiguity the stage section exists to remove.
    """
    for row in registry.get("pipeline_stages") or []:
        if row.get("stage") == stage:
            return row
    return None


def index_artifacts(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return ``{artifact_id: row}`` for the registry's ``artifacts:`` list."""
    return {
        row["artifact_id"]: row
        for row in registry.get("artifacts") or []
        if row.get("artifact_id")
    }


# ── Cycle-date resolution ────────────────────────────────────────────────────


def resolve_cycle_date(now: datetime) -> date | None:
    """Return the cycle tick ``{date}`` / ``{trading_day}`` resolve to.

    This is the **last closed trading day**, not the execution's ``run_date``.
    Returns ``None`` when the calendar is unavailable — the caller then
    reports ``UNMEASURED`` instead of probing a key it guessed.
    """
    try:
        resolved = last_closed_trading_day(now)
    except Exception:  # noqa: BLE001 — any calendar failure is UNMEASURED
        logger.warning("stage_coverage: trading calendar unavailable", exc_info=True)
        return None
    if isinstance(resolved, datetime):
        return resolved.date()
    if isinstance(resolved, date):
        return resolved
    return None


def format_key(template: str, cycle_date: date) -> str:
    """Substitute cycle placeholders into an artifact key template.

    Mirrors the fleet artifact-freshness probe's supported placeholders. An
    **unknown** placeholder is left intact rather than
    guessed, so the probe fails loudly on a key nobody can resolve instead of
    silently probing a wrong one.
    """
    iso = cycle_date.isoformat()
    try:
        return template.format(
            date=iso,
            trading_day=iso,
            cycle_label=iso,
        )
    except (KeyError, IndexError, ValueError):
        return template


# ── The pure core ────────────────────────────────────────────────────────────


def _head(s3_client: Any, bucket: str, key: str) -> tuple[str, Any]:
    """Probe one key. Returns ``(outcome, payload)``.

    ``outcome`` is ``"present"`` (payload = ``LastModified`` datetime),
    ``"absent"`` (payload = ``""``) or ``"probe_failed"`` (payload = reason).
    """
    try:
        head = s3_client.head_object(Bucket=bucket, Key=key)
        return "present", head.get("LastModified")
    except Exception as exc:  # noqa: BLE001 — classified below, never re-raised
        code = ""
        status = 0
        response = getattr(exc, "response", None)
        if isinstance(response, dict):
            code = str((response.get("Error") or {}).get("Code", ""))
            status = int(
                (response.get("ResponseMetadata") or {}).get("HTTPStatusCode", 0) or 0
            )
        if code in {"404", "NoSuchKey", "NotFound"} or status == 404:
            return "absent", ""
        return "probe_failed", f"{type(exc).__name__}: {code or exc}"


def evaluate_stage(
    registry: dict[str, Any],
    stage: str,
    *,
    s3_client: Any,
    now: datetime,
    run_date: str = "",
    cycle_date: date | None = None,
    window_start: datetime | None = None,
    enforce: bool = False,
) -> StageVerdict:
    """Return the verdict for ``stage``. Pure but for the injected S3 probes.

    Never raises. Every failure to establish an answer is
    :data:`STATUS_UNMEASURED` with a populated ``reason``.
    """
    recorded_at = now.astimezone(timezone.utc).isoformat()
    window_iso = (
        window_start.astimezone(timezone.utc).isoformat() if window_start else None
    )

    def unmeasured(reason: str, **extra: Any) -> StageVerdict:
        return StageVerdict(
            stage=stage,
            status=STATUS_UNMEASURED,
            reason=reason,
            run_date=run_date,
            cycle_date=cycle_date.isoformat() if cycle_date else "",
            window_start=window_iso,
            recorded_at=recorded_at,
            enforce=enforce,
            **extra,
        )

    row = resolve_stage_row(registry, stage)
    if row is None:
        return unmeasured(
            f"stage {stage!r} has no row in pipeline_stages: — it has declared "
            "nothing, which is not the same as declaring it writes nothing"
        )

    stage_class = str(row.get("stage_class", ""))
    declared_output = str(row.get("output", ""))

    if declared_output != "registered":
        # A POSITIVE declaration of no registerable output. Covered by having
        # said so — and still recorded, so the run can show that all 43 stages
        # answered rather than that 32 answered and 11 were never asked.
        return StageVerdict(
            stage=stage,
            status=STATUS_COVERED_NO_OUTPUT,
            stage_class=stage_class,
            declared_output=declared_output,
            reason=str(row.get("reason", "")),
            run_date=run_date,
            cycle_date=cycle_date.isoformat() if cycle_date else "",
            window_start=window_iso,
            recorded_at=recorded_at,
            enforce=enforce,
        )

    expected = [str(a) for a in (row.get("artifacts") or [])]
    if not expected:
        return unmeasured(
            f"stage {stage!r} declares output: registered with an empty "
            "artifacts list — the declaration is self-contradictory",
            stage_class=stage_class,
            declared_output=declared_output,
        )

    if cycle_date is None:
        return unmeasured(
            "cycle date unresolvable — refusing to guess the {date} "
            "substitution rather than probe a key that may not be this run's",
            stage_class=stage_class,
            declared_output=declared_output,
            expected=expected,
        )

    artifacts = index_artifacts(registry)
    missing: list[str] = []
    stale: list[str] = []
    covered: list[str] = []
    unmeasured_pairs: list[tuple[str, str]] = []

    for artifact_id in expected:
        spec = artifacts.get(artifact_id)
        if spec is None:
            unmeasured_pairs.append((artifact_id, "no artifacts: row for this id"))
            continue
        template = str(spec.get("s3_key_template", ""))
        if not template:
            unmeasured_pairs.append((artifact_id, "row carries no s3_key_template"))
            continue
        bucket = str(spec.get("s3_bucket") or DEFAULT_BUCKET)
        key = format_key(template, cycle_date)
        outcome, payload = _head(s3_client, bucket, key)
        if outcome == "absent":
            missing.append(artifact_id)
        elif outcome == "probe_failed":
            unmeasured_pairs.append((artifact_id, str(payload)))
        elif window_start is None or payload is None:
            # Present, but no window to judge recency against. NOT counted as
            # covered: a present key with no window is `unmeasured`, never a
            # pass — degrading it to healthy is how a stage that stopped
            # writing hides behind last cycle's object.
            unmeasured_pairs.append(
                (artifact_id, "present but no execution window to judge recency")
            )
        else:
            last_modified = payload
            if last_modified.tzinfo is None:
                last_modified = last_modified.replace(tzinfo=timezone.utc)
            if last_modified < window_start:
                stale.append(artifact_id)
            else:
                covered.append(artifact_id)

    # Precedence: a real absence outranks staleness, which outranks a partial
    # inability to measure. An UNMEASURED alongside a confirmed MISSING must
    # not mask the finding, and a confirmed finding must not be downgraded by
    # a sibling probe's harness fault.
    if missing:
        status = STATUS_MISSING
        reason = f"{len(missing)} declared artifact(s) absent: {', '.join(missing)}"
    elif stale:
        status = STATUS_STALE
        reason = f"{len(stale)} declared artifact(s) predate this run's window: {', '.join(stale)}"
    elif unmeasured_pairs and not covered:
        status = STATUS_UNMEASURED
        reason = "no declared artifact could be measured"
    elif unmeasured_pairs:
        status = STATUS_UNMEASURED
        reason = f"{len(covered)} of {len(expected)} artifacts confirmed; {len(unmeasured_pairs)} could not be measured"
    else:
        status = STATUS_COVERED
        reason = f"all {len(covered)} declared artifact(s) written in-window"

    return StageVerdict(
        stage=stage,
        status=status,
        stage_class=stage_class,
        declared_output=declared_output,
        expected=expected,
        missing=missing,
        stale=stale,
        covered=covered,
        unmeasured=unmeasured_pairs,
        reason=reason,
        run_date=run_date,
        cycle_date=cycle_date.isoformat(),
        window_start=window_iso,
        recorded_at=recorded_at,
        enforce=enforce,
    )


# ── Recording surface ────────────────────────────────────────────────────────


def verdict_key(verdict: StageVerdict, *, prefix: str = VERDICT_PREFIX) -> str:
    """S3 key the verdict record is written to."""
    stamp = verdict.run_date or verdict.cycle_date or "unknown"
    return f"{prefix}/{stamp}/{verdict.stage}.json"


def record_verdict(
    verdict: StageVerdict,
    *,
    s3_client: Any = None,
    cloudwatch_client: Any = None,
    bucket: str = DEFAULT_BUCKET,
    prefix: str = VERDICT_PREFIX,
) -> None:
    """Persist the verdict to S3 and publish its metric. Never raises.

    Recording is fail-soft and LOUD: the stage's primary deliverable is
    already written by the time this runs, so a recording failure must not
    destroy it — but a silent swallow would make the coverage surface report
    a smaller denominator with no signal, which is the defect this whole
    mechanism exists to remove.
    """
    if s3_client is not None:
        try:
            s3_client.put_object(
                Bucket=bucket,
                Key=verdict_key(verdict, prefix=prefix),
                Body=json.dumps(verdict.to_dict(), indent=2, default=str).encode(),
                ContentType="application/json",
            )
        except Exception:  # noqa: BLE001 — fail-soft, recorded at ERROR
            logger.error(
                "stage_coverage: FAILED to write verdict for %s to s3://%s/%s",
                verdict.stage,
                bucket,
                verdict_key(verdict, prefix=prefix),
                exc_info=True,
            )

    if cloudwatch_client is not None:
        try:
            cloudwatch_client.put_metric_data(
                Namespace=METRIC_NAMESPACE,
                MetricData=[
                    {
                        "MetricName": METRIC_NAME,
                        "Dimensions": [
                            {"Name": "Stage", "Value": verdict.stage},
                            {"Name": "Status", "Value": verdict.status},
                        ],
                        "Value": 1.0
                        if verdict.status in (STATUS_COVERED, STATUS_COVERED_NO_OUTPUT)
                        else 0.0,
                        "Unit": "None",
                    }
                ],
            )
        except Exception:  # noqa: BLE001 — fail-soft, recorded at ERROR
            logger.error(
                "stage_coverage: FAILED to publish coverage metric for %s",
                verdict.stage,
                exc_info=True,
            )


# ── Lambda-handler front door ────────────────────────────────────────────────


def assert_stage_coverage(
    stage: str,
    *,
    run_date: str = "",
    window_start: datetime | None = None,
    enforce: bool = False,
    now: datetime | None = None,
    s3_client: Any = None,
    cloudwatch_client: Any = None,
    bucket: str = DEFAULT_BUCKET,
    registry_local_path: str | None = None,
) -> dict[str, Any]:
    """Assert ``stage`` wrote what it declared. Returns the verdict dict.

    The front door for **Lambda-backed stages**: call immediately before
    returning, and merge the result into the handler's response payload under
    ``stage_coverage``. It never raises and never alters the handler's own
    outcome in observe mode — an observer that can kill the stage it observes
    is a new failure mode bolted onto the one it reports.

    ``enforce=True`` does not raise either; it sets ``is_finding`` in the
    returned dict. Acting on that remains the caller's decision, made in the
    caller's own failure route, per §2.1's one-stage-one-failure-route rule.
    """
    now = now or datetime.now(timezone.utc)
    try:
        if s3_client is None:
            import boto3

            from krepis.aws_region import resolve_region

            s3_client = boto3.client("s3", region_name=resolve_region())
        if cloudwatch_client is None:
            import boto3

            from krepis.aws_region import resolve_region

            cloudwatch_client = boto3.client("cloudwatch", region_name=resolve_region())

        try:
            registry = load_registry(
                s3_client, bucket=bucket, local_path=registry_local_path
            )
        except Exception as exc:  # noqa: BLE001
            verdict = StageVerdict(
                stage=stage,
                status=STATUS_UNMEASURED,
                reason=f"registry unreadable: {type(exc).__name__}: {exc}",
                run_date=run_date,
                recorded_at=now.astimezone(timezone.utc).isoformat(),
                enforce=enforce,
            )
        else:
            verdict = evaluate_stage(
                registry,
                stage,
                s3_client=s3_client,
                now=now,
                run_date=run_date,
                cycle_date=resolve_cycle_date(now),
                window_start=window_start,
                enforce=enforce,
            )

        record_verdict(
            verdict,
            s3_client=s3_client,
            cloudwatch_client=cloudwatch_client,
            bucket=bucket,
        )
        return verdict.to_dict()
    except Exception as exc:  # noqa: BLE001 — the assertion may never escape
        # NoRegionError specifically is no longer an environment gap: every
        # client above is built with `region_name=resolve_region()`, which
        # falls through env -> botocore session -> IMDS -> DEFAULT_REGION and
        # cannot return empty (alpha-engine-config-I7428). A NoRegionError
        # reaching here means resolve_region() itself is broken — a defect in
        # THIS module, not an unmeasurable box — so it is logged at CRITICAL
        # with a reason prefix a CloudWatch Logs metric filter or alarm can
        # match on, distinct from the ordinary UNMEASURED causes (unreadable
        # registry, unresolvable cycle date) that ARE genuinely environment
        # conditions. The verdict itself still degrades to UNMEASURED and
        # this still never raises to the caller — observe mode's contract
        # (never kill the stage it observes) is unconditional — but the
        # region-resolver-bug class of UNMEASURED is now distinguishable from
        # every other class of UNMEASURED on the log surface, and the
        # CloudWatch StageCoverage metric still gets its 0.0 datapoint below
        # (record_verdict already ran, or will on the caller's own retry) so
        # the coverage dashboard's minimum-of-metric alarm sees the gap
        # rather than a missing datapoint reading as green.
        is_region_bug = type(exc).__name__ == "NoRegionError"
        if is_region_bug:
            logger.critical(
                "stage_coverage: REGION_RESOLVER_BUG — resolve_region() "
                "returned no usable region and boto3 raised NoRegionError; "
                "this is a defect in krepis.aws_region, not an environment "
                "gap (alpha-engine-config-I7428)",
                exc_info=True,
            )
        else:
            logger.error("stage_coverage: assertion itself failed", exc_info=True)
        reason_prefix = "REGION_RESOLVER_BUG: " if is_region_bug else ""
        return StageVerdict(
            stage=stage,
            status=STATUS_UNMEASURED,
            reason=f"{reason_prefix}assertion failed: {type(exc).__name__}: {exc}",
            run_date=run_date,
            recorded_at=now.astimezone(timezone.utc).isoformat(),
            enforce=enforce,
        ).to_dict()


# ── CLI front door (bash launchers) ──────────────────────────────────────────


def _parse_window(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m krepis.stage_coverage",
        description=(
            "Assert a pipeline stage wrote the artifact it declared. "
            "Called from the stage's own launcher, immediately before it exits."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    assert_cmd = sub.add_parser("assert", help="assert one stage's declared output")
    assert_cmd.add_argument("--stage", required=True, help="SF Task-state name")
    assert_cmd.add_argument(
        "--run-date",
        default=os.environ.get("RUN_DATE", ""),
        help="execution run_date (defaults to $RUN_DATE)",
    )
    assert_cmd.add_argument(
        "--window-start",
        default=os.environ.get("STAGE_WINDOW_START", ""),
        help=(
            "ISO-8601 instant before which an artifact is STALE "
            "(defaults to $STAGE_WINDOW_START; absent ⇒ staleness unmeasured, "
            "absence still measured)"
        ),
    )
    assert_cmd.add_argument("--bucket", default=DEFAULT_BUCKET)
    assert_cmd.add_argument(
        "--registry-path",
        default=None,
        help="read the registry from a local file instead of S3 (tests)",
    )
    assert_cmd.add_argument(
        "--enforce",
        action="store_true",
        default=os.environ.get("STAGE_COVERAGE_ENFORCE", "").lower()
        in {"1", "true", "yes"},
        help=(
            f"exit {EXIT_COVERAGE_FAILURE} on a real MISSING. Promotion is this "
            "one flag; it is guarded by a test so the flip is deliberate."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint. Returns the process exit code; never raises.

    Exit codes:

    - ``0`` — observe mode always, and enforcing mode on anything that is not
      a real ``MISSING``.
    - :data:`EXIT_COVERAGE_FAILURE` — enforcing mode on a real ``MISSING``.
    - ``2`` — argparse usage error (argparse's own convention).
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)

    window_raw = getattr(args, "window_start", "") or ""
    window = _parse_window(window_raw)
    if window_raw and window is None:
        logger.warning(
            "stage_coverage: --window-start %r is unparseable; staleness will "
            "be UNMEASURED (absence is still measured)",
            window_raw,
        )

    verdict_dict = assert_stage_coverage(
        args.stage,
        run_date=args.run_date,
        window_start=window,
        enforce=args.enforce,
        bucket=args.bucket,
        registry_local_path=args.registry_path,
    )

    status = verdict_dict.get("status", STATUS_UNMEASURED)
    line = f"stage-coverage {args.stage}: {status} — {verdict_dict.get('reason', '')}"
    if status in (STATUS_COVERED, STATUS_COVERED_NO_OUTPUT):
        logger.info("%s", line)
    else:
        # Loud on every surface except the one that kills the run.
        print(f"WARNING: {line}", file=sys.stderr)

    if args.enforce and verdict_dict.get("is_finding"):
        return EXIT_COVERAGE_FAILURE
    return 0


if __name__ == "__main__":  # pragma: no cover — exercised via main() in tests
    raise SystemExit(main())
