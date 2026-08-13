"""
Per-stage peak-RSS budget: measure it, publish headroom, warn before the wall.

WHY THIS EXISTS (alpha-engine-config-I7260)
-------------------------------------------
Three of the eight distinct weekly-SF failures between 2026-08-10 and
2026-08-13 were kernel OOM kills. Two were "fixed" the same day by RAISING
instance floors (``crucible-backtester-PR653``/``PR657`` for
``spot_backtester.sh``, ``PR659`` for ``spot_evaluator.sh``). Neither floor is
derived from a measurement, and the code says so in its own comment: *"a cap
derived from another cap is not a budget."*

The reason is structural: **an OOM-killed process reports no peak RSS.** The
only runs that can tell us the real requirement are the ones that SURVIVE —
and until this module, no stage recorded it on a successful run either. So the
floors were set by doubling until the kills stopped, with no number to check
the guess against and no signal when a stage drifts back toward its ceiling.

This class is also not coverable by the daily all-stage preflight sweep
(``alpha-engine-config-I7249``): ``--preflight-only`` runs a smoke workload and
the kill only happens at full data volume. Detection is the wrong instrument.
The right one is a **bounded, trended budget with a warn band** — the memory
analogue of the preopen schedule-buffer canary (``sf-pipeline-policy`` §1.2,
``alpha-engine-config-I2412``), whose shape this module deliberately mirrors:
a hard floor, an early-warning floor beneath it, and a rolling median over the
last N observations that alarms on the TREND even when the latest single
observation is fine, with a minimum sample count so a trend is never
fabricated from one or two points.

WHY THIS LIVES IN ``ssm_dispatcher`` AND NOT ``ssm_log_capture``
-----------------------------------------------------------------
``ssm_log_capture`` looks like the chokepoint — every weekly-SF state invokes
it — but its direct child is the **dispatcher** script
(``bash infrastructure/spot_evaluator.sh``) running on the dashboard box. The
process that gets OOM-killed runs on a **spot instance**, reached over SSM.
Measuring the dispatcher's children measures the wrong machine entirely.

``krepis.ssm_dispatcher run`` IS the real chokepoint: measured 2026-08-13, all
three dispatcher repos (``nousergon-data``, ``crucible-predictor``,
``crucible-backtester``) funnel every remote step through it from
``_spot_common.sh::run_ssm``, and every weekly-SF state resolves it through the
one interpreter ``/home/ec2-user/alpha-engine-dashboard/.venv/bin/python``.
One module therefore covers every stage in every repo, with no per-launcher
edit and no Step Functions definition change.

WHAT IS MEASURED
----------------
``resource.getrusage(RUSAGE_CHILDREN).ru_maxrss`` read by a small Python
harness on the spot box, after the stage body exits. On Linux this is the
high-water mark of the largest **descendant** the harness reaped — the kernel
propagates a child's ru_maxrss up to its parent on wait(), so a body that
shells out through bash into Python into ArcticDB still reports the true peak
of the whole subtree. The denominator is ``MemTotal`` from ``/proc/meminfo``,
which is what the OOM killer actually works against; the instance type is
recorded alongside it as a label, best-effort from IMDSv2.

FAILURE POSTURE — THE ASYMMETRY IS DELIBERATE
----------------------------------------------
This is SECONDARY observability. A failure to read peak RSS must never fail a
stage that did its real work — an observability change that causes a
production outage is strictly worse than the gap it closes. So every path here
swallows, and each swallow is named:

(a) **Failure mode swallowed:** the harness cannot start (no ``python3`` on the
    box), cannot read ``/proc/meminfo`` or ``/sys``, cannot reach IMDS, emits a
    sentinel the dispatcher never sees (24KB inline-output rotation), or the
    S3 read-modify-write of the console envelope fails (AccessDenied, no
    credentials, transient network).
(b) **Why the primary deliverable survives:** the stage body runs FIRST and its
    exit code is propagated verbatim in every branch, including the branch
    where the harness itself is skipped. The dispatcher's return value is
    computed from the SSM terminal status alone and is never touched by
    anything in this module.
(c) **Concrete recording surface:** the reading is marked
    ``measured: false`` with a ``reason``, and the stage's console row is
    published as ``attention`` — the envelope vocabulary's honest middle —
    naming that reason in its ``summary``. It is NEVER omitted and NEVER
    rendered ``ok``: a stage with no reading is UNOBSERVED, not healthy
    (``principles.md`` §2.7). The same posture and the same status mapping as
    ``nousergon-data``'s ``preflight_sweep_console.py``, which is the sibling
    surface this one shares a console prefix with.

THE THRESHOLDS ARE INITIAL VALUES, NOT MEASUREMENTS
-----------------------------------------------------
:data:`HARD_HEADROOM_FLOOR` (0.15) and :data:`WARN_HEADROOM_FLOOR` (0.30) are
**declared starting points, not derived from data** — stating otherwise would
repeat the exact error this module exists to end. Their basis is written out at
their definition. They are expected to be re-derived from the first surviving
runs, along with the instance floors themselves; that is step 4 of
``alpha-engine-config-I7260`` and it cannot be done before readings exist.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import statistics
from datetime import datetime, timezone
from typing import Any, Final, Optional

logger = logging.getLogger(__name__)

# ── Console surface ──────────────────────────────────────────────────────────
#
# The fleet check-result envelope, read by nousergon-console's
# `checks-envelope` adapter, which is already configured against the WHOLE
# `ops/checks/` prefix (nous-ergon-ops/nousergon-console/config.d/
# fleet-checks.yaml). A new check_id under that prefix therefore appears on the
# console with no console change and no enumeration anywhere — the same
# property `preflight_sweep_console.py` relies on. Do not invent a second
# surface for this (`console-policy` §2.6).
CHECKS_PREFIX: Final[str] = "ops/checks"
CHECK_ID_PREFIX: Final[str] = "ae-rss"

#: The envelope's own status vocabulary. Anything outside it renders
#: UNREPORTED, which is why the mapping below is explicit rather than a str()
#: of an internal verdict.
ENVELOPE_OK: Final[str] = "ok"
ENVELOPE_ATTENTION: Final[str] = "attention"
ENVELOPE_ERROR: Final[str] = "error"

#: Weekly-SF stages run once a week, so a row older than ~1.5 weeks is stale.
#: Declared here rather than guessed by the console: a staleness threshold
#: invented at read time is indistinguishable on the surface from one that was
#: ruled.
CADENCE_MINUTES: Final[int] = 7 * 24 * 60

#: Readings retained in the envelope for the trend. Ten weekly runs ≈ one
#: quarter — long enough to see creep, short enough that a genuine step change
#: (a new stage workload) leaves the window within a quarter instead of
#: dragging the median for a year.
HISTORY_LIMIT: Final[int] = 10

# ── The warn band (mirrors sf-pipeline-policy §1.2 / config-I2412) ───────────
#
# "Headroom" here is the FRACTION OF THE BOX LEFT FREE at the stage's peak:
#     headroom = 1 - peak_rss / mem_total
# so higher is safer and the numbers read the same direction as the preopen
# canary's "minutes before open".
#
# BOTH NUMBERS BELOW ARE INITIAL VALUES. They are NOT derived from a measured
# peak RSS, because on the day this shipped no surviving run had ever recorded
# one. Their stated basis:
#
#   * The 2026-08-13 evaluator kill was a 4 GB c5.large. AL2023 plus the SSM
#     agent plus the venv account for roughly 0.7-1.0 GB before the workload
#     allocates anything — ~20% of that box. A stage sitting below 15% free is
#     therefore inside the noise band of the OS itself and is one allocation
#     spike from the kill; that is the hard floor.
#   * `crucible-backtester/infrastructure/_spot_common.sh` already encodes a
#     "6.0 GB headroom guard" on 8 GB instances (I3280), i.e. an operator
#     intuition of roughly a 25-30% margin. 30% is that intuition written down
#     as the early-warning floor rather than left implicit in a comment.
#
# Both are expected to move once ~4 surviving runs per stage exist
# (alpha-engine-config-I7260 step 4). Presenting either as measured would be
# the same defect as the floors they are here to falsify.
HARD_HEADROOM_FLOOR: Final[float] = 0.15
WARN_HEADROOM_FLOOR: Final[float] = 0.30

#: Minimum readings before a rolling median is trusted. Directly mirrors
#: `pipeline-watchdog`'s `_MIN_TREND_DAYS` — never compute a "10-run median"
#: off one or two points.
MIN_TREND_SAMPLES: Final[int] = 3

# ── Which steps produce a reading ────────────────────────────────────────────
#
# `run_ssm` is called several times per stage: infrastructure steps
# (bootstrap, dependency install, cache fetch) and then the stage workload.
# Only the workload's peak is a budget fact about the stage.
#
# The load-bearing case is `preflight-only`: the daily all-stage sweep
# (alpha-engine-config-I7249) drives every launcher with `--preflight-only`,
# whose smoke workload allocates almost nothing. Publishing those readings
# would flood every stage's row with rosy headroom from a workload that never
# touches the ~900-ticker universe — i.e. it would manufacture exactly the
# false confidence this module exists to remove.
#
# A step name absent from this set is published, which is the safe direction:
# an unlisted infrastructure step can only ever make a stage look HEAVIER than
# it is, never lighter. Adding a new infrastructure step to `_spot_common.sh`
# should add its name here.
INFRASTRUCTURE_STEPS: Final[frozenset[str]] = frozenset(
    {"bootstrap", "deps", "predictor-cache", "preflight-only"}
)

#: Marker the on-box harness prints so the dispatcher can recover the reading
#: from the streamed stdout. Deliberately unlikely to occur in workload output.
SENTINEL: Final[str] = "##krepis-rss-reading##"

_SENTINEL_RE = re.compile(re.escape(SENTINEL) + r"\s*(\{.*?\})\s*$", re.MULTILINE)

_SLUG_RE = re.compile(r"[^a-z0-9]+")

_KIB = 1024


def _now_iso(now: Optional[datetime] = None) -> str:
    return (now or datetime.now(timezone.utc)).isoformat()


def split_description(description: str) -> "tuple[str, str]":
    """Split ``run_ssm``'s ``"<stage>: <step>"`` label into its two halves.

    Every dispatcher repo builds the description the same way — measured
    2026-08-13 across ``nousergon-data``, ``crucible-predictor`` and
    ``crucible-backtester``, all three spell
    ``--description "${STAGE}: $description"``. A label without the separator
    is treated as a stage with an unnamed step rather than rejected: a
    description shape is not something an observability path may fail on.
    """
    if ":" in description:
        stage, _, step = description.partition(":")
        return stage.strip(), step.strip()
    return description.strip(), ""


def slug(value: str) -> str:
    """Lowercase, hyphen-joined slug. Stable across runs; never mints a uuid."""
    return _SLUG_RE.sub("-", value.strip().lower()).strip("-")


def check_id(stage: str) -> str:
    """Console component id for one stage's RSS-budget row.

    Derived from the stage name the dispatcher already passes, so a stage added
    to any pipeline gets a row by itself with nothing hand-listed anywhere.
    """
    return f"{CHECK_ID_PREFIX}-{slug(stage)}"


def is_publishable_step(step: str) -> bool:
    """Whether this ``run_ssm`` step's reading belongs on the stage's row.

    An EMPTY step (a description with no ``"<stage>: <step>"`` separator) does
    not publish. All three dispatcher repos spell the separator — measured
    2026-08-13 — so its absence means the caller is some other krepis consumer
    whose steps this module cannot classify, and classifying an unknown call as
    a stage workload would put a row on the fleet console for something that is
    not a weekly-SF stage.
    """
    if not step.strip():
        return False
    return slug(step) not in INFRASTRUCTURE_STEPS


# ── The on-box harness ───────────────────────────────────────────────────────

# Runs the stage body as a child, then reports the high-water mark of the whole
# reaped subtree. Kept deliberately dependency-free: stdlib only, no boto3, no
# aws CLI, no `/usr/bin/time` (which is a separate RPM and is NOT guaranteed on
# a fresh AL2023 spot). `python3` is the one interpreter AL2023 ships in the
# base AMI, and the fallback branch below covers even its absence.
_HARNESS_PY = r'''
import json, os, resource, subprocess, sys

body = sys.argv[1]
sentinel = sys.argv[2]

# The stage body runs FIRST and its exit code is the only thing that
# propagates. Nothing below this line may change it.
rc = subprocess.call(["bash", body])

record = {"measured": False, "reason": "harness did not complete"}
try:
    peak_kb = int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    mem_total_kb = None
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemTotal:"):
                mem_total_kb = int(line.split()[1])
                break
    if mem_total_kb:
        record = {
            "measured": True,
            "peak_rss_kb": peak_kb,
            "mem_total_kb": mem_total_kb,
            "instance_type": None,
            "run_token": os.environ.get("RUN_TOKEN") or None,
        }
        try:
            import urllib.request
            req = urllib.request.Request(
                "http://169.254.169.254/latest/api/token",
                method="PUT",
                headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
            )
            token = urllib.request.urlopen(req, timeout=2).read().decode()
            req = urllib.request.Request(
                "http://169.254.169.254/latest/meta-data/instance-type",
                headers={"X-aws-ec2-metadata-token": token},
            )
            record["instance_type"] = urllib.request.urlopen(req, timeout=2).read().decode()
        except Exception as exc:
            # A label, not the denominator. MemTotal is what the OOM killer
            # works against and it is already recorded.
            record["instance_type_error"] = "%s: %s" % (type(exc).__name__, exc)
    else:
        record = {"measured": False, "reason": "MemTotal absent from /proc/meminfo"}
except Exception as exc:
    record = {"measured": False, "reason": "%s: %s" % (type(exc).__name__, exc)}

try:
    sys.stdout.write("\n%s %s\n" % (sentinel, json.dumps(record)))
    sys.stdout.flush()
except Exception:
    pass

sys.exit(rc)
'''


def wrap_script(script: str, *, sentinel: str = SENTINEL) -> str:
    """Wrap a stage body so it reports its subtree's peak RSS on exit.

    The body is carried as base64 and written to a temp file rather than
    interpolated, so it may contain any heredoc, quote or delimiter — the same
    reason ``ssm_dispatcher._encode_command_payload`` base64-wraps for
    transport.

    The wrapper's exit status is the BODY's exit status on every path,
    including the no-``python3`` fallback. That invariant is the whole reason
    this is safe to enable by default.
    """
    body_b64 = base64.b64encode(script.encode("utf-8")).decode("ascii")
    harness_b64 = base64.b64encode(_HARNESS_PY.encode("utf-8")).decode("ascii")
    return f"""__krepis_rss_body="$(mktemp /tmp/krepis-rss-body.XXXXXX)"
__krepis_rss_harness="$(mktemp /tmp/krepis-rss-harness.XXXXXX)"
printf '%s' '{body_b64}' | base64 -d > "$__krepis_rss_body"
printf '%s' '{harness_b64}' | base64 -d > "$__krepis_rss_harness"
__krepis_rss_rc=0
if command -v python3 >/dev/null 2>&1; then
    python3 "$__krepis_rss_harness" "$__krepis_rss_body" '{sentinel}' || __krepis_rss_rc=$?
else
    # No interpreter for the harness. Run the body unmeasured rather than not
    # at all — this is observability, and it never gates the work.
    printf '\\n%s %s\\n' '{sentinel}' '{{"measured": false, "reason": "python3 absent on instance"}}'
    bash "$__krepis_rss_body" || __krepis_rss_rc=$?
fi
rm -f "$__krepis_rss_body" "$__krepis_rss_harness"
exit $__krepis_rss_rc
"""


def parse_reading(stdout: str) -> "Optional[dict[str, Any]]":
    """Recover the last harness record from streamed stdout, or ``None``.

    ``None`` means the sentinel never arrived — a distinct condition from
    ``{"measured": false}`` (the harness ran and could not measure), and the
    caller renders the two with different reasons.

    The LAST match wins: a workload that happens to echo the sentinel earlier
    cannot displace the harness's own line, which is always emitted after the
    body has exited.
    """
    matches = _SENTINEL_RE.findall(stdout or "")
    for blob in reversed(matches):
        try:
            parsed = json.loads(blob)
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


# ── Headroom, thresholds and the trend ───────────────────────────────────────


def headroom(peak_rss_kb: int, mem_total_kb: int) -> float:
    """Fraction of the box left free at the stage's peak. Higher is safer.

    Clamped to ``[0, 1]``: ru_maxrss can exceed MemTotal on a box with swap or
    with a short-lived overcommit, and a negative headroom would read as a
    stranger number than "zero free", which is what it means operationally.
    """
    if mem_total_kb <= 0:
        raise ValueError("mem_total_kb must be positive to compute headroom")
    return max(0.0, min(1.0, 1.0 - (float(peak_rss_kb) / float(mem_total_kb))))


def trend_median(history: "list[dict[str, Any]]") -> "Optional[tuple[float, int]]":
    """Median headroom across the retained readings, or ``None``.

    Returns ``None`` below :data:`MIN_TREND_SAMPLES` rather than a median of
    one point — the `pipeline-watchdog` rule, restated: a trend fabricated from
    too few observations is worse than no trend, because it renders with the
    same authority.
    """
    values = [
        float(r["headroom"])
        for r in history
        if isinstance(r, dict) and isinstance(r.get("headroom"), (int, float))
    ]
    if len(values) < MIN_TREND_SAMPLES:
        return None
    return statistics.median(values), len(values)


def classify(
    reading: "Optional[dict[str, Any]]",
    history: "list[dict[str, Any]]",
) -> "tuple[str, str]":
    """Map a reading plus its history onto ``(envelope_status, summary)``.

    Three tiers, mirroring the preopen buffer canary:

    * **hard floor breach** (headroom < :data:`HARD_HEADROOM_FLOOR`) → ``error``
    * **early-warning floor** (headroom < :data:`WARN_HEADROOM_FLOOR`), or a
      rolling median below it even when THIS run cleared it → ``attention``
    * otherwise → ``ok``

    A missing or unmeasurable reading is ``attention``, never ``error`` and
    never ``ok``. The argument, written down because it is a judgment call:
    ``error`` is what a real finding about the pipeline looks like, and a
    failed measurement on a stage that did its work is not one — routing it to
    ``error`` would page an operator about an observability fault wearing a
    production fault's clothes, and would make the surface untrustworthy in the
    direction that gets alarms muted. ``ok`` is forbidden outright by
    ``principles.md`` §2.7. ``attention`` is the envelope vocabulary's declared
    honest middle, and it is the same mapping ``preflight_sweep_console.py``
    already uses for its ``unmeasured`` verdict on the same console prefix.
    """
    if reading is None:
        return (
            ENVELOPE_ATTENTION,
            "UNOBSERVED — the stage ran but no peak-RSS reading reached the "
            "dispatcher (the harness sentinel never arrived; the inline SSM "
            "output cap may have rotated it away). This is not a pass: the "
            "stage's memory budget is unknown for this run.",
        )
    if not reading.get("measured"):
        return (
            ENVELOPE_ATTENTION,
            "UNOBSERVED — peak RSS could not be measured on this run: "
            f"{reading.get('reason') or 'no reason recorded'}. The stage's own "
            "work is unaffected; its memory budget is unknown for this run.",
        )

    peak = int(reading["peak_rss_kb"])
    total = int(reading["mem_total_kb"])
    free = headroom(peak, total)
    label = (
        f"peak RSS {peak / _KIB / _KIB:.2f} GiB of {total / _KIB / _KIB:.2f} GiB "
        f"({reading.get('instance_type') or 'instance type unknown'}) — "
        f"{free * 100:.1f}% headroom"
    )

    if free < HARD_HEADROOM_FLOOR:
        return (
            ENVELOPE_ERROR,
            f"{label}. BELOW the {HARD_HEADROOM_FLOOR * 100:.0f}% hard floor — "
            "this stage is inside the OS's own overhead band and one "
            "allocation spike from an OOM kill. Raise the instance floor for "
            "this stage, or reduce what it materialises.",
        )

    trend = trend_median(history)
    if free < WARN_HEADROOM_FLOOR:
        return (
            ENVELOPE_ATTENTION,
            f"{label}. Below the {WARN_HEADROOM_FLOOR * 100:.0f}% early-warning "
            "floor. Alarming on the trend, not on the kill "
            "(sf-pipeline-policy §1.2).",
        )
    if trend is not None and trend[0] < WARN_HEADROOM_FLOOR:
        return (
            ENVELOPE_ATTENTION,
            f"{label}. This run cleared the early-warning floor, but the median "
            f"headroom over the last {trend[1]} readings is "
            f"{trend[0] * 100:.1f}% — below the "
            f"{WARN_HEADROOM_FLOOR * 100:.0f}% floor. Headroom is eroding.",
        )
    return (ENVELOPE_OK, label)


def build_envelope(
    *,
    stage: str,
    step: str,
    reading: "Optional[dict[str, Any]]",
    previous: "Optional[dict[str, Any]]",
    instance_id: str,
    correlation_id: Optional[str] = None,
    now: Optional[datetime] = None,
) -> "dict[str, Any]":
    """Build this stage's console row, folding the new reading into the old one.

    Two merge rules, both there to keep the row honest across the several
    ``run_ssm`` steps that share one spot instance:

    * **Max within one instance.** The box's exposure is the highest peak any
      step reached on it, so a later, lighter step never lowers the recorded
      peak for the same ``instance_id``. This is what makes
      :data:`INFRASTRUCTURE_STEPS` a noise filter rather than a correctness
      dependency.
    * **History appends once per instance.** A new ``instance_id`` is a new
      run, so the previous run's reading is retired into ``history`` and the
      trend advances by exactly one point per stage run.
    """
    ran_at = _now_iso(now)
    previous = previous if isinstance(previous, dict) else {}
    prev_instance = previous.get("instance_id")
    history = [h for h in previous.get("history", []) if isinstance(h, dict)]

    merged = reading
    if prev_instance == instance_id and previous.get("reading"):
        prev_reading = previous["reading"]
        # Same box, later step: keep whichever peak is higher, and keep a real
        # measurement over an unmeasured one.
        if isinstance(prev_reading, dict) and prev_reading.get("measured"):
            if not (reading and reading.get("measured")):
                merged = prev_reading
            elif int(prev_reading.get("peak_rss_kb", 0)) >= int(
                reading.get("peak_rss_kb", 0)
            ):
                merged = prev_reading
    elif previous.get("reading") is not None or previous.get("headroom") is not None:
        # A different box: the previous row described a finished run. Retire it
        # into the trend before overwriting.
        if isinstance(previous.get("headroom"), (int, float)):
            history.append(
                {
                    "ran_at": previous.get("ran_at"),
                    "instance_id": prev_instance,
                    "instance_type": previous.get("instance_type"),
                    "peak_rss_kb": previous.get("peak_rss_kb"),
                    "mem_total_kb": previous.get("mem_total_kb"),
                    "headroom": previous.get("headroom"),
                    "correlation_id": previous.get("correlation_id"),
                }
            )
    history = history[-HISTORY_LIMIT:]

    status, summary = classify(merged, history)
    body: "dict[str, Any]" = {
        "schema_version": 1,
        "check_id": check_id(stage),
        "component_id": check_id(stage),
        "label": f"weekly-SF stage {stage} — memory budget",
        "status": status,
        "ran_at": ran_at,
        "cadence_minutes": CADENCE_MINUTES,
        "stage": stage,
        "step": step,
        "instance_id": instance_id,
        "correlation_id": correlation_id,
        "measured": bool(merged and merged.get("measured")),
        "unmeasured_reason": (
            None
            if (merged and merged.get("measured"))
            else (merged or {}).get("reason", "no reading reached the dispatcher")
        ),
        "peak_rss_kb": (merged or {}).get("peak_rss_kb"),
        "mem_total_kb": (merged or {}).get("mem_total_kb"),
        "instance_type": (merged or {}).get("instance_type"),
        "headroom": None,
        "hard_headroom_floor": HARD_HEADROOM_FLOOR,
        "warn_headroom_floor": WARN_HEADROOM_FLOOR,
        "thresholds_basis": (
            "INITIAL VALUES, not derived from measured peak RSS — see "
            "krepis.rss_budget module docstring and alpha-engine-config-I7260 "
            "step 4, which re-derives both these floors and the instance "
            "floors once surviving-run readings exist."
        ),
        "history": history,
        "trend_median_headroom": None,
        "trend_samples": 0,
        "reading": merged,
        "summary": summary,
        "evidence": f"ssm command on {instance_id}",
    }
    if merged and merged.get("measured"):
        body["headroom"] = round(
            headroom(int(merged["peak_rss_kb"]), int(merged["mem_total_kb"])), 4
        )
    trend = trend_median(history)
    if trend is not None:
        body["trend_median_headroom"] = round(trend[0], 4)
        body["trend_samples"] = trend[1]
    return body


def envelope_key(stage: str, prefix: str = CHECKS_PREFIX) -> str:
    return f"{prefix.rstrip('/')}/{check_id(stage)}/latest.json"


def publish(
    *,
    bucket: str,
    description: str,
    instance_id: str,
    stdout: str,
    correlation_id: Optional[str] = None,
    prefix: str = CHECKS_PREFIX,
    s3_client=None,
    now: Optional[datetime] = None,
) -> "Optional[dict[str, Any]]":
    """Read-modify-write this stage's console row. **Never raises.**

    Returns the published body, or ``None`` when nothing was published (an
    infrastructure step, or an S3 failure that was swallowed). The return value
    exists for tests and for the caller's log line; no caller may branch its
    exit code on it.
    """
    try:
        stage, step = split_description(description)
        if not stage:
            return None
        if not is_publishable_step(step):
            return None

        reading = parse_reading(stdout)
        key = envelope_key(stage, prefix)

        if s3_client is None:
            import boto3

            s3_client = boto3.client("s3")

        previous: "Optional[dict[str, Any]]" = None
        try:
            obj = s3_client.get_object(Bucket=bucket, Key=key)
            previous = json.loads(obj["Body"].read().decode("utf-8"))
        except Exception:
            # No prior row (first run for this stage) or an unreadable one.
            # Publishing a fresh row is strictly better than publishing
            # nothing; the trend simply restarts, and `trend_samples` says so.
            previous = None

        body = build_envelope(
            stage=stage,
            step=step,
            reading=reading,
            previous=previous,
            instance_id=instance_id,
            correlation_id=correlation_id,
            now=now,
        )
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(body, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
        return body
    except Exception as exc:
        # (a) swallowed: any failure of the memory-budget publication path.
        # (b) the stage's own exit code is computed by ssm_dispatcher.run from
        #     the SSM terminal status and is untouched by this function.
        # (c) recorded here, at WARNING, in the dispatcher's captured log —
        #     which ssm_log_capture ships to
        #     s3://alpha-engine-research/_ssm_logs/<slug>/<date>/ for the run.
        logger.warning(
            "rss_budget: could not publish the memory-budget row for %r "
            "(swallowed; stage exit code unaffected): %s: %s",
            description,
            type(exc).__name__,
            exc,
        )
        return None


# ── Rehearsal CLI ────────────────────────────────────────────────────────────


def _selftest() -> int:
    """Run the real harness against a real allocation, locally.

    This is the rehearsal path: it exercises :func:`wrap_script`,
    :func:`parse_reading`, :func:`headroom` and :func:`classify` end to end on
    an actual process whose peak RSS is genuinely measured — no AWS, no spot
    instance, no waiting for Saturday.
    """
    import subprocess
    import sys
    import tempfile

    body = (
        "python3 -c \"import sys; b=bytearray(96*1024*1024); "
        "sys.stdout.write('allocated %d MiB\\n' % (len(b)//1048576))\"\n"
    )
    wrapped = wrap_script(body)
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as fh:
        fh.write(wrapped)
        path = fh.name
    proc = subprocess.run(["bash", path], capture_output=True, text=True)
    print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    reading = parse_reading(proc.stdout)
    print(f"body exit code : {proc.returncode} (must be 0)")
    print(f"reading        : {json.dumps(reading)}")
    if not reading or not reading.get("measured"):
        print(
            "SELFTEST: no measured reading — this platform does not expose "
            "/proc/meminfo (macOS). The wrapper still returned the body's exit "
            "code, which is the invariant that matters; run this on Linux for "
            "a real measurement."
        )
        return 0 if proc.returncode == 0 else 1
    status, summary = classify(reading, [])
    print(f"headroom       : {headroom(reading['peak_rss_kb'], reading['mem_total_kb']):.4f}")
    print(f"verdict        : {status} — {summary}")
    return 0 if proc.returncode == 0 else 1


def _rehearse(peak_rss_kb: int, mem_total_kb: int, history_headroom: "list[float]") -> int:
    """Print the verdict for a hypothetical reading — the induction path.

    ``closes-when`` on alpha-engine-config-I7260 asks for the warn band to be
    verified by induction: drive a stage toward a small instance and confirm
    the WARN fires before the kill. This subcommand is that induction without
    burning a spot instance — it walks a stage from healthy headroom down to
    the hard floor and shows the band firing first.
    """
    history = [{"headroom": h} for h in history_headroom]
    reading = {
        "measured": True,
        "peak_rss_kb": peak_rss_kb,
        "mem_total_kb": mem_total_kb,
        "instance_type": "rehearsal",
    }
    status, summary = classify(reading, history)
    print(json.dumps({"status": status, "summary": summary}, indent=2))
    return 0


def main(argv: "Optional[list[str]]" = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m krepis.rss_budget",
        description=(
            "Per-stage peak-RSS budget: the measurement behind every weekly-SF "
            "RAM floor (alpha-engine-config-I7260). This CLI is the rehearsal "
            "surface; the measurement itself runs automatically inside "
            "krepis.ssm_dispatcher."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser(
        "selftest",
        help=(
            "Run the on-box harness locally against a real 96 MiB allocation "
            "and print the measured reading + verdict."
        ),
    )
    r = sub.add_parser(
        "rehearse",
        help=(
            "Print the verdict for a hypothetical reading — induction for the "
            "warn band with no spot instance."
        ),
    )
    r.add_argument("--peak-rss-kb", type=int, required=True)
    r.add_argument("--mem-total-kb", type=int, required=True)
    r.add_argument(
        "--history-headroom",
        default="",
        help="Comma-separated prior headroom fractions, e.g. '0.31,0.29,0.27'.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    if args.cmd == "selftest":
        return _selftest()
    hist = [
        float(v) for v in str(args.history_headroom).split(",") if v.strip()
    ]
    return _rehearse(args.peak_rss_kb, args.mem_total_kb, hist)


if __name__ == "__main__":
    import sys

    sys.exit(main())
