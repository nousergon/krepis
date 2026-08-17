"""metrics — the System Report Card v2 ``MetricRecord`` contract + status derivation.

A ``MetricRecord`` is the unit of the v2 report card: every graded component
(research / predictor / executor / backtester / substrate / agent / portfolio)
emits one, carrying not just a value but its statistical context — CI, sample
size vs floor, target, red-line, trend, and a derived status/letter. The letter
is *derived* from the status, never the source of truth (RC v2 Principle 2).

This module is the shared chokepoint: the producer (the evaluator's grading
layer) and every consumer (dashboard console, public site) agree on the schema
AND on the status semantics via the pure ``derive_*`` helpers here — so the same
``(value, CI, N)`` maps to the same GREEN/WATCH/RED everywhere.

The N/A taxonomy distinguishes the four engineering states that the legacy
"insufficient data" string conflated:
  - ``N/A-NOT-IMPL``     grader exists, producer analysis not yet wired
  - ``N/A-NOT-RUN``      producer implemented but did not run this cycle
  - ``N/A-LOW-N``        ran, but N below half the floor — CI too wide to read
  - ``N/A-MISSING-INPUT``ran, but a required upstream artifact was absent

Authoritative design: ``alpha-engine-docs/private/system-report-card-revamp-260522.md``.
Module-level aggregation (critical-gate module roll-up, BH-FDR over a tile's
component family) lives in the evaluator, not here — this is the per-component
contract only.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

StatusLiteral = Literal[
    "GREEN",
    "WATCH",
    "RED",
    "N/A-NOT-IMPL",
    "N/A-NOT-RUN",
    "N/A-LOW-N",
    "N/A-MISSING-INPUT",
]
CriticalityLiteral = Literal["critical", "supporting", "diagnostic"]
MetricTypeLiteral = Literal[
    "ic", "lift", "ratio", "pct", "count", "duration",
    "sharpe", "calibration", "p_value", "zscore", "log_return",
    # Report Card v3 (alpha-engine-config-I7473, T2 §3): a component's measured
    # leave-one-out / baseline-substitution marginal contribution to the ONE
    # objective (net-of-cost 21d log-alpha vs SPY). Producers emit it with
    # ``unit="log_alpha_21d"`` and ``red_line=0.0`` (a component whose measured
    # contribution is <= 0 is not paying for its complexity budget). Kept a
    # distinct kind from ``lift`` because ``lift`` records are IC/precision-
    # space deltas that never ran through the cost-bearing simulator.
    "contribution_lift",
]
TrendDecorationLiteral = Literal["↑↑", "↑", "→", "↓", "↓↓"]

_NA_PREFIX = "N/A"


class MetricRecord(BaseModel):
    """One graded component of the System Report Card v2.

    ``extra="allow"`` for forward-compat: a newer producer may add fields a
    older consumer hasn't learned yet without breaking the read.
    """

    model_config = ConfigDict(extra="allow")

    name: str = Field(description="snake_case stable key, e.g. 'predictor_meta_l2_ic'.")
    module: str = Field(
        description="Owning tile: portfolio | research | predictor | executor | "
        "backtester | substrate | agent."
    )
    metric_type: MetricTypeLiteral
    value: float | None = Field(default=None, description="Measured value (None when N/A-*).")
    unit: str | None = Field(
        default=None,
        description=(
            "Unit `value` is expressed in, e.g. 'ratio', 'bps', 'days', "
            "'usd'. `metric_type` names the metric's STATISTICAL KIND "
            "(ic/lift/ratio/pct/...), not its unit — the two are declared "
            "separately because a kind does not pin a unit (a 'ratio' "
            "metric_type can be a fraction-of-one or a percent-of-hundred). "
            "Required whenever `value` is set (see the validator below): a "
            "numeric field with no declared unit is the exact defect "
            "observability-policy.md §3.4 forbids — `avg_volume_20d` was "
            "emitted as a normalized ratio and consumed as raw shares, "
            "silently failing 901 of 903 tickers for months."
        ),
    )
    ci_low: float | None = Field(default=None)
    ci_high: float | None = Field(default=None)
    ci_method: str | None = Field(
        default=None, description="e.g. 'bootstrap', 'newey-west', 'wilson'."
    )
    n_samples: int | None = Field(default=None, description="Observations behind the value.")
    n_floor: int = Field(description="Documented minimum N for a confident reading.")
    target: float | None = Field(default=None, description="At/beyond here = good.")
    red_line: float | None = Field(default=None, description="At/beyond here = system-breaking.")
    trend_4w: list[float] | None = Field(default=None)
    trend_13w: list[float] | None = Field(default=None)
    trend_decoration: TrendDecorationLiteral = Field(default="→")
    status: StatusLiteral
    status_reason: str = Field(description="One operator-readable sentence; never generic.")
    criticality: CriticalityLiteral = Field(default="supporting")
    source_path: str = Field(description="S3 URI / SQLite path / artifact this was read from.")
    bh_fdr_adjusted_p: float | None = Field(default=None)
    last_updated_utc: datetime
    derived_letter: str = Field(default="N/A", description="Summary letter; derived from status+value.")

    @property
    def is_na(self) -> bool:
        return self.status.startswith(_NA_PREFIX)

    @model_validator(mode="after")
    def _unit_required_when_value_present(self) -> "MetricRecord":
        """A numeric field with no unit is forbidden, never merely discouraged.

        Fires only when `value` is not None (an N/A-* record legitimately has
        neither a value nor a unit — there is nothing to be misread). This is
        the console-native report card's own unit contract
        (alpha-engine-config-I7477, console-policy.md §5.8: "`unit` is
        required" for a Signal's `value` field) enforced at the one chokepoint
        every producer already imports, rather than left to whichever
        consumer happens to check.
        """
        if self.value is not None and not self.unit:
            raise ValueError(
                f"MetricRecord {self.name!r} sets `value`={self.value!r} with "
                "no `unit`. A numeric value with no declared unit is not a "
                "measurement a consumer can safely render or compare."
            )
        return self


def derive_trend_decoration(
    values: list[float] | None,
    *,
    higher_is_better: bool = True,
) -> TrendDecorationLiteral:
    """Map a rolling value window to a trend glyph (RC v2 Principle 5).

    Looks at the last four points: ``↑↑`` if all 3 steps improved, ``↑`` if the
    most recent 2 of those steps improved, ``↓↓``/``↓`` mirrored, else ``→``.
    Improvement is "increase" when ``higher_is_better`` else "decrease".
    """
    if not values or len(values) < 2:
        return "→"
    window = values[-4:]
    steps = [b - a for a, b in zip(window, window[1:])]
    if not higher_is_better:
        steps = [-s for s in steps]

    eps = 1e-12
    ups = sum(1 for s in steps if s > eps)
    downs = sum(1 for s in steps if s < -eps)
    recent2 = steps[-2:]

    if len(steps) >= 3 and ups == len(steps):
        return "↑↑"
    if len(steps) >= 3 and downs == len(steps):
        return "↓↓"
    if sum(1 for s in recent2 if s > eps) == len(recent2) and ups >= downs:
        return "↑"
    if sum(1 for s in recent2 if s < -eps) == len(recent2) and downs >= ups:
        return "↓"
    return "→"


def derive_letter(status: StatusLiteral) -> str:
    """Project a status onto the summary letter band (RC v2 Principle 2).

    The letter is a display convenience only — ``status`` + ``value`` are the
    source of truth. Any ``N/A-*`` projects to ``"N/A"``.
    """
    if status.startswith(_NA_PREFIX):
        return "N/A"
    return {"GREEN": "A", "WATCH": "C", "RED": "F"}[status]


def derive_status(
    *,
    value: float | None,
    n_samples: int | None,
    n_floor: int,
    target: float | None = None,
    red_line: float | None = None,
    ci_low: float | None = None,
    ci_high: float | None = None,
    implemented: bool = True,
    ran: bool = True,
    input_present: bool = True,
) -> StatusLiteral:
    """Derive the GREEN/WATCH/RED/``N/A-*`` status for one component.

    Encodes RC v2 Principles 2 (status taxonomy) and 6 (sample-size discipline)
    so producer and consumers agree. Direction is inferred from the
    ``target``/``red_line`` ordering: ``target >= red_line`` ⇒ higher-is-better,
    else lower-is-better (e.g. max-drawdown, ECE).

    The four N/A conditions take precedence in order: not-implemented →
    not-run → missing-input → low-N. Above the floor, status follows the value
    and (when provided) the confidence interval relative to target/red-line.
    """
    if not implemented:
        return "N/A-NOT-IMPL"
    if not ran:
        return "N/A-NOT-RUN"
    if not input_present:
        return "N/A-MISSING-INPUT"
    if value is None or n_samples is None or n_samples < 0.5 * n_floor:
        return "N/A-LOW-N"

    higher_is_better = target is None or red_line is None or target >= red_line

    def _at_or_better(a: float, b: float) -> bool:
        return a >= b if higher_is_better else a <= b

    def _at_or_worse(a: float, b: float) -> bool:
        return a <= b if higher_is_better else a >= b

    # RED: value at/below the red-line, or the CI sits entirely on the bad side.
    if red_line is not None:
        if _at_or_worse(value, red_line):
            return "RED"
        bad_bound = ci_low if higher_is_better else ci_high
        if bad_bound is not None and _at_or_worse(bad_bound, red_line):
            return "RED"

    # Between half-floor and floor: CI too wide to claim GREEN regardless.
    if n_samples < n_floor:
        return "WATCH"

    if target is not None:
        if _at_or_better(value, target):
            # GREEN needs the whole CI clear of the red-line (when both known).
            good_bound = ci_low if higher_is_better else ci_high
            if red_line is not None and good_bound is not None and _at_or_worse(good_bound, red_line):
                return "WATCH"
            return "GREEN"
        return "WATCH"

    # No target given: above red-line with adequate N ⇒ GREEN.
    return "GREEN"
