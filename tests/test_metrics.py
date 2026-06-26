"""Tests for krepis.metrics — MetricRecord contract + status derivation."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from krepis.metrics import (
    MetricRecord,
    derive_letter,
    derive_status,
    derive_trend_decoration,
)


def _record(**overrides):
    base = dict(
        name="predictor_meta_l2_ic",
        module="predictor",
        metric_type="ic",
        n_floor=100,
        status="GREEN",
        status_reason="L2 IC 0.48 (CI [0.21,0.74], N=626) above target 0.05.",
        source_path="s3://alpha-engine-research/predictor/metrics/latest.json#l2_ic",
        last_updated_utc=datetime(2026, 6, 4, tzinfo=timezone.utc),
    )
    base.update(overrides)
    return MetricRecord(**base)


class TestMetricRecord:
    def test_minimal_valid_record(self):
        r = _record(value=0.48, n_samples=626)
        assert r.module == "predictor" and r.value == 0.48
        assert r.trend_decoration == "→"  # default
        assert r.is_na is False

    def test_na_status_flags_is_na(self):
        assert _record(status="N/A-LOW-N").is_na is True

    def test_extra_fields_allowed(self):
        r = _record(value=0.1, n_samples=200, future_field="ok")
        assert r.value == 0.1

    def test_bad_status_rejected(self):
        with pytest.raises(ValidationError):
            _record(status="MAYBE")

    def test_bad_metric_type_rejected(self):
        with pytest.raises(ValidationError):
            _record(metric_type="vibes")


class TestDeriveTrendDecoration:
    def test_sustained_up_and_down(self):
        assert derive_trend_decoration([1, 2, 3, 4]) == "↑↑"
        assert derive_trend_decoration([4, 3, 2, 1]) == "↓↓"

    def test_flat_and_too_short(self):
        assert derive_trend_decoration([2, 2, 2, 2]) == "→"
        assert derive_trend_decoration([1]) == "→"
        assert derive_trend_decoration(None) == "→"

    def test_recent_improvement(self):
        # down then up-up: not sustained, but recent 2 improved.
        assert derive_trend_decoration([5, 1, 2, 3]) == "↑"

    def test_higher_is_better_false_flips(self):
        # decreasing values are an improvement when lower is better (e.g. drawdown).
        assert derive_trend_decoration([4, 3, 2, 1], higher_is_better=False) == "↑↑"


class TestDeriveLetter:
    def test_status_to_letter(self):
        assert derive_letter("GREEN") == "A"
        assert derive_letter("WATCH") == "C"
        assert derive_letter("RED") == "F"
        assert derive_letter("N/A-NOT-IMPL") == "N/A"


class TestDeriveStatus:
    def test_na_precedence(self):
        assert derive_status(value=0.5, n_samples=200, n_floor=100, implemented=False) == "N/A-NOT-IMPL"
        assert derive_status(value=0.5, n_samples=200, n_floor=100, ran=False) == "N/A-NOT-RUN"
        assert derive_status(value=0.5, n_samples=200, n_floor=100, input_present=False) == "N/A-MISSING-INPUT"

    def test_low_n(self):
        assert derive_status(value=0.5, n_samples=40, n_floor=100) == "N/A-LOW-N"
        assert derive_status(value=None, n_samples=200, n_floor=100) == "N/A-LOW-N"

    def test_red_when_at_or_below_red_line(self):
        s = derive_status(value=-0.01, n_samples=200, n_floor=100, target=0.05, red_line=0.0)
        assert s == "RED"

    def test_red_when_ci_entirely_below_red_line(self):
        s = derive_status(
            value=0.02, n_samples=200, n_floor=100, target=0.05, red_line=0.0,
            ci_low=-0.03, ci_high=-0.01,
        )
        assert s == "RED"

    def test_green_when_above_target_with_clear_ci(self):
        s = derive_status(
            value=0.48, n_samples=626, n_floor=100, target=0.05, red_line=0.0,
            ci_low=0.21, ci_high=0.74,
        )
        assert s == "GREEN"

    def test_watch_between_half_floor_and_floor(self):
        # N=70 is above 0.5*floor (50) but below floor (100) → WATCH regardless of value.
        s = derive_status(value=0.9, n_samples=70, n_floor=100, target=0.05, red_line=0.0)
        assert s == "WATCH"

    def test_watch_below_target_above_red_line(self):
        s = derive_status(value=0.02, n_samples=200, n_floor=100, target=0.05, red_line=0.0)
        assert s == "WATCH"

    def test_lower_is_better_direction(self):
        # max_drawdown: target 0.15, red_line 0.25 (lower is better).
        assert derive_status(value=0.08, n_samples=200, n_floor=60, target=0.15, red_line=0.25) == "GREEN"
        assert derive_status(value=0.30, n_samples=200, n_floor=60, target=0.15, red_line=0.25) == "RED"
        assert derive_status(value=0.20, n_samples=200, n_floor=60, target=0.15, red_line=0.25) == "WATCH"
