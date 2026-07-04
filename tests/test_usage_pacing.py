"""Tests for krepis.usage_pacing — linear-pace quota short-circuit."""

from datetime import datetime, timedelta

import pytest

from krepis.usage_pacing import PaceStatus, elapsed_fraction, pace_check, reset_window

ANCHOR = datetime(2026, 6, 28, 20, 59)   # one observed weekly-reset instant, PT-naive
PERIOD = timedelta(days=7)


class TestResetWindow:
    def test_at_anchor_instant(self):
        start, next_reset = reset_window(ANCHOR, ANCHOR, PERIOD)
        assert start == ANCHOR
        assert next_reset == ANCHOR + PERIOD

    def test_mid_window(self):
        now = ANCHOR + timedelta(days=3, hours=2)
        start, next_reset = reset_window(now, ANCHOR, PERIOD)
        assert start == ANCHOR
        assert next_reset == ANCHOR + PERIOD

    def test_just_before_next_reset(self):
        now = ANCHOR + PERIOD - timedelta(minutes=1)
        start, next_reset = reset_window(now, ANCHOR, PERIOD)
        assert start == ANCHOR
        assert next_reset == ANCHOR + PERIOD

    def test_exactly_at_next_reset_rolls_over(self):
        now = ANCHOR + PERIOD
        start, next_reset = reset_window(now, ANCHOR, PERIOD)
        assert start == ANCHOR + PERIOD
        assert next_reset == ANCHOR + 2 * PERIOD

    def test_before_anchor_walks_back(self):
        now = ANCHOR - timedelta(days=2)
        start, next_reset = reset_window(now, ANCHOR, PERIOD)
        assert start == ANCHOR - PERIOD
        assert next_reset == ANCHOR

    def test_several_periods_forward(self):
        now = ANCHOR + 3 * PERIOD + timedelta(hours=5)
        start, next_reset = reset_window(now, ANCHOR, PERIOD)
        assert start == ANCHOR + 3 * PERIOD
        assert next_reset == ANCHOR + 4 * PERIOD


class TestElapsedFraction:
    def test_at_reset_is_zero(self):
        assert elapsed_fraction(ANCHOR, ANCHOR, PERIOD) == 0.0

    def test_halfway_through_week(self):
        now = ANCHOR + timedelta(days=3, hours=12)
        assert elapsed_fraction(now, ANCHOR, PERIOD) == 0.5

    def test_saturday_9pm_is_144_of_168_hours(self):
        # Sunday 8:59pm PT reset -> Saturday 8:59pm PT is 144h/168h = 6/7 elapsed.
        now = ANCHOR + timedelta(days=6)
        assert abs(elapsed_fraction(now, ANCHOR, PERIOD) - (144 / 168)) < 1e-9

    def test_clamped_to_one_at_next_reset(self):
        now = ANCHOR + PERIOD
        # exactly at next reset rolls into the NEW window (0.0), never 1.0+
        assert elapsed_fraction(now, ANCHOR, PERIOD) == 0.0

    def test_never_negative(self):
        now = ANCHOR - timedelta(hours=1)
        assert elapsed_fraction(now, ANCHOR, PERIOD) >= 0.0


class TestPaceCheck:
    def test_on_pace_not_exceeded(self):
        now = ANCHOR + timedelta(days=3, hours=12)  # elapsed_frac == 0.5
        status = pace_check(used_frac=0.4, now=now, anchor=ANCHOR, period=PERIOD)
        assert isinstance(status, PaceStatus)
        assert status.elapsed_frac == 0.5
        assert status.used_frac == 0.4
        assert status.overrun == pytest.approx(-0.1)
        assert status.exceeded is False

    def test_exactly_on_pace_not_exceeded(self):
        # used_frac == elapsed_frac is NOT exceeded (strict >, not >=).
        now = ANCHOR + timedelta(days=3, hours=12)
        status = pace_check(used_frac=0.5, now=now, anchor=ANCHOR, period=PERIOD)
        assert status.exceeded is False
        assert status.overrun == 0.0

    def test_ahead_of_pace_exceeded(self):
        now = ANCHOR + timedelta(days=1)  # elapsed_frac ~= 1/7
        status = pace_check(used_frac=0.5, now=now, anchor=ANCHOR, period=PERIOD)
        assert status.exceeded is True
        assert status.overrun > 0

    def test_saturday_evening_85_percent_used_exceeds_144_168_pace(self):
        now = ANCHOR + timedelta(days=6)  # 144/168 = ~0.857 elapsed
        status = pace_check(used_frac=0.90, now=now, anchor=ANCHOR, period=PERIOD)
        assert status.exceeded is True

    def test_saturday_evening_80_percent_used_is_under_144_168_pace(self):
        now = ANCHOR + timedelta(days=6)  # ~0.857 elapsed
        status = pace_check(used_frac=0.80, now=now, anchor=ANCHOR, period=PERIOD)
        assert status.exceeded is False
