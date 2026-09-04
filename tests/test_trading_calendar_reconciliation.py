"""Reconcile krepis's static NYSE calendar against exchange_calendars.

alpha-engine-config-I9998: the static table in trading_calendar.py is a
hand-maintained holiday set with no mechanism forcing it to stay right. This
test is that mechanism — it fails CI the moment ``NYSE_HOLIDAYS`` or
``NYSE_EARLY_CLOSES`` diverges from an authoritative source over the whole
declared coverage range, rather than relying on someone noticing.

``exchange_calendars`` is a dev-only dependency (see pyproject.toml) — it is
never installed at runtime, so it does not reach any consumer that
``pip install``s krepis for an unrelated primitive. It also requires Python
>=3.10, so this module ``importorskip``s on the 3.9 leg of the CI matrix
(the same pattern already used for the ``litellm`` cap in ``dev`` deps) and
runs for real on the other four declared interpreters.

This test is what actually FOUND the missing July-3rd early closes during
authoring (day before Independence Day, when July 4 falls Tue-Fri) — they
were not in the hand-built table until this script pointed at them.
"""
from __future__ import annotations

from datetime import date, time, timedelta

import pytest

xcals = pytest.importorskip("exchange_calendars")

from krepis.trading_calendar import (  # noqa: E402
    NYSE_CALENDAR_COVERS_THROUGH,
    NYSE_EARLY_CLOSES,
    NYSE_HOLIDAYS,
)

_RANGE_START = date(2025, 1, 1)
_ET = __import__("zoneinfo").ZoneInfo("America/New_York")


def _reference_schedule():
    cal = xcals.get_calendar(
        "XNYS",
        start=_RANGE_START.isoformat(),
        end=NYSE_CALENDAR_COVERS_THROUGH.isoformat(),
    )
    return cal.schedule


def test_holiday_table_matches_reference_calendar():
    schedule = _reference_schedule()
    reference_sessions = set(schedule.index.date)

    mismatches = []
    d = _RANGE_START
    while d <= NYSE_CALENDAR_COVERS_THROUGH:
        if d.weekday() < 5:  # weekday only — NYSE_HOLIDAYS never claims weekends
            reference_open = d in reference_sessions
            ours_open = d not in NYSE_HOLIDAYS
            if reference_open != ours_open:
                mismatches.append((d, "reference_open=%s ours_open=%s" % (reference_open, ours_open)))
        d += timedelta(days=1)

    assert mismatches == [], (
        f"NYSE_HOLIDAYS diverges from exchange_calendars on {len(mismatches)} "
        f"weekday(s) in [{_RANGE_START}, {NYSE_CALENDAR_COVERS_THROUGH}]: "
        f"{mismatches[:10]}"
    )


def test_early_close_table_matches_reference_calendar():
    schedule = _reference_schedule()

    mismatches = []
    for idx, row in schedule.iterrows():
        d = idx.date()
        close_et = row["close"].tz_convert(_ET)
        reference_early = close_et.time() < time(16, 0)
        ours_early = d in NYSE_EARLY_CLOSES
        if reference_early != ours_early:
            mismatches.append((d, "reference_early=%s ours_early=%s close=%s" % (reference_early, ours_early, close_et.time())))

    assert mismatches == [], (
        f"NYSE_EARLY_CLOSES diverges from exchange_calendars on "
        f"{len(mismatches)} session(s): {mismatches[:10]}"
    )
