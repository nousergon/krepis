"""session_date / assert_within_session — the event-time (trade-date) axis.

The invariant class under test: during a live session the event axis
(``session_date``) and the knowledge axis (``last_closed_trading_day``)
differ by exactly one session. Conflating them mis-keyed the executor's
trade/NAV artifacts and mis-joined the EOD reconcile (config#1610).
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from krepis.dates import (
    assert_within_session,
    last_closed_trading_day,
    previous_trading_day,
    session_date,
)

ET = ZoneInfo("America/New_York")


class TestSessionDate:
    def test_preopen_is_todays_session(self):
        # Monday 9:00 AM ET, pre-open: the upcoming Monday session.
        ts = datetime(2026, 6, 29, 9, 0, tzinfo=ET)
        assert session_date(ts) == date(2026, 6, 29)

    def test_intraday_is_todays_session(self):
        ts = datetime(2026, 6, 29, 13, 0, tzinfo=ET)
        assert session_date(ts) == date(2026, 6, 29)

    def test_at_close_inclusive(self):
        # Exactly 4:00 PM ET still belongs to the closing session.
        ts = datetime(2026, 6, 29, 16, 0, tzinfo=ET)
        assert session_date(ts) == date(2026, 6, 29)

    def test_postclose_is_next_session(self):
        # 4:05 PM ET Monday: after-close events print the next session.
        ts = datetime(2026, 6, 29, 16, 5, tzinfo=ET)
        assert session_date(ts) == date(2026, 6, 30)

    def test_saturday_maps_to_monday(self):
        ts = datetime(2026, 6, 27, 10, 0, tzinfo=ET)  # Saturday
        assert session_date(ts) == date(2026, 6, 29)

    def test_holiday_friday_maps_over_weekend(self):
        # Fri 2026-07-03 is the observed July-4 holiday → Mon 2026-07-06.
        ts = datetime(2026, 7, 3, 10, 0, tzinfo=ET)
        assert session_date(ts) == date(2026, 7, 6)

    def test_tz_aware_utc_converted_to_et(self):
        # 2026-07-02 13:31 UTC = 9:31 AM ET, intraday July 2 — the exact
        # timestamp shape from the mislabeled nav_series file.
        ts = datetime(2026, 7, 2, 13, 31, tzinfo=timezone.utc)
        assert session_date(ts) == date(2026, 7, 2)

    def test_naive_assumed_et(self):
        ts = datetime(2026, 6, 29, 10, 0)
        assert session_date(ts) == date(2026, 6, 29)

    def test_default_now_returns_date(self):
        assert isinstance(session_date(), date)

    def test_incident_shape_daemon_startup(self):
        # The config#1610 incident: daemon started intraday 2026-07-02
        # used last_closed (2026-07-01) as its run_date. session_date is
        # the correct axis and returns the physical session.
        startup = datetime(2026, 7, 2, 9, 45, tzinfo=ET)
        assert session_date(startup) == date(2026, 7, 2)
        assert last_closed_trading_day(startup) == date(2026, 7, 1)


class TestStrict:
    def test_strict_ok_preopen_and_intraday(self):
        assert session_date(
            datetime(2026, 6, 29, 9, 0, tzinfo=ET), strict=True
        ) == date(2026, 6, 29)
        assert session_date(
            datetime(2026, 6, 29, 12, 0, tzinfo=ET), strict=True
        ) == date(2026, 6, 29)

    def test_strict_raises_on_weekend(self):
        with pytest.raises(ValueError, match="does not fall within"):
            session_date(datetime(2026, 6, 27, 10, 0, tzinfo=ET), strict=True)

    def test_strict_raises_postclose(self):
        with pytest.raises(ValueError, match="does not fall within"):
            session_date(datetime(2026, 6, 29, 16, 5, tzinfo=ET), strict=True)

    def test_strict_raises_on_holiday(self):
        with pytest.raises(ValueError, match="does not fall within"):
            session_date(datetime(2026, 7, 3, 10, 0, tzinfo=ET), strict=True)


class TestAxisComplement:
    """During any live session the two axes are exactly one session apart."""

    @pytest.mark.parametrize(
        "ts",
        [
            datetime(2026, 6, 29, 9, 0, tzinfo=ET),   # Mon pre-open
            datetime(2026, 6, 30, 11, 0, tzinfo=ET),  # Tue intraday
            datetime(2026, 7, 2, 15, 59, tzinfo=ET),  # Thu pre-holiday
            datetime(2026, 7, 6, 9, 31, tzinfo=ET),   # Mon after holiday wknd
        ],
    )
    def test_session_is_one_after_last_closed(self, ts):
        assert previous_trading_day(session_date(ts)) == last_closed_trading_day(ts)

    def test_axes_coincide_only_at_close_boundary(self):
        # At exactly 16:00 ET both axes name the same session — the only
        # moment they agree.
        ts = datetime(2026, 6, 29, 16, 0, tzinfo=ET)
        assert session_date(ts) == last_closed_trading_day(ts)


class TestAssertWithinSession:
    def test_matching_label_passes(self):
        fill = datetime(2026, 7, 2, 14, 0, tzinfo=ET)
        assert_within_session(fill, date(2026, 7, 2))
        assert_within_session(fill, "2026-07-02")  # str label accepted

    def test_incident_mislabel_raises(self):
        # The live artifact from the incident: NAV point timestamped in
        # the physical July-2 session, labeled 2026-07-01.
        point = datetime(2026, 7, 2, 13, 31, tzinfo=timezone.utc)
        with pytest.raises(ValueError, match="mis-keyed"):
            assert_within_session(point, "2026-07-01")

    def test_postclose_write_labeled_same_day_raises(self):
        # A 4:10 PM write labeled with the just-closed session is a
        # mis-key under the close-partition definition.
        ts = datetime(2026, 6, 29, 16, 10, tzinfo=ET)
        with pytest.raises(ValueError, match="mis-keyed"):
            assert_within_session(ts, date(2026, 6, 29))
