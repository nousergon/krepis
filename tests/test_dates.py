"""Tests for krepis.dates — dual-tracking date convention."""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from krepis.dates import DualDate, now_dual, session_for_timestamp


_NYSE = ZoneInfo("America/New_York")


# Reference dates used throughout these tests:
#   Fri 2026-04-24 — trading day
#   Sat 2026-04-25, Sun 2026-04-26 — weekend
#   Mon 2026-04-27 — trading day
#   Mon 2026-01-19 — MLK Day (holiday)
#   Fri 2026-04-03 — Good Friday (holiday)
#   Thu 2026-04-02 — trading day before Good Friday


class TestNowDualSpecTable:
    """now_dual() output matches the worked-examples table in
    alpha-engine-docs/private/DATE_CONVENTIONS.md verbatim."""

    def test_saturday_attributes_to_prior_friday(self):
        # Sat 00:00 UTC = Fri 8 PM ET — Fri's session closed at 4 PM ET.
        sat = datetime(2026, 4, 25, 0, 0, tzinfo=timezone.utc)
        dd = now_dual(now=sat)
        assert dd.trading_day == "2026-04-24"

    def test_sunday_attributes_to_prior_friday(self):
        sun = datetime(2026, 4, 26, 14, 0, tzinfo=timezone.utc)
        dd = now_dual(now=sun)
        assert dd.trading_day == "2026-04-24"

    def test_monday_pre_open_attributes_to_prior_friday(self):
        # Mon 9:15 AM ET — Mon's session not yet closed.
        mon_pre = datetime(2026, 4, 27, 9, 15, tzinfo=_NYSE)
        dd = now_dual(now=mon_pre)
        assert dd.trading_day == "2026-04-24"

    def test_monday_mid_session_attributes_to_prior_friday(self):
        # Mon 2 PM ET — session in progress, not completed.
        mon_mid = datetime(2026, 4, 27, 14, 0, tzinfo=_NYSE)
        dd = now_dual(now=mon_mid)
        assert dd.trading_day == "2026-04-24"

    def test_monday_post_close_attributes_to_monday(self):
        # Mon 5 PM ET — Mon's session closed at 4 PM ET.
        mon_post = datetime(2026, 4, 27, 17, 0, tzinfo=_NYSE)
        dd = now_dual(now=mon_post)
        assert dd.trading_day == "2026-04-27"

    def test_mlk_holiday_post_close_still_attributes_to_prior_friday(self):
        # MLK Mon — even post-close, no session occurred. Walk back.
        mlk = datetime(2026, 1, 19, 17, 0, tzinfo=_NYSE)
        dd = now_dual(now=mlk)
        assert dd.trading_day == "2026-01-16"

    def test_good_friday_attributes_to_prior_thursday(self):
        gf = datetime(2026, 4, 3, 12, 0, tzinfo=_NYSE)
        dd = now_dual(now=gf)
        assert dd.trading_day == "2026-04-02"


class TestNowDualCalendarDate:
    """calendar_date is UTC; trading_day is NYSE-local."""

    def test_calendar_date_is_utc(self):
        # Sat 00:00 UTC = Fri 8 PM ET. calendar_date should be Sat (UTC),
        # trading_day should be Fri (last completed session).
        sat_utc = datetime(2026, 4, 25, 0, 0, tzinfo=timezone.utc)
        dd = now_dual(now=sat_utc)
        assert dd.calendar_date == "2026-04-25"
        assert dd.trading_day == "2026-04-24"

    def test_calendar_date_does_not_walk_back(self):
        # Even on weekends/holidays, calendar_date stays as the wall-clock UTC date.
        sun = datetime(2026, 4, 26, 12, 0, tzinfo=timezone.utc)
        dd = now_dual(now=sun)
        assert dd.calendar_date == "2026-04-26"  # Sunday — unchanged

    def test_calendar_date_iso_format(self):
        ts = datetime(2026, 4, 27, 17, 0, tzinfo=_NYSE)
        dd = now_dual(now=ts)
        # Must be 10-char ISO yyyy-mm-dd
        assert len(dd.calendar_date) == 10
        assert dd.calendar_date.count("-") == 2


class TestNowDualInputHandling:
    """Input handling — defaults, naive datetime."""

    def test_default_now_returns_dual_date(self):
        dd = now_dual()
        assert isinstance(dd, DualDate)
        assert len(dd.calendar_date) == 10
        assert len(dd.trading_day) == 10

    def test_naive_datetime_assumed_utc(self):
        naive = datetime(2026, 4, 25, 0, 0)
        aware = datetime(2026, 4, 25, 0, 0, tzinfo=timezone.utc)
        assert now_dual(now=naive).trading_day == now_dual(now=aware).trading_day
        assert now_dual(now=naive).calendar_date == now_dual(now=aware).calendar_date

    def test_dual_date_is_frozen(self):
        from dataclasses import FrozenInstanceError
        dd = now_dual()
        with pytest.raises(FrozenInstanceError):
            dd.calendar_date = "2099-01-01"  # type: ignore


class TestSessionForTimestamp:
    """session_for_timestamp() backfill helper."""

    def test_returns_iso_date_string(self):
        ts = datetime(2026, 4, 27, 17, 0, tzinfo=_NYSE)
        result = session_for_timestamp(ts)
        assert result == "2026-04-27"
        assert len(result) == 10
        assert result.count("-") == 2

    def test_pre_open_timestamp_walks_back(self):
        # Mon 9 AM ET pre-open → prior Friday.
        ts = datetime(2026, 4, 27, 9, 0, tzinfo=_NYSE)
        assert session_for_timestamp(ts) == "2026-04-24"

    def test_post_close_timestamp_returns_today(self):
        ts = datetime(2026, 4, 27, 17, 0, tzinfo=_NYSE)
        assert session_for_timestamp(ts) == "2026-04-27"

    def test_naive_assumed_utc(self):
        naive = datetime(2026, 4, 25, 12, 0)
        aware = datetime(2026, 4, 25, 12, 0, tzinfo=timezone.utc)
        assert session_for_timestamp(naive) == session_for_timestamp(aware)

    def test_mlk_holiday_walks_back(self):
        ts = datetime(2026, 1, 19, 12, 0, tzinfo=_NYSE)
        assert session_for_timestamp(ts) == "2026-01-16"

    def test_good_friday_walks_back_to_thursday(self):
        ts = datetime(2026, 4, 3, 12, 0, tzinfo=_NYSE)
        assert session_for_timestamp(ts) == "2026-04-02"

    def test_weekend_walks_back(self):
        ts = datetime(2026, 4, 26, 14, 0, tzinfo=timezone.utc)  # Sun
        assert session_for_timestamp(ts) == "2026-04-24"  # Fri


class TestNowDualMatchesSessionForTimestamp:
    """The two helpers should agree on trading_day for any given moment —
    they both wrap last_closed_trading_day. Drift between them would be a bug."""

    def test_agreement_on_pre_open_monday(self):
        ts = datetime(2026, 4, 27, 9, 15, tzinfo=_NYSE)
        assert now_dual(now=ts).trading_day == session_for_timestamp(ts)

    def test_agreement_on_post_close_monday(self):
        ts = datetime(2026, 4, 27, 17, 0, tzinfo=_NYSE)
        assert now_dual(now=ts).trading_day == session_for_timestamp(ts)

    def test_agreement_on_mlk_holiday(self):
        ts = datetime(2026, 1, 19, 12, 0, tzinfo=_NYSE)
        assert now_dual(now=ts).trading_day == session_for_timestamp(ts)


# ── Freshness helpers (trading-day-aware) ────────────────────────────────────


from datetime import date as _date  # noqa: E402

from krepis.dates import (  # noqa: E402
    expected_last_close,
    is_fresh_in_trading_days,
    trading_days_stale,
)


class TestExpectedLastClose:
    """``expected_last_close(run_date)`` returns the most recent NYSE close
    that exists as of run_date — Friday on weekends, the day itself on a
    settled trading day, the prior trading day on a holiday."""

    def test_saturday_returns_friday(self):
        assert expected_last_close(_date(2026, 5, 23)) == _date(2026, 5, 22)

    def test_sunday_returns_friday(self):
        # The 2026-05-24 SF recovery's exact reference point.
        assert expected_last_close(_date(2026, 5, 24)) == _date(2026, 5, 22)

    def test_memorial_day_monday_returns_prior_friday(self):
        # Memorial Day 2026 = Mon 5/25 (NYSE closed).
        assert expected_last_close(_date(2026, 5, 25)) == _date(2026, 5, 22)

    def test_tuesday_after_memorial_day_returns_tuesday(self):
        # Tuesday 5/26 — its own close has settled (we anchor at 23 UTC).
        assert expected_last_close(_date(2026, 5, 26)) == _date(2026, 5, 26)

    def test_accepts_iso_string(self):
        assert expected_last_close("2026-05-24") == _date(2026, 5, 22)

    def test_good_friday_walks_back_to_thursday(self):
        # Good Friday 2026 = Apr 3 (NYSE closed).
        assert expected_last_close(_date(2026, 4, 3)) == _date(2026, 4, 2)


class TestTradingDaysStale:
    """``trading_days_stale(last_date, reference)`` counts NYSE sessions
    between last_date (exclusive) and the expected last close of reference
    (inclusive). Zero when last_date carries the most recent available close."""

    def test_saturday_redrive_friday_macro_is_zero(self):
        # Original Saturday-SF semantic: Fri close on Sat run_date.
        assert trading_days_stale(_date(2026, 5, 22), _date(2026, 5, 23)) == 0

    def test_sunday_redrive_friday_macro_is_zero(self):
        # The exact case that broke calendar-day arithmetic on 2026-05-24.
        # Fri→Sun = 2 calendar days, 0 trading days.
        assert trading_days_stale(_date(2026, 5, 22), _date(2026, 5, 24)) == 0

    def test_memorial_day_monday_friday_macro_is_zero(self):
        # Fri→Memorial-Mon = 3 calendar days, 0 trading days.
        assert trading_days_stale(_date(2026, 5, 22), _date(2026, 5, 25)) == 0

    def test_tuesday_after_memorial_day_expects_tuesday_close(self):
        # Tuesday's close settled by 23 UTC ⇒ Friday macro is 1 session behind.
        assert trading_days_stale(_date(2026, 5, 22), _date(2026, 5, 26)) == 1

    def test_genuinely_stale_returns_session_count(self):
        # Wed 5/13 → Fri 5/22 = 7 trading days (skipping the Sat/Sun weekend).
        assert trading_days_stale(_date(2026, 5, 13), _date(2026, 5, 22)) == 7

    def test_future_last_date_returns_zero(self):
        # Defensive: last_date ahead of reference is treated as "fresh."
        assert trading_days_stale(_date(2026, 5, 30), _date(2026, 5, 22)) == 0

    def test_accepts_iso_string_reference(self):
        assert trading_days_stale(_date(2026, 5, 22), "2026-05-24") == 0


class TestIsFreshInTradingDays:
    """``is_fresh_in_trading_days(last_date, reference, max_stale=N)`` is the
    canonical freshness predicate. Default ``max_stale=0`` means "must carry
    the most recent NYSE close that exists." Larger max_stale tolerates
    publish-latency in consumer-side preflights (polygon T+1)."""

    def test_default_strict_passes_on_sunday_with_friday_macro(self):
        # The 2026-05-24 SF recovery: must pass under the new trading-day gate.
        assert is_fresh_in_trading_days(_date(2026, 5, 22), _date(2026, 5, 24))

    def test_default_strict_fails_when_one_session_behind(self):
        # macro at Wed close, asking for Thursday's expected close.
        assert not is_fresh_in_trading_days(_date(2026, 5, 13), _date(2026, 5, 14))

    def test_max_stale_one_tolerates_t_plus_one_lag(self):
        # Consumer-preflight pattern: tolerate 1-session lag for T+1 publish.
        assert is_fresh_in_trading_days(
            _date(2026, 5, 13), _date(2026, 5, 14), max_stale=1,
        )

    def test_max_stale_keyword_only(self):
        import pytest
        with pytest.raises(TypeError):
            is_fresh_in_trading_days(  # type: ignore[call-arg]
                _date(2026, 5, 22), _date(2026, 5, 24), 1,
            )

    def test_holiday_aware_via_nyse_calendar(self):
        # Tue 5/26 — its own close has settled at 23 UTC, so Friday's macro
        # is 1 session behind. max_stale=0 fails, max_stale=1 passes.
        assert not is_fresh_in_trading_days(_date(2026, 5, 22), _date(2026, 5, 26))
        assert is_fresh_in_trading_days(
            _date(2026, 5, 22), _date(2026, 5, 26), max_stale=1,
        )
