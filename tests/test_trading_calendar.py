"""Unit tests for krepis.trading_calendar."""
from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

import pytest

from krepis.trading_calendar import (
    NYSE_HOLIDAYS,
    add_trading_days,
    count_trading_days,
    is_market_hours,
    is_trading_day,
    next_trading_day,
    previous_trading_day,
    subtract_trading_days,
)

_ET = ZoneInfo("America/New_York")


class TestIsTradingDay:
    def test_regular_weekday(self):
        assert is_trading_day(date(2026, 4, 16)) is True  # Thursday

    def test_weekend_saturday(self):
        assert is_trading_day(date(2026, 4, 18)) is False

    def test_weekend_sunday(self):
        assert is_trading_day(date(2026, 4, 19)) is False

    def test_new_years_day(self):
        assert is_trading_day(date(2026, 1, 1)) is False

    def test_good_friday_2026(self):
        assert is_trading_day(date(2026, 4, 3)) is False

    def test_independence_day_observed_2026(self):
        """2026 July 4 is a Saturday; observed on Friday July 3."""
        assert is_trading_day(date(2026, 7, 3)) is False
        assert is_trading_day(date(2026, 7, 2)) is True


class TestNextTradingDay:
    def test_skips_weekend(self):
        assert next_trading_day(date(2026, 4, 17)) == date(2026, 4, 20)  # Fri → Mon

    def test_skips_holiday(self):
        assert next_trading_day(date(2026, 4, 2)) == date(2026, 4, 6)

    def test_consecutive_trading_days(self):
        assert next_trading_day(date(2026, 4, 15)) == date(2026, 4, 16)


class TestHolidayCoverage:
    def test_covers_through_2030(self):
        assert {d.year for d in NYSE_HOLIDAYS} >= {2025, 2026, 2027, 2028, 2029, 2030}


class TestPreviousTradingDay:
    def test_skips_weekend(self):
        # Mon 4/20 → Fri 4/17
        assert previous_trading_day(date(2026, 4, 20)) == date(2026, 4, 17)

    def test_skips_good_friday_2026(self):
        # Mon 4/6 → Thu 4/2 (skip Fri 4/3 holiday + weekend)
        assert previous_trading_day(date(2026, 4, 6)) == date(2026, 4, 2)


class TestAddTradingDays:
    def test_zero_returns_start(self):
        assert add_trading_days(date(2026, 4, 17), 0) == date(2026, 4, 17)
        # Even when start is itself not a trading day, n=0 is a no-op.
        assert add_trading_days(date(2026, 4, 18), 0) == date(2026, 4, 18)

    def test_skips_weekend(self):
        assert add_trading_days(date(2026, 4, 17), 1) == date(2026, 4, 20)

    def test_skips_good_friday_2026(self):
        # Thu 4/2 + 5 trading days = Fri 4/10 (skip Fri 4/3 Good Friday).
        # Calendar BD would have wrongly returned 4/9.
        assert add_trading_days(date(2026, 4, 2), 5) == date(2026, 4, 10)

    def test_skips_thanksgiving_2025(self):
        assert add_trading_days(date(2025, 11, 26), 1) == date(2025, 11, 28)

    def test_negative_n_raises(self):
        with pytest.raises(ValueError):
            add_trading_days(date(2026, 4, 17), -1)


class TestSubtractTradingDays:
    def test_zero_returns_start(self):
        assert subtract_trading_days(date(2026, 4, 17), 0) == date(2026, 4, 17)

    def test_skips_weekend(self):
        # Mon 4/20 - 1 = Fri 4/17
        assert subtract_trading_days(date(2026, 4, 20), 1) == date(2026, 4, 17)

    def test_skips_good_friday_2026(self):
        # Mon 4/6 - 1 = Thu 4/2 (skip Fri 4/3 holiday)
        assert subtract_trading_days(date(2026, 4, 6), 1) == date(2026, 4, 2)

    def test_round_trip_inverse_of_add(self):
        d = date(2026, 4, 2)  # Thu before Good Friday
        for n in range(0, 30):
            assert subtract_trading_days(add_trading_days(d, n), n) == d

    def test_negative_n_raises(self):
        with pytest.raises(ValueError):
            subtract_trading_days(date(2026, 4, 17), -1)


class TestCountTradingDays:
    def test_end_le_start_returns_zero(self):
        d = date(2026, 4, 17)
        assert count_trading_days(d, d) == 0
        assert count_trading_days(d, date(2026, 4, 16)) == 0

    def test_skips_weekend(self):
        # Fri 4/17 → Mon 4/20: 1 trading day (just 4/20)
        assert count_trading_days(date(2026, 4, 17), date(2026, 4, 20)) == 1

    def test_skips_good_friday_2026(self):
        # Thu 4/2 → Mon 4/6: 1 trading day (just 4/6, skip Fri 4/3 holiday)
        assert count_trading_days(date(2026, 4, 2), date(2026, 4, 6)) == 1

    def test_inverse_of_add(self):
        # count(start, add(start, n)) == n for any non-trading-day start.
        for start in [date(2026, 4, 17), date(2026, 4, 18), date(2026, 4, 3)]:
            for n in [0, 1, 5, 10, 30]:
                assert count_trading_days(start, add_trading_days(start, n)) == n


@pytest.mark.parametrize(
    "eval_date,horizon,expected",
    [
        # Good Friday 2026 window
        (date(2026, 4, 2), 1, date(2026, 4, 6)),
        (date(2026, 4, 2), 5, date(2026, 4, 10)),
        # Thanksgiving 2025
        (date(2025, 11, 25), 5, date(2025, 12, 3)),
        # New Year 2026
        (date(2025, 12, 31), 1, date(2026, 1, 2)),
    ],
)
def test_add_trading_days_table(eval_date, horizon, expected):
    assert add_trading_days(eval_date, horizon) == expected


class TestIsMarketHours:
    """alpha-engine-config-I7111 — the session predicate lifted out of
    ``crucible-executor/executor/market_hours.py``.

    The boundary cases are the point: two live Step Functions pipelines gate
    on this, and the postclose one is *triggered at* 16:00:0x ET.
    """

    # 2026-08-12 is a Wednesday and not an NYSE holiday.
    _WED = date(2026, 8, 12)

    def test_midsession_is_open(self):
        assert is_market_hours(datetime(2026, 8, 12, 12, 0, tzinfo=_ET)) is True

    def test_open_instant_is_inclusive(self):
        assert is_market_hours(datetime(2026, 8, 12, 9, 30, 0, tzinfo=_ET)) is True

    def test_one_second_before_open_is_closed(self):
        assert is_market_hours(datetime(2026, 8, 12, 9, 29, 59, tzinfo=_ET)) is False

    def test_close_instant_is_exclusive(self):
        # Load-bearing: ne-postclose-trading-pipeline's daemon-shutdown
        # trigger lands at 16:00:0x ET. A close-INCLUSIVE boundary would
        # refuse the settlement run that carries NAV continuity.
        assert is_market_hours(datetime(2026, 8, 12, 16, 0, 0, tzinfo=_ET)) is False

    def test_one_second_before_close_is_open(self):
        assert is_market_hours(datetime(2026, 8, 12, 15, 59, 59, tzinfo=_ET)) is True

    def test_preopen_is_closed(self):
        # ne-preopen-trading-pipeline's 05:15 PT cron == 08:15 ET.
        assert is_market_hours(datetime(2026, 8, 12, 8, 15, tzinfo=_ET)) is False

    def test_postclose_settlement_window_is_closed(self):
        # Every observed eod-* execution start, 2026-07-08..2026-08-12.
        assert is_market_hours(datetime(2026, 8, 12, 16, 0, 4, tzinfo=_ET)) is False
        assert is_market_hours(datetime(2026, 8, 12, 16, 0, 56, tzinfo=_ET)) is False

    def test_weekend_midsession_clock_is_closed(self):
        assert is_market_hours(datetime(2026, 8, 15, 12, 0, tzinfo=_ET)) is False  # Sat
        assert is_market_hours(datetime(2026, 8, 16, 12, 0, tzinfo=_ET)) is False  # Sun

    def test_holiday_midsession_clock_is_closed(self):
        # Good Friday 2026 — a weekday whose wall clock is inside the window.
        assert date(2026, 4, 3) in NYSE_HOLIDAYS
        assert is_market_hours(datetime(2026, 4, 3, 12, 0, tzinfo=_ET)) is False

    def test_naive_datetime_is_read_as_eastern(self):
        assert is_market_hours(datetime(2026, 8, 12, 12, 0)) is True
        assert is_market_hours(datetime(2026, 8, 12, 8, 0)) is False

    def test_utc_input_is_converted_not_compared_raw(self):
        # 16:00 UTC on 2026-08-12 (EDT) == 12:00 ET -> open. Comparing the
        # raw UTC clock against 09:30-16:00 would call it closed.
        assert is_market_hours(datetime(2026, 8, 12, 16, 0, tzinfo=timezone.utc)) is True
        # 13:00 UTC == 09:00 ET -> still pre-open.
        assert is_market_hours(datetime(2026, 8, 12, 13, 0, tzinfo=timezone.utc)) is False

    def test_dst_is_handled_by_the_zone_not_a_fixed_offset(self):
        # January (EST, UTC-5): 14:30 UTC == 09:30 ET -> open at the bell.
        assert is_market_hours(datetime(2027, 1, 5, 14, 30, tzinfo=timezone.utc)) is True
        assert is_market_hours(datetime(2027, 1, 5, 14, 29, tzinfo=timezone.utc)) is False
        # July (EDT, UTC-4): 14:30 UTC == 10:30 ET -> open; 13:30 == 09:30.
        assert is_market_hours(datetime(2027, 7, 6, 13, 30, tzinfo=timezone.utc)) is True
        assert is_market_hours(datetime(2027, 7, 6, 13, 29, tzinfo=timezone.utc)) is False

    def test_close_override_is_read_per_call(self):
        at_1545 = datetime(2026, 8, 12, 15, 45, tzinfo=_ET)
        assert is_market_hours(at_1545) is True
        assert is_market_hours(at_1545, close_et=time(15, 30)) is False

    def test_open_override_is_read_per_call(self):
        at_0900 = datetime(2026, 8, 12, 9, 0, tzinfo=_ET)
        assert is_market_hours(at_0900) is False
        assert is_market_hours(at_0900, open_et=time(8, 0)) is True

    def test_no_argument_call_reads_the_eastern_clock(self):
        # Guards the default branch (``now is None``), where a regression
        # that skipped the tz attach would compare a naive UTC clock
        # against the ET window and be wrong by 4-5 hours.
        assert is_market_hours() is is_market_hours(datetime.now(_ET))

    def test_unused_holiday_import_is_the_single_table(self):
        # The executor's duplicate copy is deleted in crucible-executor;
        # this asserts the surviving table is the one this module owns.
        assert date(2026, 11, 26) in NYSE_HOLIDAYS  # Thanksgiving
        assert is_market_hours(datetime(2026, 11, 26, 12, 0, tzinfo=_ET)) is False
