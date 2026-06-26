"""Unit tests for krepis.trading_calendar."""
from datetime import date

import pytest

from krepis.trading_calendar import (
    NYSE_HOLIDAYS,
    add_trading_days,
    count_trading_days,
    is_trading_day,
    next_trading_day,
    previous_trading_day,
    subtract_trading_days,
)


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
