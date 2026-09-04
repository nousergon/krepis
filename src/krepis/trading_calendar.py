"""
trading_calendar.py — NYSE trading day check with holiday awareness.

Lightweight implementation that doesn't require exchange_calendars or
pandas_market_calendars. Maintains a static list of NYSE holidays through
``NYSE_CALENDAR_COVERS_THROUGH`` (see below), reconciled in CI against
``exchange_calendars`` (a dev-only dependency — never installed at runtime)
by ``tests/test_trading_calendar_reconciliation.py``.

Deliberate departure from the reference implementations (alpha-engine-config-
I9998): a full swap to ``exchange_calendars``/``pandas_market_calendars``
is the strictly-correct answer and was rejected here because it pulls
numpy/pandas transitively onto every consumer of this library, including
Lambdas that install krepis for unrelated primitives (SSM secrets, alert
transport, cost telemetry). The CI reconciliation check gives the
correctness property — divergence from the authoritative calendar fails
the build — without the runtime dependency. If the dependency becomes
acceptable at the consumer tier, prefer the swap and delete this table.

Usage:
    python trading_calendar.py              # check today
    python trading_calendar.py 2026-04-03   # check specific date

Exit codes:
    Always exits 0 — Step Function checks stdout markers, not exit code.

Stdout markers:
    "TRADING DAY"  = NYSE is open (proceed with pipeline)
    "MARKET_CLOSED" = weekend or holiday (skip pipeline)
"""

from __future__ import annotations

import sys
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

# NYSE observed holidays through NYSE_CALENDAR_COVERS_THROUGH (below).
# Source: https://www.nyse.com/markets/hours-calendars
# Updated annually — add new years as they're published. Kept honest by
# tests/test_trading_calendar_reconciliation.py, which fails CI on any
# divergence from exchange_calendars for the declared range (I9998) — this
# comment alone did not catch the table missing 2025-01-09, below.
NYSE_HOLIDAYS: set[date] = {
    # 2025
    date(2025, 1, 1),    # New Year's Day
    date(2025, 1, 9),    # National Day of Mourning — President Carter
    date(2025, 1, 20),   # MLK Day
    date(2025, 2, 17),   # Presidents' Day
    date(2025, 4, 18),   # Good Friday
    date(2025, 5, 26),   # Memorial Day
    date(2025, 6, 19),   # Juneteenth
    date(2025, 7, 4),    # Independence Day
    date(2025, 9, 1),    # Labor Day
    date(2025, 11, 27),  # Thanksgiving
    date(2025, 12, 25),  # Christmas
    # 2026
    date(2026, 1, 1),    # New Year's Day
    date(2026, 1, 19),   # MLK Day
    date(2026, 2, 16),   # Presidents' Day
    date(2026, 4, 3),    # Good Friday
    date(2026, 5, 25),   # Memorial Day
    date(2026, 6, 19),   # Juneteenth
    date(2026, 7, 3),    # Independence Day (observed, July 4 is Saturday)
    date(2026, 9, 7),    # Labor Day
    date(2026, 11, 26),  # Thanksgiving
    date(2026, 12, 25),  # Christmas
    # 2027
    date(2027, 1, 1),    # New Year's Day
    date(2027, 1, 18),   # MLK Day
    date(2027, 2, 15),   # Presidents' Day
    date(2027, 3, 26),   # Good Friday
    date(2027, 5, 31),   # Memorial Day
    date(2027, 6, 18),   # Juneteenth (observed, June 19 is Saturday)
    date(2027, 7, 5),    # Independence Day (observed, July 4 is Sunday)
    date(2027, 9, 6),    # Labor Day
    date(2027, 11, 25),  # Thanksgiving
    date(2027, 12, 24),  # Christmas (observed, Dec 25 is Saturday)
    # 2028
    date(2028, 1, 17),   # MLK Day
    date(2028, 2, 21),   # Presidents' Day
    date(2028, 4, 14),   # Good Friday
    date(2028, 5, 29),   # Memorial Day
    date(2028, 6, 19),   # Juneteenth
    date(2028, 7, 4),    # Independence Day
    date(2028, 9, 4),    # Labor Day
    date(2028, 11, 23),  # Thanksgiving
    date(2028, 12, 25),  # Christmas
    # 2029
    date(2029, 1, 1),    # New Year's Day
    date(2029, 1, 15),   # MLK Day
    date(2029, 2, 19),   # Presidents' Day
    date(2029, 3, 30),   # Good Friday
    date(2029, 5, 28),   # Memorial Day
    date(2029, 6, 19),   # Juneteenth
    date(2029, 7, 4),    # Independence Day
    date(2029, 9, 3),    # Labor Day
    date(2029, 11, 22),  # Thanksgiving
    date(2029, 12, 25),  # Christmas
    # 2030
    date(2030, 1, 1),    # New Year's Day
    date(2030, 1, 21),   # MLK Day
    date(2030, 2, 18),   # Presidents' Day
    date(2030, 4, 19),   # Good Friday
    date(2030, 5, 27),   # Memorial Day
    date(2030, 6, 19),   # Juneteenth
    date(2030, 7, 4),    # Independence Day
    date(2030, 9, 2),    # Labor Day
    date(2030, 11, 28),  # Thanksgiving
    date(2030, 12, 25),  # Christmas
    # 2031
    date(2031, 1, 1),    # New Year's Day
    date(2031, 1, 20),   # MLK Day
    date(2031, 2, 17),   # Presidents' Day
    date(2031, 4, 11),   # Good Friday
    date(2031, 5, 26),   # Memorial Day
    date(2031, 6, 19),   # Juneteenth
    date(2031, 7, 4),    # Independence Day
    date(2031, 9, 1),    # Labor Day
    date(2031, 11, 27),  # Thanksgiving
    date(2031, 12, 25),  # Christmas
    # 2032
    date(2032, 1, 1),    # New Year's Day
    date(2032, 1, 19),   # MLK Day
    date(2032, 2, 16),   # Presidents' Day
    date(2032, 3, 26),   # Good Friday
    date(2032, 5, 31),   # Memorial Day
    date(2032, 6, 18),   # Juneteenth (observed, June 19 is Saturday)
    date(2032, 7, 5),    # Independence Day (observed, July 4 is Sunday)
    date(2032, 9, 6),    # Labor Day
    date(2032, 11, 25),  # Thanksgiving
    date(2032, 12, 24),  # Christmas (observed, Dec 25 is Saturday)
}

# Last calendar date this module has verified holiday + early-close data
# for. `is_trading_day` (and everything built on it) RAISES past this date
# rather than silently treating every weekday as a trading day — the
# degrade-unsafe failure measured in alpha-engine-config-I9998 (the table
# ran out at 2030-12-25 and every weekday after, including Christmas,
# read as a trading day with no assertion, no expiry check, no log line).
# Bump this (with the holiday + early-close tables) when the calendar is
# next extended — never past what has actually been verified.
NYSE_CALENDAR_COVERS_THROUGH: date = date(2032, 12, 31)


class TradingCalendarExpiredError(ValueError):
    """Raised when a date falls past the verified calendar range.

    Distinct from plain ``ValueError`` so a caller that wants to catch
    this specific condition (e.g. to page for a calendar-table refresh)
    doesn't have to pattern-match an error string.
    """


def is_trading_day(d: date | None = None) -> bool:
    """Return True if the given date is an NYSE trading day.

    Raises :class:`TradingCalendarExpiredError` for any ``d`` past
    :data:`NYSE_CALENDAR_COVERS_THROUGH` — the calendar table has no
    holiday or early-close data beyond that date, and silently answering
    ``True`` for every such weekday (including Christmas) is the unsafe
    direction for a function every artifact key is built on.
    """
    if d is None:
        d = date.today()
    if d > NYSE_CALENDAR_COVERS_THROUGH:
        raise TradingCalendarExpiredError(
            f"is_trading_day({d.isoformat()}): NYSE_HOLIDAYS only covers "
            f"through {NYSE_CALENDAR_COVERS_THROUGH.isoformat()} — extend "
            f"NYSE_HOLIDAYS, NYSE_EARLY_CLOSES and "
            f"NYSE_CALENDAR_COVERS_THROUGH in krepis.trading_calendar "
            f"before resolving dates past that range."
        )
    if d.weekday() > 4:  # Saturday=5, Sunday=6
        return False
    if d in NYSE_HOLIDAYS:
        return False
    return True


def next_trading_day(d: date | None = None) -> date:
    """Return the next NYSE trading day after the given date."""
    if d is None:
        d = date.today()
    d = d + timedelta(days=1)
    while not is_trading_day(d):
        d = d + timedelta(days=1)
    return d


def previous_trading_day(d: date | None = None) -> date:
    """Return the most recent NYSE trading day strictly before the given date."""
    if d is None:
        d = date.today()
    d = d - timedelta(days=1)
    while not is_trading_day(d):
        d = d - timedelta(days=1)
    return d


def add_trading_days(start: date, n: int) -> date:
    """Add ``n`` NYSE trading days to ``start`` (n >= 0).

    Skips weekends + NYSE holidays. ``add_trading_days(d, 0) == d``
    (no rounding to a trading day if ``d`` itself is not one — only
    the forward steps land on trading days).

    Use ``subtract_trading_days`` for negative offsets.
    """
    if n < 0:
        raise ValueError(f"add_trading_days requires n >= 0, got {n}")
    current = start
    for _ in range(n):
        current = next_trading_day(current)
    return current


def subtract_trading_days(start: date, n: int) -> date:
    """Subtract ``n`` NYSE trading days from ``start`` (n >= 0)."""
    if n < 0:
        raise ValueError(f"subtract_trading_days requires n >= 0, got {n}")
    current = start
    for _ in range(n):
        current = previous_trading_day(current)
    return current


def count_trading_days(start: date, end: date) -> int:
    """Count NYSE trading days strictly between ``start`` and ``end``.

    Half-open interval ``(start, end]`` — same convention as
    ``add_trading_days``: ``count_trading_days(d, add_trading_days(d, n)) == n``
    for any ``n >= 0`` and ``d`` (whether or not ``d`` is a trading day).

    Returns 0 when ``end <= start``.
    """
    if end <= start:
        return 0
    total = 0
    current = start
    while current < end:
        current = current + timedelta(days=1)
        if is_trading_day(current):
            total += 1
    return total


# Sessions that close early (1 PM ET) rather than the regular 4 PM close —
# the day after Thanksgiving, Christmas Eve when it is itself a trading
# day (it is a full NYSE_HOLIDAYS closure, not an early close, in the
# years the observed Christmas holiday falls on it — 2027 and 2032
# below), and the day before Independence Day when July 4 falls
# midweek (Tue-Fri) so July 3 is itself a trading day. Reconciled
# against exchange_calendars in CI alongside NYSE_HOLIDAYS (I9998) —
# the July-3rd early closes below were found BY that reconciliation
# script, not composed by hand; do not hand-add a new one without
# re-running it.
NYSE_EARLY_CLOSES: set[date] = {
    date(2025, 7, 3),    # day before Independence Day (July 4 is Friday)
    date(2025, 11, 28),  # day after Thanksgiving
    date(2025, 12, 24),  # Christmas Eve
    date(2026, 11, 27),  # day after Thanksgiving
    date(2026, 12, 24),  # Christmas Eve
    date(2027, 11, 26),  # day after Thanksgiving
    # 2027-12-24 is the observed Christmas holiday (12/25 is a Saturday) —
    # a full closure, not an early close.
    date(2028, 7, 3),    # day before Independence Day (July 4 is Tuesday)
    date(2028, 11, 24),  # day after Thanksgiving
    # 2028-12-24 is a Sunday — not a trading day at all.
    date(2029, 7, 3),    # day before Independence Day (July 4 is Wednesday)
    date(2029, 11, 23),  # day after Thanksgiving
    date(2029, 12, 24),  # Christmas Eve
    date(2030, 7, 3),    # day before Independence Day (July 4 is Thursday)
    date(2030, 11, 29),  # day after Thanksgiving
    date(2030, 12, 24),  # Christmas Eve
    date(2031, 7, 3),    # day before Independence Day (July 4 is Friday)
    date(2031, 11, 28),  # day after Thanksgiving
    date(2031, 12, 24),  # Christmas Eve
    date(2032, 11, 26),  # day after Thanksgiving
    # 2032-12-24 is the observed Christmas holiday (12/25 is a Saturday) —
    # a full closure, not an early close. 2032 July 4 is a Sunday
    # (observed Monday July 5) so July 3 (Saturday) is not a trading day.
}

# NYSE regular-session close. Consumers that need the ACTUAL close on a
# given day (last_closed_trading_day) must read session_close_et(d)
# instead — this constant alone does not know about early closes.
_NYSE_CLOSE_ET = time(16, 0)
_EARLY_CLOSE_ET = time(13, 0)
_NYSE_TZ = ZoneInfo("America/New_York")

# NYSE regular-session open. Paired with ``_NYSE_CLOSE_ET`` above so the
# session window has exactly one definition in the fleet.
_NYSE_OPEN_ET = time(9, 30)


def session_close_et(d: date) -> time:
    """Return the actual NYSE close time for session ``d``.

    ``time(13, 0)`` on a day in :data:`NYSE_EARLY_CLOSES`, else the
    regular ``time(16, 0)`` close. This is the close :func:`last_closed_trading_day`
    must use — that function answers "has this session actually closed",
    and answering it with the regular close on an early-close day reports
    a session as still-open for three hours after it settled (I9998 §3).
    """
    return _EARLY_CLOSE_ET if d in NYSE_EARLY_CLOSES else _NYSE_CLOSE_ET


def is_market_hours(
    now: datetime | None = None,
    *,
    open_et: time | None = None,
    close_et: time | None = None,
) -> bool:
    """Return True iff ``now`` falls inside a live NYSE regular session.

    The session is the half-open interval ``[09:30, 16:00)`` ET on a
    trading day — open-inclusive, close-EXCLUSIVE. The exclusive close is
    load-bearing, not a detail: the post-close chain (daemon shutdown ->
    ``ne-postclose-trading-pipeline``) fires *at* 16:00:0x ET, and a
    close-inclusive boundary would classify the legitimate settlement run
    as in-session. It matches ``session_date``'s partition, whose
    ``(close(S-1), close(S)]`` convention puts the close instant itself in
    the closing session rather than the next one.

    Lifted here from ``crucible-executor/executor/market_hours.py``
    (alpha-engine-config-I7111), which carried a second hand-maintained
    copy of :data:`NYSE_HOLIDAYS`. This module already owns the NYSE
    calendar and the close constant, so the session predicate belongs
    beside them: one holiday table, one session window, one definition of
    "the market is open". Reachable as
    ``nousergon_lib.trading_calendar.is_market_hours`` — that module is an
    alias for this one.

    Deliberately answers only "is the regular session live" and
    deliberately does NOT consult :func:`session_close_et` /
    :data:`NYSE_EARLY_CLOSES`: early-close sessions (the day after
    Thanksgiving, Christmas Eve) close at 13:00 ET and are reported as
    open until 16:00 here; that is the conservative direction for every
    caller this serves — a caller refusing to act in-session refuses for
    three extra hours rather than acting inside a session it thought had
    ended. :func:`last_closed_trading_day` is the opposite case (the
    knowledge axis, not the action axis) and DOES use the actual close —
    see its docstring.

    Args:
        now: naive (assumed NYSE-local) or tz-aware datetime; defaults to
            the current moment in NYSE time.
        open_et / close_et: session-window overrides, for callers that
            pin a different close (``crucible-executor`` exposes them as
            ``MARKET_CLOSE_HOUR`` / ``MARKET_CLOSE_MINUTE``). Read per
            call, never captured at import — a window baked in at process
            start is a deploy artifact rather than a configuration.
    """
    if now is None:
        now = datetime.now(_NYSE_TZ)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=_NYSE_TZ)
    else:
        now = now.astimezone(_NYSE_TZ)

    if not is_trading_day(now.date()):
        return False

    return (open_et or _NYSE_OPEN_ET) <= now.time() < (close_et or _NYSE_CLOSE_ET)


def session_date(
    now: datetime | None = None, *, strict: bool = False
) -> date:
    """Return the NYSE session a moment belongs to (event-time axis).

    Partitions time by session closes: session S owns the interval
    ``(close(S-1), close(S)]`` in ET. This is the *trade-date* axis — the
    session a fill, NAV mark, or account snapshot physically belongs to —
    and is deliberately distinct from ``last_closed_trading_day`` (the
    *knowledge/as-of* axis: the newest fully-closed session whose data a
    computation may use). During a live session the two differ by exactly
    one session; conflating them is the off-by-one that mis-joined the EOD
    reconcile (config#1610, 2026-07-02).

      - Monday 9 AM ET     → Mon (pre-open: the upcoming session)
      - Monday 1 PM ET     → Mon (intraday)
      - Monday 4:00 PM ET  → Mon (at the close, inclusive)
      - Monday 4:05 PM ET  → Tue (post-close events print next session)
      - Saturday           → Mon (next session)
      - Fri 2026-07-03 (holiday) → Mon 2026-07-06

    Session-scoped processes (the intraday daemon) should resolve this
    ONCE at startup and freeze it — the frozen value stays correct through
    the close, and freezing prevents a post-close shutdown path from
    drifting onto the next session.

    Args:
        now: naive (assumed NYSE-local) or tz-aware datetime; defaults to
            the current moment in NYSE time.
        strict: when True, raise ``ValueError`` unless ``now`` falls on the
            returned session's own calendar day at-or-before the close —
            i.e. fail loud instead of silently attributing a weekend,
            holiday, or post-close start to the next session. Session-
            scoped processes that must never start outside their own
            session (the daemon) pass ``strict=True``.
    """
    if now is None:
        now = datetime.now(_NYSE_TZ)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=_NYSE_TZ)
    else:
        now = now.astimezone(_NYSE_TZ)

    today = now.date()
    if is_trading_day(today) and now.time() <= _NYSE_CLOSE_ET:
        return today
    nxt = next_trading_day(today)
    if strict:
        raise ValueError(
            f"session_date(strict=True): {now.isoformat()} does not fall "
            f"within a live NYSE session (next session: {nxt.isoformat()}). "
            f"Refusing to attribute a weekend/holiday/post-close start to a "
            f"future session."
        )
    return nxt


def assert_within_session(ts: datetime, session: date | str) -> None:
    """Fail-loud content-vs-key check: ``ts`` must belong to ``session``.

    Writers of session-keyed event artifacts (trade log, nav_series,
    snapshots) call this before persisting, so an event timestamped in a
    different session than its label raises at write time instead of
    silently mis-keying — the write-time guard that would have caught the
    daemon's D-1 mislabeling (config#1610) on day one.
    """
    if isinstance(session, str):
        session = date.fromisoformat(session)
    actual = session_date(ts)
    if actual != session:
        raise ValueError(
            f"Event timestamp {ts.isoformat()} belongs to session "
            f"{actual.isoformat()}, not the labeled session "
            f"{session.isoformat()} — refusing to write a mis-keyed "
            f"session artifact."
        )


def last_closed_trading_day(now: datetime | None = None) -> date:
    """Return the most recent NYSE trading day whose session has actually closed.

    Unified "last closed trading day" semantic for data consumers in
    both pre-open and post-close contexts:

      - Monday 9 AM ET    → Fri (Monday's session has not closed yet)
      - Monday 4:30 PM ET → Mon (Monday's session has closed)
      - Sunday 10 AM ET   → Fri (nothing has closed since Fri)
      - Tue after MLK Day → Fri before MLK Day (MLK is not a trading day)

    Morning consumers naturally land on the prior trading day (market
    hasn't closed yet); EOD consumers naturally land on the same day
    (market has closed). Both consumers ask the same question and get
    the correct answer without knowing which context they're in.

    Accepts either a naive datetime (assumed in NYSE local time) or a
    timezone-aware datetime (converted to NYSE time for comparison).
    Defaults to now in NYSE time.

    Reads the actual close via :func:`session_close_et` (unlike
    :func:`is_market_hours`, which deliberately pins the regular 16:00
    close for every day) — this is the knowledge axis: a postclose
    consumer running at 14:00 ET on an early-close day must see today's
    session as already closed, not two hours from closing
    (alpha-engine-config-I9998 §3; measured: previously returned the
    PRIOR session at 14:00 ET on an early-close day).
    """
    if now is None:
        now = datetime.now(_NYSE_TZ)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=_NYSE_TZ)
    else:
        now = now.astimezone(_NYSE_TZ)

    today = now.date()
    if is_trading_day(today) and now.time() >= session_close_et(today):
        return today
    d = today - timedelta(days=1)
    while not is_trading_day(d):
        d = d - timedelta(days=1)
    return d


if __name__ == "__main__":
    check_date = date.today()
    if len(sys.argv) > 1:
        check_date = date.fromisoformat(sys.argv[1])

    trading = is_trading_day(check_date)
    day_name = check_date.strftime("%A")

    if trading:
        print(f"{check_date} ({day_name}): TRADING DAY")
        sys.exit(0)
    else:
        reason = "weekend" if check_date.weekday() > 4 else "NYSE holiday"
        nxt = next_trading_day(check_date)
        print(f"{check_date} ({day_name}): MARKET_CLOSED ({reason}) — next trading day: {nxt}")
        # Exit 0 so SSM reports Success — Step Function checks stdout marker
        # instead of exit code to distinguish holidays from script crashes.
        sys.exit(0)
