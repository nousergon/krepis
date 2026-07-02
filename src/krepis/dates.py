"""
dates.py — canonical "current date" attribution + freshness checks for
trade-related artifacts.

Implements the dual-tracking convention from
``alpha-engine-docs/private/DATE_CONVENTIONS.md``: every trade-related
artifact records both a ``calendar_date`` (wall-clock UTC) and a
``trading_day`` (last completed NYSE session). The ``trading_day``
attribution is strictly backward-looking — never ahead of "now" — so
artifacts produced on weekends, holidays, or pre-open weekday mornings
attribute to the most recent session that has actually closed.

Use this at every artifact-write site::

    from krepis.dates import now_dual

    dd = now_dual()
    record = {
        "calendar_date": dd.calendar_date,
        "trading_day": dd.trading_day,
        ...
    }

``trading_day`` is the system's PRIMARY axis — "what closed data am I
working with?" — and is correct for everything computed FROM data
(predictions, signals, features, eval joins). A small set of artifacts are
instead *records of physical events* (fills, NAV marks, account
snapshots); those are keyed by the **session axis** — "when did this event
happen?" — via :func:`session_date`, because brokers, benchmarks, and the
EOD snapshot/reconcile join all speak trade dates. During a live session
the two axes differ by exactly one session; see DATE_CONVENTIONS.md and
config#1610 for the doctrine::

    from krepis.dates import session_date

    run_date = session_date(strict=True)  # daemon startup: the session
                                          # being traded, fail-loud on
                                          # weekend/holiday/post-close

Writers of session-keyed artifacts self-check content-vs-key::

    from krepis.dates import assert_within_session

    assert_within_session(fill_time, run_date)  # raises on mis-key

For backfilling historical rows that only have a wall-clock timestamp::

    from krepis.dates import session_for_timestamp

    trading_day = session_for_timestamp(row["created_at"])

For freshness checks across the system, use the trading-day-aware helpers
rather than calendar-day arithmetic::

    from krepis.dates import is_fresh_in_trading_days

    # Producer-side postflight: did macro.SPY land the most recent close?
    if not is_fresh_in_trading_days(spy_last_date, run_date):
        raise PostflightError(...)

    # Consumer-side preflight: was the data refreshed within ≤1 trading day?
    if not is_fresh_in_trading_days(ticker_last_date, today, max_stale=1):
        raise PreflightError(...)

The freshness helpers replace the calendar-day-arithmetic patterns that bit
the 2026-05-24 SF recovery — every post-Saturday redrive trips a calendar-
day gate even when the data carries the most recent NYSE close. Calendar
days only happen to work on Saturday because Fri→Sat is +1 in both calendar
and trading-day arithmetic. See [[feedback_lift_invariants_to_chokepoint
_after_second_recurrence]] + [[feedback_sota_institutional_default_no_shortcuts]].

This module is a thin wrapper over
``krepis.trading_calendar.{last_closed_trading_day,count_trading_days}``
— its purpose is to standardize the *output shape* (DualDate) and provide a
single canonical entry point so every consumer sees the same semantics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, time, timezone

from .trading_calendar import (
    assert_within_session,
    count_trading_days,
    is_trading_day,
    last_closed_trading_day,
    previous_trading_day,
    session_date,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DualDate:
    """Calendar + trading_day attribution for a moment in time.

    Both fields are ISO ``yyyy-mm-dd`` strings — easy to serialize across
    JSON / SQLite / parquet boundaries with no timezone ambiguity.

    Attributes:
        calendar_date: wall-clock UTC date when the artifact was produced.
            Audit trail. Same on holidays/weekends as any other day; reflects
            *when* the process ran, not which session the data is about.
        trading_day: last NYSE trading session whose 4:00 PM ET close has
            occurred at or before the given moment. Strictly backward-looking;
            equals ``last_closed_trading_day(now)``. Never ahead of "now".

    Example::

        >>> from datetime import datetime, timezone
        >>> from krepis.dates import now_dual
        >>> sat = datetime(2026, 4, 25, 0, 0, tzinfo=timezone.utc)
        >>> dd = now_dual(now=sat)
        >>> dd.calendar_date
        '2026-04-25'
        >>> dd.trading_day  # Sat isn't a session; walk back to Fri
        '2026-04-24'
    """

    calendar_date: str
    trading_day: str


def now_dual(*, now: datetime | None = None) -> DualDate:
    """Canonical current-date attribution for the alpha-engine system.

    Every artifact-write site should call this to populate the
    ``calendar_date`` and ``trading_day`` columns/fields, rather than
    reaching for ``date.today()`` or ``datetime.now().date()``. Calling
    here ensures consistent semantics across modules and prevents the
    drift that motivated the convention (see
    ``alpha-engine-docs/private/DATE_CONVENTIONS.md``).

    Args:
        now: timezone-aware datetime. Defaults to current UTC time.
            Naive datetimes are interpreted as UTC for ``calendar_date``
            and forwarded to ``last_closed_trading_day`` (which itself
            assumes NYSE-local for naive inputs — this is intentional;
            the helper hides the conversion).

    Returns:
        DualDate where ``calendar_date`` is the UTC date of ``now`` and
        ``trading_day`` is the last NYSE session that has fully closed at
        or before ``now``.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    cal_utc = now.astimezone(timezone.utc).date()
    td = last_closed_trading_day(now)

    return DualDate(
        calendar_date=cal_utc.isoformat(),
        trading_day=td.isoformat(),
    )


def resolve_trading_day(date_str: str | None = None) -> str:
    """Normalize a date string to the most recent NYSE trading day on or before it.

    Canonical artifact-keying helper. Per ``DATE_CONVENTIONS``: every trade
    artifact keys by the TRADING DAY, not the calendar date. The Saturday SF
    threads a CALENDAR run_date (``date(Execution.StartTime)`` — e.g. Sat
    2026-05-30) while Research + ``signals.json`` + the backtester key by the
    trading day (Fri 2026-05-29). Normalizing the calendar run_date here keeps
    every consumer reading the same ``{module}/{trading_day}`` keys the
    producers wrote — the research↔backtester pit-parity drift and the
    evaluator's 0/18-graded report card both traced to a consumer that trusted
    the calendar run_date verbatim.

    Single source of truth for the normalizer previously duplicated as
    ``pipeline_common.resolve_trading_day`` (backtester) and
    ``grading.handler._to_trading_day`` (evaluator) — both thin wrappers over
    ``trading_calendar``. Lifted to the shared chokepoint on the second
    adoption.

    Behavior:
        * Idempotent — a trading-day input returns unchanged, so re-normalizing
          an already-normalized value (e.g. ``now_dual().trading_day`` or a
          bash-normalized ``RUN_DATE``) is a no-op.
        * Default (``date_str is None``) = today UTC, then normalized.
        * Tolerant of an ISO datetime / longer string — only the leading
          ``yyyy-mm-dd`` is parsed.
        * Defensive: on any parse failure, returns the input unchanged with a
          WARNING rather than raising. A date-normalization miss must not abort
          the caller (the backtester run or the non-fatal grading path).

    Args:
        date_str: ISO ``yyyy-mm-dd`` (or longer ISO) string, or ``None`` for
            today UTC.

    Returns:
        ISO ``yyyy-mm-dd`` string of the most recent NYSE trading day on or
        before the input (the input itself if it is already a trading day),
        or the raw input unchanged on a parse failure.

    Example::

        >>> resolve_trading_day("2026-05-30")  # Saturday
        '2026-05-29'
        >>> resolve_trading_day("2026-05-29")  # Friday (already a trading day)
        '2026-05-29'
        >>> resolve_trading_day("2026-05-30T12:00:00Z")  # ISO datetime tolerated
        '2026-05-29'
    """
    raw = date_str if date_str is not None else date.today().isoformat()
    try:
        d = date.fromisoformat(raw[:10])
        td = d if is_trading_day(d) else previous_trading_day(d)
        return td.isoformat()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "resolve_trading_day(%r) failed (%s) — using input unchanged", raw, exc
        )
        return raw


# ── Freshness checks (trading-day-aware) ─────────────────────────────────────


def expected_last_close(run_date: date | str) -> date:
    """The most recent NYSE close that exists as of ``run_date``.

    For Saturday/Sunday/holiday-Monday → the prior Friday (or earlier if a
    holiday-adjacent week). For trading days → the same date (the day's close
    has settled, anchored at end-of-day for the purpose of staleness checks).

    This is the canonical reference point for "what's the freshest data we
    could expect to see at this run_date?" — used by every producer-side
    postflight + consumer-side preflight in the system.

    Args:
        run_date: ISO ``yyyy-mm-dd`` string OR a ``datetime.date``.

    Returns:
        The expected last-closed NYSE session date as a ``datetime.date``.

    Example::

        >>> from datetime import date
        >>> expected_last_close(date(2026, 5, 24))  # Sunday
        datetime.date(2026, 5, 22)
        >>> expected_last_close("2026-05-25")  # Memorial Day (Mon)
        datetime.date(2026, 5, 22)
        >>> expected_last_close(date(2026, 5, 26))  # Tuesday after Memorial Day
        datetime.date(2026, 5, 26)
    """
    if isinstance(run_date, str):
        run_date = datetime.strptime(run_date, "%Y-%m-%d").date()
    # Anchor at 23:00 UTC of run_date so the resolver sees that day's NYSE
    # close as settled if run_date is itself a trading day. (NYSE close is
    # 4 PM ET = 20-21 UTC depending on DST; 23 UTC is unambiguously after.)
    anchor = datetime.combine(run_date, time(23, 0), tzinfo=timezone.utc)
    return last_closed_trading_day(anchor)


def trading_days_stale(last_date: date, reference: date | str) -> int:
    """Number of NYSE trading sessions ``last_date`` is behind ``reference``.

    Returns 0 when ``last_date`` is at or ahead of the reference's expected
    last-closed trading day. Holiday-aware (NYSE calendar, not US Federal).

    Semantically the canonical staleness metric for any "is this artifact
    carrying the most recent close that exists?" check across the system.
    Replaces the calendar-day arithmetic (``(reference - last_date).days``)
    that breaks on every non-Saturday redrive.

    Args:
        last_date: the artifact's stored last_date (``datetime.date``).
        reference: ISO ``yyyy-mm-dd`` string OR ``datetime.date`` — the
            run_date or "today" against which freshness is being asked.

    Returns:
        Integer count of NYSE sessions in
        ``(last_date, expected_last_close(reference)]``. Zero means
        "carries the most recent available close." Positive means
        "missed N sessions."

    Example::

        >>> from datetime import date
        >>> trading_days_stale(date(2026, 5, 22), date(2026, 5, 24))  # Fri vs Sun
        0
        >>> trading_days_stale(date(2026, 5, 22), date(2026, 5, 25))  # Fri vs Memorial-Mon
        0
        >>> trading_days_stale(date(2026, 5, 22), date(2026, 5, 26))  # Fri vs Tue close
        1
        >>> trading_days_stale(date(2026, 5, 13), date(2026, 5, 22))  # Wed 5/13 vs Fri 5/22
        7
    """
    expected = expected_last_close(reference)
    if last_date >= expected:
        return 0
    return count_trading_days(last_date, expected)


def is_fresh_in_trading_days(
    last_date: date,
    reference: date | str,
    *,
    max_stale: int = 0,
) -> bool:
    """Canonical freshness predicate: is ``last_date`` ≤ ``max_stale`` trading days behind ``reference``?

    Default ``max_stale=0`` means "must carry the most recent NYSE close that
    exists as of reference" — the strictest gate, used by producer-side
    postflights where the producer just wrote and should be current.
    ``max_stale=1`` tolerates one missing session — used by consumer-side
    preflights that need to survive the T+1 latency of polygon's daily
    aggregate publish.

    Args:
        last_date: artifact's stored last_date.
        reference: run_date the freshness is being asked at.
        max_stale: max permitted trading-day lag. Keyword-only to force
            explicit semantics at every call site.

    Returns:
        ``True`` iff the artifact carries data within ``max_stale`` trading
        days of the most recent NYSE close that exists as of ``reference``.

    Example::

        >>> from datetime import date
        >>> is_fresh_in_trading_days(date(2026, 5, 22), date(2026, 5, 24))  # Fri vs Sun
        True
        >>> is_fresh_in_trading_days(date(2026, 5, 13), date(2026, 5, 22))  # 7-session lag
        False
        >>> is_fresh_in_trading_days(date(2026, 5, 13), date(2026, 5, 22), max_stale=10)
        True
    """
    return trading_days_stale(last_date, reference) <= max_stale


def session_for_timestamp(ts: datetime) -> str:
    """Trading day a timestamp belongs to under the dual-tracking convention.

    Backward-looking: returns the most recent NYSE trading session whose
    4:00 PM ET close has occurred at or before ``ts``. Used to backfill the
    ``trading_day`` column on historical rows that only have a wall-clock
    timestamp (``created_at``, ``fill_time``, etc.).

    Args:
        ts: timezone-aware datetime. Naive timestamps are assumed UTC.

    Returns:
        ISO ``yyyy-mm-dd`` string of the trading day.

    Example::

        >>> from datetime import datetime
        >>> from zoneinfo import ZoneInfo
        >>> # Mon 9 AM ET — session not yet closed
        >>> ts = datetime(2026, 4, 27, 9, 0, tzinfo=ZoneInfo("America/New_York"))
        >>> session_for_timestamp(ts)
        '2026-04-24'
    """
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return last_closed_trading_day(ts).isoformat()
