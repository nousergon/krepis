"""usage_pacing.py — linear-pace short-circuit for a rolling usage quota.

Answers one question: *given a quota that resets on a fixed cadence (e.g.
Anthropic's Claude Max weekly reset), is consumption running AHEAD of a
straight-line pace through the current window?* This is a materially
different (and earlier-warning) signal than a static ceiling: a fixed
85%/95%-of-quota threshold only fires once most of the week's budget is
already gone, so a burst that front-loads the week (e.g. 60% of quota by
Tuesday) goes uncaught until Thursday/Friday. Comparing ``used_frac`` against
``elapsed_frac`` (the fraction of the reset window that has passed) catches
that burst immediately, at any point in the window.

This module is pure math — no I/O, no AWS, no knowledge of what the quota
actually measures (tokens, WET, dollars, requests). Callers own: fetching the
current usage total, expressing it as a fraction of their own ceiling, and
picking their own reset anchor/period.

Motivating consumer: alpha-engine's backlog-groom pacing gate (a pre-boot
short-circuit + mid-run wind-down check against Brian's Claude Max weekly
quota, keyed to the observed reset instant). Originally duplicated between
that consumer and a dashboard view; lifted here as the shared primitive
(second-adoption consolidation).

Example::

    from datetime import datetime, timedelta
    from krepis.usage_pacing import pace_check

    anchor = datetime(2026, 6, 28, 20, 59)   # one observed reset instant
    period = timedelta(days=7)
    now = datetime(2026, 7, 2, 20, 59)       # 4 days into the window
    status = pace_check(used_frac=0.70, now=now, anchor=anchor, period=period)
    status.elapsed_frac   # ~0.571 (4/7 of the window elapsed)
    status.exceeded       # True — 70% used against ~57% of the window elapsed
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta


def reset_window(now: datetime, anchor: datetime, period: timedelta) -> tuple[datetime, datetime]:
    """Return ``(window_start, next_reset)`` for the reset cycle containing ``now``.

    ``anchor`` is one observed reset instant (need not be the first ever reset —
    any instant on the cadence works, since resets recur every ``period``).
    ``now`` and ``anchor`` must share the same awareness (both naive or both
    tz-aware); this function does no timezone conversion.
    """
    k = (now - anchor) // period
    start = anchor + k * period
    if start > now:
        start -= period
    return start, start + period


def elapsed_fraction(now: datetime, anchor: datetime, period: timedelta) -> float:
    """Fraction of the current reset window elapsed, clamped to ``[0.0, 1.0]``."""
    start, _ = reset_window(now, anchor, period)
    frac = (now - start) / period
    return max(0.0, min(1.0, frac))


@dataclass(frozen=True)
class PaceStatus:
    """Result of comparing usage against the straight-line pace through a window.

    ``exceeded`` is ``True`` when ``used_frac > elapsed_frac`` — consumption is
    running ahead of a linear pace (e.g. would exhaust the quota before the
    next reset if the current rate holds). ``overrun`` is the signed gap
    (``used_frac - elapsed_frac``); positive means ahead of pace, negative
    means under pace (headroom).
    """

    elapsed_frac: float
    used_frac: float
    overrun: float
    exceeded: bool


def pace_check(used_frac: float, now: datetime, anchor: datetime, period: timedelta) -> PaceStatus:
    """Compare ``used_frac`` (0.0-1.0+ fraction of quota consumed) against the
    fraction of the current reset window elapsed. Pure / no I/O — the caller
    is responsible for computing ``used_frac`` from its own usage source and
    ceiling.
    """
    elapsed = elapsed_fraction(now, anchor, period)
    overrun = used_frac - elapsed
    return PaceStatus(elapsed_frac=elapsed, used_frac=used_frac, overrun=overrun,
                       exceeded=overrun > 0)
