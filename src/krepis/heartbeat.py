"""
Long-running phase heartbeat mechanism — §116 rule 5 chokepoint.

**Convention (declared):** every long-running phase SHOULD emit progress
heartbeats at a declared cadence shorter than its own timeout budget, so
operators can distinguish "slow" from "wedged" without SSHing in.

A heartbeat is a structured log line at INFO level containing the slug,
UTC timestamp, and interval.  Phases that hardware a magic-number
iteration count instead of a time-based interval (e.g. the
crucible-backtester's ``_HEARTBEAT_EVERY = 250``) MAY keep their
iteration-based approach as long as the time dimension is derivable
(iteration count × typical per-iteration duration < timeout) — but new
phases SHOULD prefer a wall-clock interval.

**Chokepoint test:** a test asserting a declared long-running phase
has a heartbeat cadence shorter than its own timeout budget (see
``tests/test_heartbeat.py``).

**Mechanism — Python callers:**

.. code-block:: python

    from krepis.heartbeat import HEARTBEAT_INTERVAL_S, emit_heartbeat

    for i, batch in enumerate(batches):
        process(batch)
        if i % 250 == 0:          # iteration-based, existing pattern
            emit_heartbeat("backtest")

    # Or time-based for new phases:
    import time
    last = time.monotonic()
    for batch in batches:
        process(batch)
        if time.monotonic() - last >= HEARTBEAT_INTERVAL_S:
            emit_heartbeat("enrich")
            last = time.monotonic()

**Mechanism — bash callers** (background subprocess, killed when the
phase completes):

.. code-block:: bash

    python -m krepis.heartbeat emit --slug spot-train --interval 300 &
    HEARTBEAT_PID=$!
    trap 'kill "$HEARTBEAT_PID" 2>/dev/null || true' EXIT

    # ... long-running phase ...

    kill "$HEARTBEAT_PID"

**Precedents (pre-convention, independently invented):**

- ``crucible-backtester/backtest.py`` — ``_HEARTBEAT_EVERY = 250`` (a per-date
  INFO heartbeat, added after a >100-minute silent phase in the 2026-04-22
  4th dry-run).  A hardcoded iteration constant, not derived or declared
  anywhere lookup-able.  Import ``HEARTBEAT_INTERVAL_S`` from this module
  instead so the value is derivable.
- ``crucible-predictor/infrastructure/spot_train.sh`` — emits a CloudWatch
  ``Heartbeat`` metric only at phase COMPLETION (model-zoo rotation, spec
  train, select, full training).  This is an end-of-phase signal, not a
  progress heartbeat DURING the phase.  Use the background-subprocess
  pattern above for in-phase heartbeats.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Recommended maximum interval between heartbeats for a long-running phase.
# 300 seconds (5 minutes) is the fleet default for phases without a tighter
# requirement.  A phase with a total timeout of N seconds should heartbeat
# at least every ``min(N // 3, HEARTBEAT_INTERVAL_S)`` seconds.
HEARTBEAT_INTERVAL_S: int = 300


def emit_heartbeat(slug: str, *, interval_s: int = HEARTBEAT_INTERVAL_S) -> None:
    """Log a heartbeat message at INFO level with a structured format.

    Call this inside the main loop of a long-running phase at (or more
    frequently than) the declared cadence.  The heartbeat includes the
    slug, a UTC timestamp, and the configured interval so operators can
    distinguish "slow" (heartbeats still arriving on schedule) from
    "wedged" (heartbeats stopped).

    Args:
        slug: label identifying the phase (e.g. ``"spot-train"``,
            ``"backtest"``, ``"enrich"``).  Displayed in the heartbeat
            message alongside the timestamp.
        interval_s: the cadence at which this caller intends to heartbeat
            (default: :const:`HEARTBEAT_INTERVAL_S`).  Included in the
            message so a reader can evaluate whether the cadence is
            appropriate for the phase's timeout budget.

    Example output::

        HEARTBEAT [spot-train] alive at 2026-07-23T12:00:00+00:00 (interval=300s)
    """
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    msg = f"HEARTBEAT [{slug}] alive at {now} (interval={interval_s}s)"
    logger.info(msg)
    # Also print to stderr so the line reaches CloudWatch / SSM StandardOutput
    # even when the caller does not configure a Python logging handler.
    print(msg, file=sys.stderr)


def emit_heartbeat_if_elapsed(
    slug: str,
    *,
    interval_s: int = HEARTBEAT_INTERVAL_S,
    last_heartbeat_at: float | None = None,
    now: float | None = None,
) -> bool:
    """Call :func:`emit_heartbeat` if ``interval_s`` has elapsed since the last one.

    Convenience wrapper for time-based callers that want a single
    per-loop-iteration guard::

        last = 0.0
        for batch in batches:
            process(batch)
            last = krepis.heartbeat.emit_heartbeat_if_elapsed(
                "enrich", last_heartbeat_at=last
            )

    Args:
        slug: passed through to :func:`emit_heartbeat`.
        interval_s: minimum seconds between heartbeats.
        last_heartbeat_at: ``time.monotonic()`` result from the last
            heartbeat, or ``0.0`` / ``None`` for the first call (always
            emits on the first call).
        now: override ``time.monotonic()`` for deterministic tests.

    Returns:
        The updated ``last_heartbeat_at`` value — always ``now`` when a
        heartbeat was emitted, unchanged otherwise.  Assign the return
        value back to the caller's tracking variable.
    """
    now = now if now is not None else time.monotonic()
    if last_heartbeat_at is None or (now - last_heartbeat_at) >= interval_s:
        emit_heartbeat(slug, interval_s=interval_s)
        return now
    return last_heartbeat_at or 0.0


def _heartbeat_loop(slug: str, *, interval_s: int = HEARTBEAT_INTERVAL_S) -> None:
    """Emit heartbeats on an infinite loop.  Used via CLI for bash callers.

    Killed by SIGTERM when the wrapping phase completes.  The final output
    before termination is a heartbeat at ``interval_s`` granularity — there
    is no "cleanup" heartbeat on SIGTERM (the phase's own status marker is
    the authoritative end-of-phase signal).
    """
    while True:
        emit_heartbeat(slug, interval_s=interval_s)
        time.sleep(interval_s)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m krepis.heartbeat",
        description=(
            "Emit progress heartbeats for a long-running phase at a declared "
            "cadence (§116 rule 5).  Run as a background subprocess from bash "
            "and kill it when the phase completes."
        ),
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    emit_p = subparsers.add_parser(
        "emit",
        help="Emit heartbeats on an infinite loop until killed.",
    )
    emit_p.add_argument(
        "--slug",
        required=True,
        help="Slug identifying the long-running phase (e.g. 'spot-train').",
    )
    emit_p.add_argument(
        "--interval",
        type=int,
        default=HEARTBEAT_INTERVAL_S,
        help=(
            f"Heartbeat cadence in seconds (default: {HEARTBEAT_INTERVAL_S}).  "
            "Shorter than one third of the phase's total timeout."
        ),
    )

    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING)

    if args.cmd == "emit":
        _heartbeat_loop(args.slug, interval_s=args.interval)

    return 0


if __name__ == "__main__":
    sys.exit(main())
