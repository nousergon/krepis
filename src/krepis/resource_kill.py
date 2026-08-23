"""
The fleet's single resource-kill classifier (alpha-engine-config-I7442).

``sf-pipeline-policy.md`` §3, clause ``SFP-3-resource-kill-halts-and-is-named``
requires that an OOM or a timeout be **named** — as ``OOM`` or ``TIMEOUT``,
with the stage, the limit and the observed value — in the failure cause the
operator reads, not only in the spot log. The governance register recorded that
clause as ``kind: none`` with the reason:

    the chokepoint is a shared resource-kill classifier in the spot launcher
    (exit 137 / ``Killed`` / watchdog expiry -> a typed failure cause), **which
    does not exist**. Until it does, this is prose and the failure mode is a
    resource kill wearing a domain failure's message.

This module is that chokepoint. Before it, the same knowledge existed as two
independent, drifting copies —
:mod:`krepis.ssm_log_capture` (``_RESOURCE_KILL_RE`` / ``_OOM_RETURNCODES`` /
``_classify_resource_kill``) and :mod:`krepis.ssm_dispatcher`
(``_OOM_RESPONSE_CODES`` / ``_classify_terminal_failure``) — which already
disagreed: the dispatcher never scanned text at all, and the log-capture copy
never saw SSM's ``TimedOut`` status. Both now delegate here
(``policy-shared-code``: lift at the second adoption; this was the second).

**Three independent signals, any one sufficient.** A resource kill is
adversarial to detect precisely because the process is gone and cannot report
anything about itself:

1. **A return code.** ``137`` (128+SIGKILL under the POSIX shell convention) or
   ``-9`` (SIGKILL as a Python negative return code) is authoritative — when it
   survives. Most launchers launder it: ``if ! cmd; then echo "ERROR: X
   failed"; exit 1; fi`` turns a 137 into a 1 and the only honest signal is
   gone.
2. **SSM's own terminal status.** ``TimedOut`` means the SSM agent killed the
   command at ``executionTimeout`` before any exit code existed.
3. **A kill LINE in the captured output.** bash's own foreground-job-killed
   message (``<script>: line 16: 26748 Killed   python -u backtest.py``), a
   kernel ``oom-killer`` banner, a ``MemoryError`` traceback. This is the signal
   that survives a laundered return code — and the one the 2026-08-15 weekly run
   destroyed, because the line lived past SSM's 24KB inline cap in the S3 output
   prefix that teardown deleted (see :mod:`krepis.spot_evidence`).

**Why the kill line is scanned from the TAIL.** The last CAUSE-shaped line of a
failing run is usually its exit path, not its cause; but a kill line is
different — it is emitted by the shell at the moment of death, so the LAST one
is the real one. Scanning tail-first also means a bounded tail read is
sufficient, which is what makes the S3 range-get in
:mod:`krepis.ssm_dispatcher` cheap enough to do on every terminal failure.

**Absence is rendered, never omitted.** :func:`format_cause` always spells
``limit=`` and ``observed=``. A missing field reads as a satisfied obligation;
``observed=unavailable`` reads as what it is. An OOM-killed process reports no
peak RSS by construction — that is the structural fact, and stating it is more
honest than leaving the reader to infer it from silence.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Final, Optional

#: The two classifications this module returns. Deliberately a closed set: the
#: policy names exactly these two words, and the operator-facing message is
#: matched on them by downstream surfaces (``ssm_log_capture`` re-detects the
#: rendered line, sf-watch greps the SF cause).
OOM: Final[str] = "OOM"
TIMEOUT: Final[str] = "TIMEOUT"

#: Return codes that mean "killed for memory". 137 = 128+SIGKILL (POSIX shell
#: exit-code convention); -9 = SIGKILL as a Python negative ``returncode``.
OOM_RETURNCODES: Final[frozenset] = frozenset({137, -9})

#: 124 = the ``timeout`` coreutil's own "I killed the child at the budget"
#: convention; 143 = 128+SIGTERM; -14/-15 = SIGALRM/SIGTERM as Python negative
#: return codes. A watchdog SIGTERM and a ``timeout --signal=TERM`` both land
#: here.
TIMEOUT_RETURNCODES: Final[frozenset] = frozenset({124, 143, -14, -15})

#: SSM's own terminal status meaning "I killed this at ``executionTimeout``".
#: The only case where no exit code was ever produced by the workload.
TIMEOUT_STATUSES: Final[frozenset] = frozenset({"TimedOut"})

#: Lines that name a MEMORY kill. Anchored loosely on purpose: these come from
#: bash's job-control message, the kernel's oom-killer, glibc, CPython and Go,
#: and no single format spans them. A false positive costs a slightly-off lead
#: line; a miss costs the operator the entire diagnosis.
OOM_LINE_RE: Final = re.compile(
    r"\bKilled\b"
    r"|\bOOM\b"
    r"|[Oo]ut of memory"
    r"|oom.?kill"
    r"|Cannot allocate memory"
    r"|\bMemoryError\b"
    r"|\bSIGKILL\b"
    r"|\bsignal 9\b"
)

#: Lines that name a TIME kill. ``alpha-engine-watchdog`` is the fleet's own
#: systemd hard-timeout unit (rendered by :mod:`krepis.spot_bootstrap`); when it
#: fires, the box powers off mid-workload and this is the only trace.
#:
#: A bare ``timed out`` is DELIBERATELY absent. Every retrying HTTP client in
#: the fleet logs it on a request it then succeeds at, so admitting it would
#: classify an ordinary transient as a stage-level TIMEOUT — and a false
#: resource kill is worse than none: sf-pipeline-policy §3 forbids retrying a
#: resource kill on an unchanged workload, so a false one suppresses a
#: legitimate relaunch. Only tokens emitted BY a killer are listed.
TIMEOUT_LINE_RE: Final = re.compile(
    r"\bexecutionTimeout\b"
    r"|\bExecutionTimedOut\b"
    r"|\bSIGALRM\b"
    r"|\balpha-engine-watchdog\b"
    r"|\bec2-spot-watchdog\b"
    r"|\bTimeoutExpired\b"
)

#: Union of both, for callers that only need "is this line about a resource
#: kill at all" (``ssm_log_capture``'s streaming line scan).
KILL_LINE_RE: Final = re.compile(
    "(?:{})|(?:{})".format(OOM_LINE_RE.pattern, TIMEOUT_LINE_RE.pattern)
)

#: Cap on a rendered kill line. The whole point of the cause message is to fit
#: inside a downstream window (SSM's 24KB ``StandardErrorContent``, a Step
#: Functions ``cause`` field, a Telegram message); one unbounded line would
#: evict the fields beside it.
KILL_LINE_CLIP: Final[int] = 400

#: Rendered where a value genuinely is not knowable, rather than dropping the
#: field. See the module docstring: a missing field reads as satisfied.
UNKNOWN: Final[str] = "unknown"

#: The specific, structural reason an OOM has no observed value. Not a
#: placeholder — it is the finding.
OOM_OBSERVED_UNAVAILABLE: Final[str] = (
    "unavailable (a SIGKILLed process reports no peak RSS)"
)


def _clip(line: Optional[str]) -> str:
    if not line:
        return ""
    line = line.strip()
    if len(line) <= KILL_LINE_CLIP:
        return line
    return line[:KILL_LINE_CLIP] + "... (+{} chars)".format(len(line) - KILL_LINE_CLIP)


def find_kill_line(text: Optional[str]) -> Optional[str]:
    """Return the LAST resource-kill-shaped line in ``text``, or ``None``.

    Tail-first, and that is load-bearing rather than an optimisation: a kill
    line is emitted by the shell or the kernel at the moment of death, so the
    last one is the real one — unlike a generic cause line, where the last
    match is usually the exit path (the defect ``ssm_log_capture``'s separate
    ``cause_line``/``last_output_line`` pair exists for).
    """
    if not text:
        return None
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped and KILL_LINE_RE.search(stripped):
            return stripped
    return None


def classify(
    *,
    returncode: Optional[int] = None,
    status: Optional[str] = None,
    kill_line: Optional[str] = None,
    text: Optional[str] = None,
) -> Optional[str]:
    """Return :data:`OOM`, :data:`TIMEOUT`, or ``None`` (not a resource kill).

    Any one signal is sufficient; they are checked in order of how much a
    caller can have corrupted them.

    ``returncode`` is authoritative when it survived to this layer.
    ``status`` is SSM's terminal status (only ``TimedOut`` is meaningful).
    ``kill_line`` is a single already-selected line; ``text`` is a blob this
    function will scan with :func:`find_kill_line` when ``kill_line`` is not
    supplied — pass the TAIL of the captured output, not the head.

    OOM wins a tie. A SIGKILL that follows an ``executionTimeout`` mention is
    still the memory kill: the budget message is context, the kill is the
    event.
    """
    if returncode in OOM_RETURNCODES:
        return OOM
    line = kill_line if kill_line is not None else find_kill_line(text)
    if line and OOM_LINE_RE.search(line):
        return OOM
    if returncode in TIMEOUT_RETURNCODES:
        return TIMEOUT
    if status and status in TIMEOUT_STATUSES:
        return TIMEOUT
    if line and TIMEOUT_LINE_RE.search(line):
        return TIMEOUT
    return None


def format_cause(
    *,
    classification: str,
    stage: str,
    returncode: Optional[int] = None,
    kill_line: Optional[str] = None,
    limit: Optional[str] = None,
    observed: Optional[str] = None,
) -> str:
    """Render the operator-facing cause line for a classified resource kill.

    Shape (``sf-pipeline-policy`` §3 obligation 3 — say OOM or TIMEOUT, and
    name the stage, the limit and the observed value)::

        RESOURCE KILL (OOM) stage=predictor-backtest rc=137 limit=<...>
        observed=<...> — bash: line 16: 26748 Killed  python -u backtest.py

    ``limit`` and ``observed`` are ALWAYS rendered. When a caller cannot supply
    one it becomes ``unknown`` (or, for an OOM's observed value, the explicit
    :data:`OOM_OBSERVED_UNAVAILABLE` — a SIGKILLed process cannot report its
    peak RSS, which is the structural fact, not a gap in this code).
    """
    if observed is None and classification == OOM:
        observed = OOM_OBSERVED_UNAVAILABLE
    parts = [
        "RESOURCE KILL ({})".format(classification),
        "stage={}".format(stage or UNKNOWN),
        "rc={}".format(UNKNOWN if returncode is None else returncode),
        "limit={}".format(limit or UNKNOWN),
        "observed={}".format(observed or UNKNOWN),
    ]
    head = " ".join(parts)
    clipped = _clip(kill_line)
    if clipped:
        return "{} — {}".format(head, clipped)
    return "{} — no kill line survived in the captured output".format(head)


def main(argv: Optional[list] = None) -> int:
    """CLI so a bash launcher gets the same verdict without re-deriving it.

    ``python -m krepis.resource_kill classify --stage X --rc 137``

    Exit ``0`` when a resource kill is classified (the cause line is printed to
    stdout), ``3`` when it is not. A launcher therefore reads::

        if cause=$(... classify --stage "$S" --rc "$rc" --log "$L"); then
            echo "$cause" >&2
        fi

    Never exits non-zero for an internal problem — a classifier that fails
    loudly on the failure path would replace the diagnosis it exists to
    produce.
    """
    parser = argparse.ArgumentParser(
        prog="python -m krepis.resource_kill",
        description=(
            "Classify a terminal failure as an OOM / TIMEOUT resource kill and "
            "render the operator-facing cause line required by "
            "sf-pipeline-policy §3 (alpha-engine-config-I7442)."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("classify", help="Classify and render a cause line.")
    p.add_argument("--stage", required=True, help="Stage name for the cause line.")
    p.add_argument("--rc", type=int, default=None, help="Observed return code.")
    p.add_argument("--status", default=None, help="SSM terminal status, if any.")
    p.add_argument(
        "--log",
        default=None,
        help="Path to a captured log to scan for a kill line ('-' for stdin).",
    )
    p.add_argument(
        "--tail-bytes",
        type=int,
        default=64 * 1024,
        help="Bytes of --log to scan, from the END (default: 65536).",
    )
    p.add_argument("--limit", default=None, help="The resource limit that was hit.")
    p.add_argument("--observed", default=None, help="The observed value, if known.")
    p.add_argument("--json", action="store_true", help="Emit a JSON object.")

    args = parser.parse_args(argv)

    text = None
    if args.log == "-":
        text = sys.stdin.read()
    elif args.log:
        try:
            with open(args.log, "rb") as fh:
                try:
                    fh.seek(-abs(args.tail_bytes), 2)
                except OSError:
                    fh.seek(0)
                text = fh.read().decode("utf-8", errors="replace")
        except OSError as exc:
            # (a) swallowed: the log is unreadable. (b) the return-code and
            # status signals still classify without it. (c) recorded on stderr,
            # which the caller's ssm_log_capture ships to _ssm_logs/.
            print(
                "resource_kill: could not read {!r} ({}); "
                "classifying from rc/status only".format(args.log, exc),
                file=sys.stderr,
            )

    kill_line = find_kill_line(text)
    classification = classify(
        returncode=args.rc, status=args.status, kill_line=kill_line
    )
    if args.json:
        print(
            json.dumps(
                {
                    "classification": classification,
                    "stage": args.stage,
                    "returncode": args.rc,
                    "status": args.status,
                    "kill_line": kill_line,
                    "cause": (
                        format_cause(
                            classification=classification,
                            stage=args.stage,
                            returncode=args.rc,
                            kill_line=kill_line,
                            limit=args.limit,
                            observed=args.observed,
                        )
                        if classification
                        else None
                    ),
                }
            )
        )
        return 0 if classification else 3
    if not classification:
        return 3
    print(
        format_cause(
            classification=classification,
            stage=args.stage,
            returncode=args.rc,
            kill_line=kill_line,
            limit=args.limit,
            observed=args.observed,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
