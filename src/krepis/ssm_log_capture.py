"""
SSM-step log capture + S3 ship-on-exit chokepoint.

Consolidation substrate for the trap-and-log-ship pattern that previously
appeared as an inline bash EXIT trap in every long Step Functions SSM
state across the alpha-engine fleet (MorningEnrich, DataPhase1,
RAGIngestion, DriftDetection in alpha-engine-data; PredictorTraining in
alpha-engine-predictor; Backtester, Parity, Evaluator in
alpha-engine-backtester). The pre-lift form looked like::

    trap 'aws s3 cp /var/log/X.log "s3://alpha-engine-research/_ssm_logs/X/$(date -u +%Y-%m-%d)/$(hostname)-$(date -u +%H%M%SZ).log" --only-show-errors || true' EXIT
    bash infrastructure/<launcher>.sh ... 2>&1 | tee /var/log/X.log

The pattern was originally added by alpha-engine-data PR #244
(2026-05-15) to close the diagnostic gap where SSM's 24KB
``StandardOutputContent`` cap was hiding the root cause of long-step
failures: by the time SF Catch surfaced exit-1, the spot instance had
self-terminated and the full ``/var/log/X.log`` was gone with it. The
EXIT trap fires before the script's real exit propagates, ships the log
to S3, then yields back so the real exit code reaches the SF.

**Why the lift to lib (2026-05-22):** PR #253 in alpha-engine-data
(merged 2026-05-17) switched all 8 Saturday-SF spot states from plain
``commands`` JSON arrays to ``commands.$ States.Array(...)`` so they
could splice ``$.run_date`` / ``$.preflight_args`` via ``States.Format``.
Inside ``States.Array`` arg strings, ASL's documented escape for an
inner single quote is ``\\'`` — but in practice the AWS ASL evaluator
does NOT unescape ``\\'`` to ``'``, it passes the backslash through
literally. The trap line ``'trap \\'cmd\\' EXIT'`` rendered into the
SSM ``_script.sh`` as ``trap \\'cmd\\' EXIT``; bash interpreted the
``\\'`` outside quotes as a literal apostrophe stripped of its quoting
power, then word-split the line and passed every token after ``aws`` to
``trap`` as a signal name. Symptom: ``trap: s3: invalid signal
specification``, exit 127 at line 7 of ``_script.sh``. The 2026-05-22
Friday-PM shell-run dry-pass of the Saturday SF caught this exactly as
designed (it was the first execution under the broken pattern; no
Saturday SF had run between #253 merge and the dry-pass).

Per the ``~/Development/CLAUDE.md`` SOTA / institutional-approach rule —
sub-sub-rule "when mirroring a pattern across repos, consider lifting
it into ``nousergon-lib``... Pure-Bash primitives can stay mirrored
unless re-expressible as a Python CLI entry callable from Bash, in
which case the CLI re-expression is the institutional path" — this
module is the canonical Python primitive. The SF JSON now spells a
single ``States.Format``-rendered string (no bash trap, no bash
quoting, no ASL escape surface) and the consumer behavior lives here
where it can be tested independently of every state's JSON shape.

**Public API:**

- :func:`run` — execute an inner command, tee its merged stdout+stderr
  to a local log file AND to the parent process's stdout, on exit
  (any code, including subprocess crash) ship the log to S3, return
  the inner exit code verbatim.
- :func:`format_subprocess_failure` — format a terminal-failure message
  naming the failing step + last output line, so callers never surface
  a bare return code (the :mod:`krepis.ssm_log_capture` level §116
  rule 4 "no naked rc" chokepoint).
- CLI: ``python -m krepis.ssm_log_capture run --slug <X>
  --log /var/log/<X>.log [--step-name <N>] [--correlation-id <ID>]
  -- <inner-cmd...>``. Designed for SF JSON; a single
  ``States.Format`` template with ``$.preflight_args`` interpolated
  via ``{}`` produces the entire invocation as one un-quoted token
  list — no bash trap, no inner single quotes.

**S3 layout:**

``s3://{bucket}/_ssm_logs/{slug}/{YYYY-MM-DD}/{hostname}-{HHMMSSZ}[-{correlation_id}].log``

Defaults: ``bucket=alpha-engine-research``, prefix ``_ssm_logs``. Date,
time, and hostname are computed at exit time (so a multi-hour run that
straddles UTC midnight gets the actual exit-side date in the key). When
a ``--correlation-id`` is provided (or ``$RUN_TOKEN`` is set), the value
is appended before ``.log`` for traceability.

**Failure behavior — never raises:**

- Inner command's exit code is propagated verbatim. Subprocess setup
  failure (e.g., ``FileNotFoundError`` on the binary) is logged to the
  log file and stderr, returns 127 to match the bash convention.
- When the inner command fails (non-zero exit), a formatted failure
  message naming the failing step + last output line is printed to
  stderr before returning — never a bare exit code (§116 rule 4
  "no naked rc").
- S3 upload failures (boto3 ``ClientError``, missing creds, missing
  log file) are logged at WARNING and swallowed. The SF Catch must
  see the true inner exit, not a secondary log-capture failure that
  would mask it. Matches :mod:`krepis.alerts`' fail-safe
  posture.

**Correlation-id chokepoint (§116 rule 6):**

The ``--correlation-id`` CLI argument (or ``$RUN_TOKEN`` env var) is the
fleet-wide chokepoint guaranteeing every log surface carries a run/execution
correlation id. When neither is present the CLI AUTO-GENERATES one — best
effort from ``$SSM_COMMAND_ID`` (set by the SSM agent on the box for
long-running commands), falling back to a uuid4 — and logs a WARNING that
traceability is degraded (config#4223). The flag/env remain the signal of
intent ("I've provided a correlation id"); their absence degrades gracefully
rather than hard-failing, because a hard failure mid-pipeline (the
2026-07-25 weekly-SF kill, config#4223) took every downstream log surface
down with it. The correlation id appears as a ``# correlation-id: <id>``
header line at the top of the captured log file and is appended to the S3
key for traceability.

It is also **exported to the inner command as ``$RUN_TOKEN``**. Resolution
reads env -> CLI; the export is what makes the reverse direction exist, so a
launcher that dispatches nested work (its own ``SendCommand`` to a second
box) carries the caller's id down instead of minting one. Without it a
correlation id supplied as a CLI flag names only the parent's own log, and
every nested workload becomes unfindable by the execution that owns it —
measured 2026-07-31, when a whole PredictorTraining branch shipped under a
fabricated id and its failure was invisible to sf-watch
(alpha-engine-config-I5949 / I5956).
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import socket
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Optional

logger = logging.getLogger(__name__)

DEFAULT_BUCKET: Final[str] = "alpha-engine-research"
S3_PREFIX: Final[str] = "_ssm_logs"

# Env var consulted for correlation-id when ``--correlation-id`` is not
# explicitly passed. Set by the groom spot dispatcher (``GROOM_RUN_TOKEN``)
# and the data-spot-dispatcher (``run_token``). The module-level single
# variable lets the SF-dispatched callers inherit the dispatcher's token
# without a CLI change.
CORRELATION_ID_ENV_VAR: Final[str] = "RUN_TOKEN"

# Env var the SSM agent itself sets on the box for long-running commands
# (``aws ssm send-command``/Step Functions SSM states). Best-effort source
# for an auto-generated correlation id (config#4223): when the CLI flag and
# $RUN_TOKEN are both absent, a command dispatched through SSM still gets a
# stable, execution-scoped id from this rather than a throwaway uuid4.
SSM_COMMAND_ID_ENV_VAR: Final[str] = "SSM_COMMAND_ID"


def _child_env(
    env: dict[str, str] | None, correlation_id: str | None
) -> dict[str, str]:
    """Build the inner command's environment, carrying the correlation id down.

    The resolved correlation id identifies THIS capture, so the child must
    observe the same value — a launcher that dispatches nested work reads it
    from ``$RUN_TOKEN`` and has no other way to learn it. Resolution reads
    env -> CLI; without this the reverse direction does not exist, so a
    correlation id supplied as a CLI flag names the parent's own log and
    reaches nothing the parent then launches.

    Measured consequence (2026-07-31, ne-weekly-freshness-pipeline): the SF
    passed ``--correlation-id watch-rerun-2026-07-31-1``; the child
    ``spot_train.sh`` saw ``RUN_TOKEN`` unset, took its developer fallback,
    and shipped the whole PredictorTraining branch under a fabricated
    ``spot-reference-20260731``. The training failure that broke that run was
    therefore unfindable by execution (alpha-engine-config-I5949 / I5956).

    An explicit ``env`` override still supplies the base environment; the
    resolved correlation id is layered on top so the parent log and every
    nested log agree on one id.
    """
    child = dict(env) if env is not None else os.environ.copy()
    if correlation_id:
        child[CORRELATION_ID_ENV_VAR] = correlation_id
    return child


def _exit_key(
    slug: str,
    *,
    now: datetime | None = None,
    host: str | None = None,
    correlation_id: str | None = None,
) -> str:
    """Compute the S3 key for the log upload at exit time.

    Public for tests; the canonical layout is
    ``_ssm_logs/{slug}/{YYYY-MM-DD}/{hostname}-{HHMMSSZ}.log``. When
    ``correlation_id`` is provided it is appended before the ``.log``
    suffix for traceability.
    """
    now = now or datetime.now(timezone.utc)
    host = host or socket.gethostname()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H%M%SZ")
    suffix = f"-{correlation_id}" if correlation_id else ""
    return f"{S3_PREFIX}/{slug}/{date_str}/{host}-{time_str}{suffix}.log"


def _ship_log_to_s3(
    slug: str, log_path: Path, bucket: str, *, correlation_id: str | None = None
) -> tuple[bool, str]:
    """Upload ``log_path`` to S3.

    Returns ``(ok, detail)``. Never raises. Computes the key at call
    time so the timestamp reflects when the trap fires, not when the
    wrapper started.
    """
    key = _exit_key(slug, correlation_id=correlation_id)
    if not log_path.exists():
        return False, f"log file not found: {log_path}"
    try:
        import boto3

        s3 = boto3.client("s3")
        s3.upload_file(str(log_path), bucket, key)
        return True, f"s3://{bucket}/{key}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


#: Lines worth leading a failure message with. Deliberately broad: a false
#: positive costs a slightly-off first line, where a miss costs the reader a
#: manual S3 fetch of the whole log. Anchored loosely because these come from
#: Python logging, bare tracebacks, shell `set -e` output and Go binaries, and
#: no single format spans them.
_CAUSE_RE = re.compile(
    r"(Traceback \(most recent call last\)|^\s*raise\b|\bERROR\b|\bCRITICAL\b"
    r"|\bphase failed\b|\bstatus=error\b|\b[A-Za-z_]*Error\b|\b[A-Za-z_]*Exception\b"
    r"|^\s*assert\b|\bFAILED\b|\bfatal:)"
)

#: Cap on either line rendered into a failure message. The whole point is to
#: fit inside a downstream window (SSM's 24KB StandardErrorContent, a Telegram
#: message); a single unbounded line would evict the pair it is part of.
_FAILURE_LINE_CLIP = 400


def _clip(line: str | None) -> str:
    if not line:
        return "no output captured"
    line = line.strip()
    if len(line) <= _FAILURE_LINE_CLIP:
        return line
    return line[:_FAILURE_LINE_CLIP] + f"… (+{len(line) - _FAILURE_LINE_CLIP} chars)"


def format_subprocess_failure(
    step_name: str,
    *,
    returncode: int,
    last_output_line: str | None = None,
    cause_line: str | None = None,
) -> str:
    """Format a terminal-failure message naming the failing step + its cause.

    The §116 rule 4 ("no naked rc") chokepoint: callers that use this
    function on their exit path never surface a bare ``exit status 1``
    without context.

    The LAST line of a failing run is usually the exit path, not the cause.
    On 2026-08-11 the weekly SF's DataPhase1 died on

        ERROR [data-collector] historical_constituents phase failed: No
        'Selected changes to the list' table found ...

    at 22:05, then ran on for fifteen more minutes before exiting; the last
    line was ``failed to run commands: exit status 1``. What reached the
    Telegram alert and the Step Functions ``cause`` field was a git-fetch
    banner, and diagnosing it took a manual ``aws s3 cp`` of the full log
    (alpha-engine-config-I6945). ``cause_line`` — the last CAUSE-SHAPED line
    seen, per :data:`_CAUSE_RE` — is preferred when the two differ, and the
    last line is kept alongside it rather than dropped.

    Args:
        step_name: human-readable label for the failing step
            (e.g. ``"morning-enrich"``, ``"spot-train"``).
        returncode: the subprocess return code (non-zero indicates failure).
        last_output_line: the last non-empty line of merged stdout/stderr
            captured from the failing subprocess, or ``None`` when no
            output was captured (setup failure, empty output).
        cause_line: the last line matching :data:`_CAUSE_RE`, or ``None``.
            Leads the message when it differs from ``last_output_line``;
            both are rendered so nothing that was previously surfaced is
            lost.

    Returns:
        A single-line string suitable for ``stderr``. New callers should
        call this in preference to re-deriving ``sf_watch_run.sh``'s
        bespoke pattern.

    Example output::

        ssm_log_capture: ERROR: [spot-train] failed (rc=1) — AssertionError: validation failed
    """
    head = f"ssm_log_capture: ERROR: [{step_name}] failed (rc={returncode})"
    if cause_line and cause_line != last_output_line:
        # Both, in this order: the cause is what the reader needs, the last
        # line is what they would otherwise have had and occasionally adds
        # the exit path. Truncated so one runaway line cannot push the pair
        # back out of whatever window renders this.
        return f"{head} — {_clip(cause_line)} (last output: {_clip(last_output_line)})"
    if cause_line or last_output_line:
        return f"{head} — {_clip(cause_line or last_output_line)}"
    return f"{head} — no output captured"


def _resolve_correlation_id(cli_arg: str | None) -> str | None:
    """Resolve correlation id from CLI arg or env var.

    Precedence:
    1. Explicit ``cli_arg`` (from ``--correlation-id``).
    2. ``$RUN_TOKEN`` env var.
    3. ``None`` — the Python API returns None so callers can choose their
       own resolution. The :func:`main` function (CLI entrypoint)
       auto-generates when both sources are absent (config#4223) — the
       CLI never errors on a missing correlation id.
    """
    if cli_arg:
        return cli_arg
    env_val = os.environ.get(CORRELATION_ID_ENV_VAR)
    if env_val:
        return env_val
    return None


def _auto_generate_correlation_id() -> str:
    """Best-effort correlation id for the no-flag/no-env case (config#4223).

    Precedence:
    1. ``$SSM_COMMAND_ID`` — set by the SSM agent on the box for
       long-running commands, so an SSM-dispatched workload without an
       explicit id still gets a stable, execution-scoped one.
    2. A fresh uuid4.

    The caller logs at WARNING when this path fires, so operators know
    traceability is degraded (no cross-surface correlation was declared).
    """
    ssm_id = os.environ.get(SSM_COMMAND_ID_ENV_VAR)
    if ssm_id:
        return ssm_id
    return str(uuid.uuid4())


def _write_correlation_header(logf, correlation_id: str | None) -> None:
    """Write the ``# correlation-id: ...`` header line into the log file.

    Best-effort (never raises). Written before the inner command's output
    so every captured log file carries the correlation id as its first
    line.
    """
    if not correlation_id:
        return
    try:
        header = f"# correlation-id: {correlation_id}\n".encode("utf-8")
        logf.write(header)
        logf.flush()
    except Exception:
        pass


def run(
    slug: str,
    log_path: Path | str,
    cmd: list[str],
    *,
    bucket: str | None = None,
    env: dict[str, str] | None = None,
    correlation_id: str | None = None,
    step_name: str | None = None,
) -> int:
    """Run ``cmd``, tee output to ``log_path`` and parent stdout, ship the log on exit.

    Mirrors the pre-lift inline pattern::

        bash <launcher> ... 2>&1 | tee /var/log/<slug>.log
        # plus: trap 'aws s3 cp /var/log/<slug>.log "s3://..." || true' EXIT

    When the inner command fails (non-zero exit), a formatted failure
    message naming the failing step + last output line is printed to
    stderr — never a bare exit code (§116 rule 4 "no naked rc").

    When ``correlation_id`` is resolved (explicitly or from the env var),
    a ``# correlation-id: <id>`` header line is written at the top of the
    captured log file, the value is appended to the S3 key
    (``-{correlation_id}`` before ``.log``), and it is exported to the inner
    command as ``$RUN_TOKEN`` so a launcher that dispatches nested work
    carries the same id rather than minting its own.

    Args:
        slug: log slug used in the S3 key (e.g., ``"morning-enrich"``).
        log_path: local log path to tee to (e.g., ``"/var/log/morning-enrich.log"``).
        cmd: inner command as a list of argv (passed to subprocess
            directly — no shell parsing, no quoting surface).
        bucket: S3 bucket override (default: ``alpha-engine-research``).
        env: environment override for the subprocess (default: inherit).
            Supplies the base environment only — the resolved correlation id
            is layered on top as ``$RUN_TOKEN``, so the parent log and every
            nested log agree on one id.
        correlation_id: run/execution correlation id for the log surface.
            When ``None``, resolved from ``$RUN_TOKEN`` env var. The CLI
            enforces presence as the §116 rule 6 chokepoint; the Python
            API is lenient for backwards compatibility.
        step_name: human-readable name for the failing step in the
            ``format_subprocess_failure`` message. Defaults to ``slug``
            when not provided.

    Returns:
        Inner command's exit code. ``127`` if the subprocess could not
        start (matches bash ``command not found`` convention).
    """
    bucket = bucket or DEFAULT_BUCKET
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_correlation_id = _resolve_correlation_id(correlation_id)
    step = step_name or slug

    last_output_line: Optional[str] = None
    # The last CAUSE-SHAPED line, which is rarely the last line: a failing
    # phase logs its error and the run continues to a generic exit
    # (config-I6945).
    last_cause_line: Optional[str] = None
    exit_code = 1
    try:
        with open(log_path, "wb") as logf:
            _write_correlation_header(logf, resolved_correlation_id)
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=_child_env(env, resolved_correlation_id),
            )
            assert proc.stdout is not None
            fd = proc.stdout.fileno()
            # Buffer for tracking last meaningful line across chunk boundaries
            partial = ""
            while True:
                chunk = os.read(fd, 8192)
                if not chunk:
                    break
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
                logf.write(chunk)
                logf.flush()
                # Track last non-empty output line for the failure message
                decoded = chunk.decode("utf-8", errors="replace")
                lines = (partial + decoded).split("\n")
                partial = lines.pop() if lines else ""
                for line in lines:
                    stripped = line.strip()
                    if stripped:
                        last_output_line = stripped
                        if _CAUSE_RE.search(stripped):
                            last_cause_line = stripped
            proc.wait()
            exit_code = proc.returncode
    except FileNotFoundError as exc:
        msg = f"krepis.ssm_log_capture: cannot exec {cmd!r}: {exc}\n"
        _append_log(log_path, msg)
        print(msg, file=sys.stderr)
        exit_code = 127
    except Exception as exc:
        msg = f"krepis.ssm_log_capture: subprocess setup failed: {type(exc).__name__}: {exc}\n"
        _append_log(log_path, msg)
        print(msg, file=sys.stderr)
        exit_code = 127

    try:
        # Print formatted failure message on non-zero exit (§116 rule 4).
        if exit_code != 0:
            failure_msg = format_subprocess_failure(
                step,
                returncode=exit_code,
                last_output_line=last_output_line,
                cause_line=last_cause_line,
            )
            print(failure_msg, file=sys.stderr)
            _append_log(log_path, failure_msg + "\n")
    except Exception:
        pass

    try:
        ok, detail = _ship_log_to_s3(
            slug, log_path, bucket, correlation_id=resolved_correlation_id
        )
        if ok:
            logger.info("ssm_log_capture: shipped %s", detail)
            print(f"ssm_log_capture: shipped {detail}", file=sys.stderr)
        else:
            logger.warning("ssm_log_capture: ship failed (%s)", detail)
            print(f"ssm_log_capture: log ship to S3 FAILED: {detail}", file=sys.stderr)
    except Exception:
        pass

    return exit_code


def _append_log(log_path: Path, msg: str) -> None:
    """Best-effort append to the log file. Never raises."""
    try:
        with open(log_path, "ab") as logf:
            logf.write(msg.encode("utf-8"))
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m krepis.ssm_log_capture",
        description=(
            "Run an inner command with stdout/stderr tee'd to a local log "
            "file + parent stdout, ship the log to S3 on exit, propagate "
            "the inner exit code. The institutional replacement for the "
            "inline `trap 'aws s3 cp ...' EXIT` pattern that broke under "
            "ASL States.Array escape semantics (alpha-engine-data PR #244 "
            "→ this lift)."
        ),
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    run_p = subparsers.add_parser(
        "run",
        help="Run a command with log capture + S3 ship-on-exit.",
    )
    run_p.add_argument(
        "--slug",
        required=True,
        help=(
            "Log slug for the S3 key (e.g., 'morning-enrich'). Identifies "
            "the SSM step under the _ssm_logs/ tree."
        ),
    )
    run_p.add_argument(
        "--log",
        required=True,
        help="Local log file path (e.g., /var/log/morning-enrich.log).",
    )
    run_p.add_argument(
        "--bucket",
        default=None,
        help=f"S3 bucket override (default: {DEFAULT_BUCKET}).",
    )
    run_p.add_argument(
        "--step-name",
        default=None,
        help=(
            "Human-readable name for the failing step in the failure message "
            "(defaults to --slug when not provided)."
        ),
    )
    run_p.add_argument(
        "--correlation-id",
        default=None,
        help=(
            "Run/execution correlation id embedded in the log file header "
            "and S3 key for traceability. Falls back to $RUN_TOKEN env var, "
            "then to an auto-generated id ($SSM_COMMAND_ID or uuid4) with a "
            "WARNING — the fleet §116 rule 6 chokepoint (config#4223): "
            "absence degrades gracefully, never hard-fails."
        ),
    )
    run_p.add_argument(
        "inner_cmd",
        nargs=argparse.REMAINDER,
        help=(
            "Inner command after `--`, e.g., "
            "`-- bash infrastructure/spot_data_weekly.sh --morning-enrich-only`."
        ),
    )

    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING)

    inner = args.inner_cmd or []
    if inner and inner[0] == "--":
        inner = inner[1:]
    if not inner:
        parser.error("inner command required after `--`")

    # §116 rule 6 chokepoint: every dispatched workload carries a correlation
    # id. When none is declared (flag or $RUN_TOKEN), AUTO-GENERATE one
    # (config#4223) and log at WARNING so operators know traceability is
    # degraded. A hard failure here is what killed the 2026-07-25 weekly SF
    # mid-execution — the krepis update deployed between SF phases took every
    # downstream ssm_log_capture caller down with it. The chokepoint's job is
    # that no log surface LACKS an id, and auto-generation satisfies that;
    # the flag/env remain the declared-intent signal.
    correlation_id = args.correlation_id or os.environ.get(CORRELATION_ID_ENV_VAR)
    if not correlation_id:
        correlation_id = _auto_generate_correlation_id()
        logger.warning(
            "ssm_log_capture: no --correlation-id and no $%s — auto-generated "
            "correlation id %r (traceability degraded: logs will not correlate "
            "to the triggering execution without it)",
            CORRELATION_ID_ENV_VAR, correlation_id,
        )

    return run(
        args.slug,
        args.log,
        list(inner),
        bucket=args.bucket,
        correlation_id=correlation_id,
        step_name=args.step_name,
    )


if __name__ == "__main__":
    sys.exit(main())
