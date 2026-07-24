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
correlation id. When neither is present the CLI errors before running —
a new dispatched workload cannot use this module without providing one.
The correlation id appears as a ``# correlation-id: <id>`` header line at
the top of the captured log file and is appended to the S3 key for
traceability.
"""

from __future__ import annotations

import argparse
import logging
import os
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


def format_subprocess_failure(
    step_name: str,
    *,
    returncode: int,
    last_output_line: str | None = None,
) -> str:
    """Format a terminal-failure message naming the failing step + last output line.

    The §116 rule 4 ("no naked rc") chokepoint: callers that use this
    function on their exit path never surface a bare ``exit status 1``
    without context.  The message includes the step name, numeric return
    code, and — when available — the last non-empty line the failing
    subprocess emitted.

    Args:
        step_name: human-readable label for the failing step
            (e.g. ``"morning-enrich"``, ``"spot-train"``).
        returncode: the subprocess return code (non-zero indicates failure).
        last_output_line: the last non-empty line of merged stdout/stderr
            captured from the failing subprocess, or ``None`` when no
            output was captured (setup failure, empty output).

    Returns:
        A single-line string suitable for ``stderr``. New callers should
        call this in preference to re-deriving ``sf_watch_run.sh``'s
        bespoke pattern.

    Example output::

        ssm_log_capture: ERROR: [spot-train] failed (rc=1) — AssertionError: validation failed
    """
    if last_output_line:
        return (
            f"ssm_log_capture: ERROR: [{step_name}] failed "
            f"(rc={returncode}) — {last_output_line}"
        )
    return (
        f"ssm_log_capture: ERROR: [{step_name}] failed "
        f"(rc={returncode}) — no output captured"
    )


def _resolve_correlation_id(cli_arg: str | None) -> str | None:
    """Resolve correlation id from CLI arg, env var, or auto-generate.

    Precedence:
    1. Explicit ``cli_arg`` (from ``--correlation-id``).
    2. ``$RUN_TOKEN`` env var.
    3. ``None`` — the CLI enforces presence when ``--correlation-id`` is
       absent, but the Python API returns None so callers can choose their
       own resolution. The :func:`main` function (CLI entrypoint) errors
       before running if both sources are absent.
    """
    if cli_arg:
        return cli_arg
    env_val = os.environ.get(CORRELATION_ID_ENV_VAR)
    if env_val:
        return env_val
    return None


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
    captured log file, and the value is appended to the S3 key
    (``-{correlation_id}`` before ``.log``).

    Args:
        slug: log slug used in the S3 key (e.g., ``"morning-enrich"``).
        log_path: local log path to tee to (e.g., ``"/var/log/morning-enrich.log"``).
        cmd: inner command as a list of argv (passed to subprocess
            directly — no shell parsing, no quoting surface).
        bucket: S3 bucket override (default: ``alpha-engine-research``).
        env: environment override for the subprocess (default: inherit).
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
    exit_code = 1
    try:
        with open(log_path, "wb") as logf:
            _write_correlation_header(logf, resolved_correlation_id)
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env if env is not None else os.environ.copy(),
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
            "and S3 key for traceability. Falls back to $RUN_TOKEN env var. "
            "Required by the fleet §116 rule 6 chokepoint: when both are "
            "absent, the command refuses to run."
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

    # §116 rule 6 chokepoint: every dispatched workload must carry a
    # correlation id. Reject before running so the gap is never silent.
    correlation_id = args.correlation_id or os.environ.get(CORRELATION_ID_ENV_VAR)
    if not correlation_id:
        parser.error(
            "--correlation-id is required (or set $RUN_TOKEN) — the fleet "
            "§116 rule 6 (logging standard) requires every log surface to "
            "carry a run/execution correlation id for traceability. Without "
            "one, a new dispatched workload's logs cannot be correlated back "
            "to its triggering execution."
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
