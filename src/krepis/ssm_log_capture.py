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
- CLI: ``python -m krepis.ssm_log_capture run --slug <X>
  --log /var/log/<X>.log -- <inner-cmd...>``. Designed for SF JSON;
  a single ``States.Format`` template with ``$.preflight_args``
  interpolated via ``{}`` produces the entire invocation as one
  un-quoted token list — no bash trap, no inner single quotes.

**S3 layout:**

``s3://{bucket}/_ssm_logs/{slug}/{YYYY-MM-DD}/{hostname}-{HHMMSSZ}.log``

Defaults: ``bucket=alpha-engine-research``, prefix ``_ssm_logs``. Date,
time, and hostname are computed at exit time (so a multi-hour run that
straddles UTC midnight gets the actual exit-side date in the key).

**Failure behavior — never raises:**

- Inner command's exit code is propagated verbatim. Subprocess setup
  failure (e.g., ``FileNotFoundError`` on the binary) is logged to the
  log file and stderr, returns 127 to match the bash convention.
- S3 upload failures (boto3 ``ClientError``, missing creds, missing
  log file) are logged at WARNING and swallowed. The SF Catch must
  see the true inner exit, not a secondary log-capture failure that
  would mask it. Matches :mod:`krepis.alerts`' fail-safe
  posture.
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)

DEFAULT_BUCKET: Final[str] = "alpha-engine-research"
S3_PREFIX: Final[str] = "_ssm_logs"


def _exit_key(slug: str, *, now: datetime | None = None, host: str | None = None) -> str:
    """Compute the S3 key for the log upload at exit time.

    Public for tests; the canonical layout is
    ``_ssm_logs/{slug}/{YYYY-MM-DD}/{hostname}-{HHMMSSZ}.log``.
    """
    now = now or datetime.now(timezone.utc)
    host = host or socket.gethostname()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H%M%SZ")
    return f"{S3_PREFIX}/{slug}/{date_str}/{host}-{time_str}.log"


def _ship_log_to_s3(slug: str, log_path: Path, bucket: str) -> tuple[bool, str]:
    """Upload ``log_path`` to S3.

    Returns ``(ok, detail)``. Never raises. Computes the key at call
    time so the timestamp reflects when the trap fires, not when the
    wrapper started.
    """
    key = _exit_key(slug)
    if not log_path.exists():
        return False, f"log file not found: {log_path}"
    try:
        import boto3

        s3 = boto3.client("s3")
        s3.upload_file(str(log_path), bucket, key)
        return True, f"s3://{bucket}/{key}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def run(
    slug: str,
    log_path: Path | str,
    cmd: list[str],
    *,
    bucket: str | None = None,
    env: dict[str, str] | None = None,
) -> int:
    """Run ``cmd``, tee output to ``log_path`` and parent stdout, ship the log on exit.

    Mirrors the pre-lift inline pattern::

        bash <launcher> ... 2>&1 | tee /var/log/<slug>.log
        # plus: trap 'aws s3 cp /var/log/<slug>.log "s3://..." || true' EXIT

    Args:
        slug: log slug used in the S3 key (e.g., ``"morning-enrich"``).
        log_path: local log path to tee to (e.g., ``"/var/log/morning-enrich.log"``).
        cmd: inner command as a list of argv (passed to subprocess
            directly — no shell parsing, no quoting surface).
        bucket: S3 bucket override (default: ``alpha-engine-research``).
        env: environment override for the subprocess (default: inherit).

    Returns:
        Inner command's exit code. ``127`` if the subprocess could not
        start (matches bash ``command not found`` convention).
    """
    bucket = bucket or DEFAULT_BUCKET
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    exit_code = 1
    try:
        with open(log_path, "wb") as logf:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env if env is not None else os.environ.copy(),
            )
            assert proc.stdout is not None
            fd = proc.stdout.fileno()
            while True:
                chunk = os.read(fd, 8192)
                if not chunk:
                    break
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
                logf.write(chunk)
                logf.flush()
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
    finally:
        ok, detail = _ship_log_to_s3(slug, log_path, bucket)
        if ok:
            logger.info("ssm_log_capture: shipped %s", detail)
            print(f"ssm_log_capture: shipped {detail}", file=sys.stderr)
        else:
            logger.warning("ssm_log_capture: ship failed (%s)", detail)
            print(f"ssm_log_capture: log ship to S3 FAILED: {detail}", file=sys.stderr)

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

    return run(args.slug, args.log, list(inner), bucket=args.bucket)


if __name__ == "__main__":
    sys.exit(main())
