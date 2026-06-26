"""
SSM send-command + poll-for-completion chokepoint.

Consolidation substrate for the ``run_ssm`` bash helper that previously
appeared as a ~54-line mirror in every dispatcher script that drives a
spot instance over the SSM transport. The first occurrence shipped in
alpha-engine-predictor #168 (2026-05-15) as part of the SSH/SCP→SSM
migration; the second and third occurrences land when alpha-engine-data's
``spot_data_weekly.sh`` and alpha-engine-backtester's ``spot_backtest.sh``
migrate off SSH+SCP onto the same SSM transport. Per
``~/Development/CLAUDE.md`` SOTA sub-sub-rule + the
``[[feedback_lift_invariants_to_chokepoint_after_second_recurrence]]``
discipline, the pattern lifts to lib at the second recurrence.

The pre-lift bash shape was::

    run_ssm "<description>" "<bash script>" [timeout_seconds]
    # 1. base64-encode the script body (transport-safe wrapping of inner
    #    heredocs / quoting)
    # 2. aws ssm send-command --document-name AWS-RunShellScript \
    #      --instance-ids "$INSTANCE_ID" \
    #      --output-s3-bucket-name "$S3_BUCKET" \
    #      --output-s3-key-prefix "${S3_STAGING_PREFIX}/ssm-output" \
    #      --timeout-seconds "$timeout_s" \
    #      --parameters file://$pfile
    # 3. while :; do
    #      aws ssm get-command-invocation --command-id $cmd_id
    #      stream stdout delta; check Status; break on terminal
    #    done
    # 4. on Success → return 0
    # 5. on Failed/TimedOut/Cancelled → fetch stderr, print, return 1

The Python primitive in this module exposes the same contract — base64
wrap, send, poll, stream, propagate exit — but lives in one place so
the polling cadence, error-class handling, and S3 output-key layout
match across every consumer.

**Why a CLI, not a bash function:**

Per the SOTA / institutional-approach sub-sub-rule ("when mirroring a
pattern across repos, consider lifting it into ``nousergon-lib``...
Pure-Bash primitives can stay mirrored unless re-expressible as a
Python CLI entry callable from Bash, in which case the CLI re-expression
is the institutional path"). The dispatcher script invokes::

    python -m krepis.ssm_dispatcher run \\
      --instance-id "$INSTANCE_ID" \\
      --description "bootstrap" \\
      --timeout 3600 \\
      --output-bucket "$S3_BUCKET" \\
      --output-key-prefix "${S3_STAGING_PREFIX}/ssm-output" \\
      --region "$AWS_REGION" \\
      --script-stdin <<'BOOTSTRAP'
    set -eo pipefail
    ...
    BOOTSTRAP

Exit code 0 on Success; 1 on terminal non-Success; 2 on bad input. The
inner script's stdout streams to the dispatcher's stdout as it arrives
(SSM ``StandardOutputContent`` delta); on terminal non-Success the
``StandardErrorContent`` is fetched + printed before the dispatcher
exits.

**InvocationDoesNotExist race:**

After ``send-command`` returns a ``CommandId``, the first poll of
``get-command-invocation`` can race the SSM control plane's registration
and return ``InvocationDoesNotExist``. The 2026-05-23 Saturday SF showed
this exact failure mode at event 16 (MorningEnrich first poll), absorbed
by the SF Catch but representing a substrate weakness. This module
treats ``InvocationDoesNotExist`` as a transient "Pending" status for
the first ~60s after SendCommand (the registration window) and as a
terminal failure thereafter. Mirrors the bash predecessor's
``2>/dev/null || echo Pending`` swallow without the all-errors-look-like-Pending
ambiguity.

**Failure behavior — never raises:**

- Inner command's terminal status maps to exit code 0 (Success) or 1
  (Failed / TimedOut / Cancelled / TerminalError). The dispatcher
  script's ``set -e`` then propagates that exit upward to the SF Catch.
- Subprocess setup failure (boto3 missing, IAM denied at SendCommand
  time, instance not registered) is logged + returns 1. The caller
  reads the failure from CloudWatch / SSM history; this module's job
  is to be a thin transport, not a recovery layer.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Final, Optional

logger = logging.getLogger(__name__)

# Status taxonomy from SSM's get-command-invocation. Terminal non-Success
# statuses all map to exit 1.
TERMINAL_NON_SUCCESS: Final[frozenset[str]] = frozenset(
    {"Cancelled", "Failed", "TimedOut", "Cancelling", "TerminalError"}
)
PENDING_STATUSES: Final[frozenset[str]] = frozenset(
    {"Pending", "InProgress", "Delayed"}
)
SUCCESS_STATUS: Final[str] = "Success"

# Window during which InvocationDoesNotExist counts as a registration race
# rather than a true failure. Mirrors the empirical observation that the
# SSM control plane has settled by ~30s post-SendCommand under normal
# conditions; 60s is a defensive ceiling.
REGISTRATION_GRACE_SECONDS: Final[int] = 60

# Poll cadence — matches the bash predecessor's `sleep 5`.
DEFAULT_POLL_INTERVAL_SECONDS: Final[float] = 5.0

# StandardOutputContent / StandardErrorContent fields are capped at 24KB
# in get-command-invocation responses. Beyond the cap the buffer rotates
# (we detect by a length decrease) and the full log lives in the
# configured S3 output prefix.
SSM_INLINE_OUTPUT_CAP_BYTES: Final[int] = 24 * 1024

# Stdout/stderr tail length captured in the diagnostics JSON written on
# terminal non-Success. Chosen at 4KB to mirror the typical pre-lift bash
# diagnostic posture (operators want enough tail to grep the failure
# signature without consuming the full SSM inline cap); the full log
# lives in --output-bucket when configured.
DIAGNOSTICS_TAIL_BYTES: Final[int] = 4 * 1024


class SsmDispatchError(Exception):
    """Non-recoverable SSM send-command / poll failure."""


def _tail_bytes(text: str, max_bytes: int = DIAGNOSTICS_TAIL_BYTES) -> str:
    """Return the last ``max_bytes`` of ``text``, snapping to a line boundary.

    Used to bound the size of stdout/stderr payloads embedded in the
    diagnostics JSON. Snap to a newline so the tail starts at a clean
    line break rather than mid-line — operators grepping for a failure
    signature get a coherent prefix instead of a truncated token. If no
    newline exists in the window, the raw byte tail is returned.
    """
    if not text:
        return ""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    tail = encoded[-max_bytes:].decode("utf-8", errors="replace")
    nl = tail.find("\n")
    if nl != -1 and nl < len(tail) - 1:
        return tail[nl + 1 :]
    return tail


def _ship_diagnostics(
    *,
    bucket: str,
    prefix: str,
    status: str,
    command_id: str,
    description: str,
    exit_window_utc: str,
    stdout_tail: str,
    stderr_tail: str,
    instance_id: str,
    boto3_client=None,
    stderr_stream=None,
) -> tuple[bool, str]:
    """Write the terminal-non-Success diagnostics JSON to S3.

    Returns ``(ok, detail)``. **Never raises** — mirrors the failure-mode
    posture of ``krepis.ssm_log_capture._ship_log_to_s3``. Any
    S3 upload failure (NoCredentialsError, AccessDenied, transient
    network) is logged to ``stderr_stream`` + swallowed; the inner
    dispatcher exit code is preserved.

    Key shape: ``s3://{bucket}/{prefix}/{YYYY-MM-DD}.json``. The date
    component is the UTC date at exit-window time. The current spec
    permits one diagnostics file per ``{prefix}`` per UTC day; consumer
    dispatchers that need multi-failure-per-day discrimination should
    incorporate the repo or stage name into ``--diagnostics-prefix``.
    """
    err = stderr_stream if stderr_stream is not None else sys.stderr
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    key = f"{prefix.rstrip('/')}/{date_str}.json"
    payload = {
        "status": status,
        "command_id": command_id,
        "description": description,
        "exit_window_utc": exit_window_utc,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
        "instance_id": instance_id,
    }
    try:
        if boto3_client is None:
            import boto3

            s3 = boto3.client("s3")
        else:
            s3 = boto3_client
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(payload, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
        return True, f"s3://{bucket}/{key}"
    except Exception as exc:
        msg = (
            f"ssm_dispatcher: diagnostics-write to s3://{bucket}/{key} "
            f"failed (swallowed; inner exit preserved): "
            f"{type(exc).__name__}: {exc}\n"
        )
        err.write(msg)
        err.flush()
        return False, f"{type(exc).__name__}: {exc}"


def _encode_command_payload(script: str) -> str:
    """Wrap ``script`` for AWS-RunShellScript transport.

    The pre-lift bash helper base64-encoded the script body and emitted
    a single command ``echo <b64> | base64 -d | bash``. This is the
    transport-safe wrapping that lets the script contain heredocs,
    embedded Python, single quotes, etc. without ASL/SSM escaping
    surface.
    """
    b64 = base64.b64encode(script.encode("utf-8")).decode("ascii")
    return f"echo {b64} | base64 -d | bash"


def run(
    instance_id: str,
    description: str,
    script: str,
    *,
    timeout_seconds: int = 3600,
    output_bucket: Optional[str] = None,
    output_key_prefix: Optional[str] = None,
    region: str = "us-east-1",
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    diagnostics_bucket: Optional[str] = None,
    diagnostics_prefix: Optional[str] = None,
    stdout_stream=None,
    stderr_stream=None,
    sleep=time.sleep,
    monotonic=time.monotonic,
    boto3_client=None,
    s3_client=None,
) -> int:
    """Send ``script`` to ``instance_id`` via SSM, poll until terminal, stream stdout.

    Args:
        instance_id: target EC2 instance ID (must be SSM-registered).
        description: short label for SSM history + dispatcher logs.
        script: bash script body. Will be base64-wrapped + executed as
            a single AWS-RunShellScript command.
        timeout_seconds: SSM command timeout (handed to SendCommand).
        output_bucket: S3 bucket for SSM to write the full stdout/stderr
            (past the 24KB inline cap). Optional; if unset, only inline
            output is available.
        output_key_prefix: S3 key prefix for the SSM output bucket.
        region: AWS region.
        poll_interval_seconds: gap between get-command-invocation polls.
        diagnostics_bucket: S3 bucket for the terminal-non-Success
            diagnostics JSON (L369). When both ``diagnostics_bucket`` and
            ``diagnostics_prefix`` are set, every terminal non-Success
            outcome writes ``{prefix}/{YYYY-MM-DD}.json`` with status +
            command_id + description + exit_window + stdout/stderr tails
            + instance_id. Best-effort; S3 failure is swallowed so the
            inner exit code is preserved.
        diagnostics_prefix: S3 key prefix for diagnostics JSON. Typical
            shape from consumer dispatchers: ``_spot_diagnostics/{repo}``.
        stdout_stream: destination for streamed inner stdout (default:
            ``sys.stdout``).
        stderr_stream: destination for the terminal-failure stderr dump
            (default: ``sys.stderr``).
        sleep / monotonic: time hooks (overridable for tests).
        boto3_client: optional boto3 ``ssm`` client (for tests). When
            ``None``, constructed via ``boto3.client('ssm', region_name=region)``.
        s3_client: optional boto3 ``s3`` client used only for the
            diagnostics-write path. When ``None`` and a diagnostics-write
            is triggered, constructed via ``boto3.client('s3')``.

    Returns:
        ``0`` on terminal Success.
        ``1`` on any terminal non-Success status, send-command failure,
        or unrecoverable poll failure.

    Never raises.
    """
    out = stdout_stream if stdout_stream is not None else sys.stdout
    err = stderr_stream if stderr_stream is not None else sys.stderr

    try:
        if boto3_client is None:
            import boto3

            ssm = boto3.client("ssm", region_name=region)
        else:
            ssm = boto3_client
    except Exception as exc:
        print(
            f"ssm_dispatcher: boto3 client construction failed: "
            f"{type(exc).__name__}: {exc}",
            file=err,
        )
        return 1

    payload = _encode_command_payload(script)
    send_kwargs: dict = {
        "InstanceIds": [instance_id],
        "DocumentName": "AWS-RunShellScript",
        "Comment": description[:100],  # SSM Comment cap is 100 chars
        "TimeoutSeconds": int(timeout_seconds),
        "Parameters": {"commands": [payload]},
    }
    if output_bucket:
        send_kwargs["OutputS3BucketName"] = output_bucket
    if output_key_prefix:
        send_kwargs["OutputS3KeyPrefix"] = output_key_prefix

    try:
        resp = ssm.send_command(**send_kwargs)
    except Exception as exc:
        print(
            f"ssm_dispatcher: send_command failed for {description!r}: "
            f"{type(exc).__name__}: {exc}",
            file=err,
        )
        return 1

    command_id = resp.get("Command", {}).get("CommandId")
    if not command_id:
        print(
            f"ssm_dispatcher: send_command returned no CommandId for {description!r}",
            file=err,
        )
        return 1

    print(f"    [ssm {description}] command-id={command_id}", file=err)

    start_monotonic = monotonic()
    last_out_len = 0

    while True:
        sleep(poll_interval_seconds)

        try:
            inv = ssm.get_command_invocation(
                CommandId=command_id,
                InstanceId=instance_id,
            )
        except Exception as exc:
            code = _classify_boto_exception(exc)
            if code == "InvocationDoesNotExist":
                elapsed = monotonic() - start_monotonic
                if elapsed <= REGISTRATION_GRACE_SECONDS:
                    # Registration race per the 2026-05-23 Saturday SF
                    # event-16 substrate weakness; keep polling.
                    continue
                print(
                    f"ssm_dispatcher: {description!r} command {command_id} "
                    f"never registered (InvocationDoesNotExist after "
                    f"{elapsed:.0f}s)",
                    file=err,
                )
                return 1
            # Other transient classes that the bash predecessor swallowed
            # via `2>/dev/null || echo Pending`. Be explicit: only the
            # listed set is treated as transient; anything else is a hard
            # failure.
            if code in {"ThrottlingException", "RequestLimitExceeded"}:
                continue
            print(
                f"ssm_dispatcher: get_command_invocation for {description!r} "
                f"raised {code}: {exc}",
                file=err,
            )
            return 1

        status = inv.get("Status", "Pending")
        std_out = inv.get("StandardOutputContent", "") or ""

        if len(std_out) > last_out_len:
            out.write(std_out[last_out_len:])
            out.flush()
            last_out_len = len(std_out)
        elif len(std_out) < last_out_len:
            # 24KB cap rotated the buffer; the full log is in S3 (if
            # output_bucket was configured).
            cap_note = (
                f"    [ssm {description}] (stdout exceeded "
                f"{SSM_INLINE_OUTPUT_CAP_BYTES // 1024}KB cap — full log: "
                f"s3://{output_bucket}/{output_key_prefix}/)\n"
                if output_bucket
                else (
                    f"    [ssm {description}] (stdout exceeded "
                    f"{SSM_INLINE_OUTPUT_CAP_BYTES // 1024}KB cap — "
                    "configure --output-bucket for full log)\n"
                )
            )
            err.write(cap_note)
            err.flush()
            last_out_len = len(std_out)

        if status == SUCCESS_STATUS:
            return 0
        if status in TERMINAL_NON_SUCCESS:
            std_err = inv.get("StandardErrorContent", "") or ""
            err.write(
                f"ERROR: SSM step {description!r} terminal status={status}\n"
            )
            if std_err:
                err.write(
                    f"--- stderr ({SSM_INLINE_OUTPUT_CAP_BYTES // 1024}KB cap; "
                )
                if output_bucket:
                    err.write(
                        f"full: s3://{output_bucket}/{output_key_prefix}/) ---\n"
                    )
                else:
                    err.write("configure --output-bucket for full log) ---\n")
                err.write(std_err)
                if not std_err.endswith("\n"):
                    err.write("\n")
            err.flush()
            if diagnostics_bucket and diagnostics_prefix:
                exit_window_utc = datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                )
                ok, detail = _ship_diagnostics(
                    bucket=diagnostics_bucket,
                    prefix=diagnostics_prefix,
                    status=status,
                    command_id=command_id,
                    description=description,
                    exit_window_utc=exit_window_utc,
                    stdout_tail=_tail_bytes(std_out),
                    stderr_tail=_tail_bytes(std_err),
                    instance_id=instance_id,
                    boto3_client=s3_client,
                    stderr_stream=err,
                )
                if ok:
                    err.write(
                        f"    [ssm {description}] diagnostics → {detail}\n"
                    )
                    err.flush()
            return 1
        if status not in PENDING_STATUSES:
            # Unknown status — treat as a hard failure, log it.
            err.write(
                f"ssm_dispatcher: {description!r} returned unknown status "
                f"{status!r}; treating as failure\n"
            )
            err.flush()
            return 1
        # Pending / InProgress / Delayed — keep polling.


def _classify_boto_exception(exc: BaseException) -> str:
    """Extract the ``Error.Code`` from a botocore ClientError.

    Returns the exception class name when no ``response.Error.Code`` is
    available (e.g., on non-botocore exceptions). Tests patch this for
    deterministic InvocationDoesNotExist surfacing.
    """
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        code = response.get("Error", {}).get("Code")
        if code:
            return str(code)
    return type(exc).__name__


def _read_script(args: argparse.Namespace) -> str:
    if args.script_file:
        with open(args.script_file, "r", encoding="utf-8") as fh:
            return fh.read()
    if args.script_stdin:
        return sys.stdin.read()
    raise SystemExit(
        "ssm_dispatcher: must pass either --script-file PATH or --script-stdin "
        "(with the script body on stdin)"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m krepis.ssm_dispatcher",
        description=(
            "Send a bash script to an SSM-registered EC2 instance via "
            "AWS-RunShellScript, poll until terminal, stream stdout to "
            "this process, and propagate the inner exit status. The "
            "institutional replacement for the ~54-line run_ssm bash "
            "helper mirrored across alpha-engine-* dispatcher scripts."
        ),
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    run_p = subparsers.add_parser(
        "run",
        help="Dispatch a script to an instance and stream its output.",
    )
    run_p.add_argument(
        "--instance-id",
        required=True,
        help="Target EC2 instance ID (must be SSM-registered).",
    )
    run_p.add_argument(
        "--description",
        required=True,
        help=(
            "Short label for the SSM command Comment + dispatcher log "
            "lines (e.g., 'bootstrap', 'full-training')."
        ),
    )
    run_p.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="SSM command timeout in seconds (default: 3600).",
    )
    run_p.add_argument(
        "--output-bucket",
        default=None,
        help=(
            "S3 bucket where SSM writes the full stdout/stderr beyond "
            "the inline 24KB cap. Optional; without it, only the inline "
            "delta is available."
        ),
    )
    run_p.add_argument(
        "--output-key-prefix",
        default=None,
        help="S3 key prefix under --output-bucket for the SSM output.",
    )
    run_p.add_argument(
        "--region",
        default=os.environ.get("AWS_REGION", "us-east-1"),
        help="AWS region (default: $AWS_REGION or us-east-1).",
    )
    run_p.add_argument(
        "--poll-interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL_SECONDS,
        help=(
            "Seconds between get-command-invocation polls (default: "
            f"{DEFAULT_POLL_INTERVAL_SECONDS:g})."
        ),
    )
    run_p.add_argument(
        "--diagnostics-bucket",
        default=None,
        help=(
            "S3 bucket for the terminal-non-Success diagnostics JSON (L369). "
            "When set together with --diagnostics-prefix, every terminal "
            "non-Success outcome writes a JSON record with status + "
            "command_id + stdout/stderr tails + instance_id. Best-effort; "
            "S3 failure is swallowed (never masks the inner exit code)."
        ),
    )
    run_p.add_argument(
        "--diagnostics-prefix",
        default=None,
        help=(
            "S3 key prefix for diagnostics JSON. Typical shape from "
            "consumer dispatchers: '_spot_diagnostics/{repo}'. Key: "
            "'{prefix}/{YYYY-MM-DD}.json'."
        ),
    )
    script_grp = run_p.add_mutually_exclusive_group(required=True)
    script_grp.add_argument(
        "--script-file",
        default=None,
        help="Path to a local file containing the bash script body.",
    )
    script_grp.add_argument(
        "--script-stdin",
        action="store_true",
        help="Read the bash script body from stdin (heredoc-friendly).",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    script = _read_script(args)
    if not script.strip():
        print(
            "ssm_dispatcher: empty script body (refusing to dispatch a no-op)",
            file=sys.stderr,
        )
        return 2

    return run(
        instance_id=args.instance_id,
        description=args.description,
        script=script,
        timeout_seconds=args.timeout,
        output_bucket=args.output_bucket,
        output_key_prefix=args.output_key_prefix,
        region=args.region,
        poll_interval_seconds=args.poll_interval,
        diagnostics_bucket=args.diagnostics_bucket,
        diagnostics_prefix=args.diagnostics_prefix,
    )


if __name__ == "__main__":
    sys.exit(main())
