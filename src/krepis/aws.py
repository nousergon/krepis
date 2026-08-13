"""AWS Lambda invoke helpers — the deploy-canary resilience chokepoint.

Consolidation substrate for the **"invoke a Lambda with bounded retry on the
throttle / reserved-concurrency-limit class, then fail loud"** idiom that was
mirrored across four alpha-engine ``deploy.sh`` canary blocks (crucible-research,
nousergon-data, crucible-predictor, crucible-evaluator — config#1494).

A canary ``aws lambda invoke`` can hit ``TooManyRequestsException`` /
``ReservedFunctionConcurrentInvocationLimitExceeded`` when the function's
concurrency slot is momentarily occupied — an overlapping deploy's canary
(cancelling a GitHub Actions run does NOT stop the Lambda execution it already
dispatched) or an in-flight scheduled invocation. The AWS CLI's own retry
(max 2, seconds-scale) can't outwait an in-flight execution, and under
``set -euo pipefail`` the invoke's non-zero exit aborted the whole deploy on a
transient smoke-test throttle (bit crucible-research CI 2026-07-01, config#1493).
Each ``deploy.sh`` grew its own Bash copy of "retry ONLY on the throttle signal,
bounded exp backoff + jitter, fail loud on exhaustion." This module is the
single source of truth for that policy so the four callsites stop drifting.

Two layers are exported, mirroring :mod:`krepis.http_retry`:

  * :func:`invoke_lambda_with_retry` — the full boto3 invoke-with-retry.
    Returns an :class:`InvokeResult` (the invoke API metadata + the response
    payload bytes); the caller still owns the FUNCTION's own-status
    interpretation (``OK`` / ``SKIPPED`` / a bad ``statusCode`` / a
    ``FunctionError``), exactly as it did when parsing ``aws lambda invoke``
    output. Raises :class:`LambdaInvokeError` on a non-retryable boto error or
    exhausted retries — the fail-loud signal, distinct from the function
    returning a bad status.
  * The Bash-callable CLI (``python -m krepis.aws invoke-canary``), mirroring
    ``krepis.alerts``: writes the response payload to ``--out`` and prints the
    invoke METADATA (StatusCode / FunctionError / ExecutedVersion) as JSON to
    stdout, so a Bash caller parses exactly what ``aws lambda invoke`` gave it
    before. Exit 0 once the invoke API call succeeds; non-zero on a
    non-throttle boto error or exhausted retries.

Design note (anti-over-engineering, per :mod:`krepis.http_retry`): this
captures the one invariant the four deploy canaries share — throttle-only
retry with fail-loud exhaustion. It is NOT a general Lambda-management wrapper.
The throttle timing reuses :func:`krepis.http_retry.backoff_delay` rather than
re-deriving the full-jitter math.
"""

from __future__ import annotations

import json as _json
import logging as _logging
import random as _random
import time as _time
from typing import Callable, Iterable

_DEFAULT_LOGGER = _logging.getLogger(__name__)


def _backoff_delay(
    attempt: int,
    *,
    base: float,
    cap: float,
    rng: "_random.Random | None" = None,
) -> float:
    """Full-jitter exponential backoff: ``min(base * 2**attempt + U(0, base), cap)``.

    ``attempt`` is 0-indexed. Deliberately inlined (not imported from
    :mod:`krepis.http_retry`) so this deploy-critical AWS module's import
    surface stays stdlib + boto3 — the canary invoke is load-bearing and should
    not pull in an HTTP-requests module for a 3-line formula. ``rng`` is
    injectable for deterministic tests.
    """
    wait = base * (2 ** attempt)
    jitter = (rng or _random).uniform(0, base)
    return min(wait + jitter, cap)

# The retryable class for a SYNCHRONOUS Lambda invoke. A concurrency slot
# momentarily held by an in-flight execution surfaces as boto
# ``TooManyRequestsException`` (the ``Reason`` detail —
# ``ReservedFunctionConcurrentInvocationLimitExceeded`` /
# ``ConcurrentInvocationLimitExceeded`` — rides in the error message). Every
# OTHER ``ClientError`` code (ResourceNotFound, AccessDenied, bad payload) is
# deterministic — retrying it is pointless, so it fails loud immediately.
DEFAULT_RETRYABLE_INVOKE_CODES: "frozenset[str]" = frozenset(
    {"TooManyRequestsException"}
)

# Canary defaults: ~3 min over 5 sleeps (full jitter) — generous enough to
# outwait an overlapping dry-run canary's cold-started execution, bounded so a
# genuinely stuck slot fails loud rather than hanging CI. min(5*2**a + U(0,5), 90):
# ~5, 10, 20, 40, 80s (+ jitter, capped at 90).
_DEFAULT_MAX_ATTEMPTS = 6
_DEFAULT_BACKOFF_BASE = 5.0
_DEFAULT_BACKOFF_CAP = 90.0


class LambdaInvokeError(RuntimeError):
    """Raised when a canary invoke cannot COMPLETE: a non-retryable boto
    ``ClientError`` (surfaced immediately) or the retryable throttle class
    surviving ``max_attempts``.

    This is the fail-loud signal for a deploy caller — it means "the smoke
    test never ran", which is categorically different from "the function ran
    and returned a bad status" (the caller judges that from the payload). The
    originating exception is preserved as ``__cause__``; ``.code`` /
    ``.attempts`` / ``.label`` carry context.
    """

    def __init__(self, label: str, attempts: int, code: str, message: str) -> None:
        self.label = label
        self.attempts = attempts
        self.code = code
        super().__init__(
            f"{label or 'invoke'} failed after {attempts} attempt(s): "
            f"{code or 'error'}: {message}"
        )


class InvokeResult:
    """The invoke API metadata plus the response payload bytes.

    ``status_code`` / ``function_error`` / ``executed_version`` are the invoke
    API metadata (``FunctionError`` lives here, NOT in the payload — the field
    predictor's canary parses). ``payload`` is the raw response body bytes (the
    function's own ``{"status": ...}`` / ``{"statusCode": ...}`` JSON).
    """

    def __init__(
        self,
        status_code: "int | None",
        function_error: "str | None",
        executed_version: "str | None",
        payload: bytes,
    ) -> None:
        self.status_code = status_code
        self.function_error = function_error
        self.executed_version = executed_version
        self.payload = payload

    def metadata_json(self) -> str:
        """Serialize the invoke metadata the way a Bash caller expects (the
        same keys ``aws lambda invoke`` prints to stdout)."""
        return _json.dumps(
            {
                "StatusCode": self.status_code,
                "FunctionError": self.function_error or "",
                "ExecutedVersion": self.executed_version or "",
            }
        )


def invoke_lambda_with_retry(
    function_name: str,
    payload: "bytes | str",
    *,
    region: "str | None" = None,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    backoff_base: float = _DEFAULT_BACKOFF_BASE,
    backoff_cap: float = _DEFAULT_BACKOFF_CAP,
    retryable_codes: Iterable[str] = DEFAULT_RETRYABLE_INVOKE_CODES,
    client=None,
    logger: "_logging.Logger | None" = None,
    label: str = "",
    sleep: Callable[[float], None] = _time.sleep,
) -> InvokeResult:
    """Invoke ``function_name`` (a name, ``name:alias``, or ``name:version``)
    with a JSON ``payload``, retrying ONLY on the throttle/concurrency class
    with bounded full-jitter backoff.

    Retries a ``ClientError`` whose code is in ``retryable_codes`` (default:
    ``TooManyRequestsException``) up to ``max_attempts``. Any other
    ``ClientError`` (ResourceNotFound / AccessDenied / bad payload) fails loud
    immediately. Exhausting the retryable class also fails loud. Both raise
    :class:`LambdaInvokeError`.

    Returns an :class:`InvokeResult` on the first successful invoke API call —
    the function's OWN status (a bad ``statusCode`` / ``FunctionError`` /
    payload ``{"status": "ERROR"}``) is the caller's to judge, exactly as when
    parsing ``aws lambda invoke`` output. ``client`` / ``sleep`` are injectable
    for tests. ``max_attempts`` must be >= 1.
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
    log = logger or _DEFAULT_LOGGER
    retryable = frozenset(retryable_codes)
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    if client is None:  # pragma: no cover — exercised via injected client in tests
        import boto3

        client = boto3.client("lambda", region_name=region)
    from botocore.exceptions import ClientError

    last_code = last_msg = ""
    for attempt in range(max_attempts):
        last = attempt == max_attempts - 1
        try:
            resp = client.invoke(FunctionName=function_name, Payload=payload)
        except ClientError as exc:
            err = exc.response.get("Error", {}) if hasattr(exc, "response") else {}
            code = err.get("Code", "") or ""
            msg = err.get("Message", "") or str(exc)
            last_code, last_msg = code, msg
            if code in retryable and not last:
                delay = _backoff_delay(attempt, base=backoff_base, cap=backoff_cap)
                log.warning(
                    "%s throttled (%s) — concurrency slot busy, backing off "
                    "%.1fs (attempt %d/%d)",
                    label or function_name,
                    code,
                    delay,
                    attempt + 1,
                    max_attempts,
                )
                sleep(delay)
                continue
            # Non-retryable, or the retryable class exhausted on the last
            # attempt — fail loud.
            raise LambdaInvokeError(
                label or function_name, attempt + 1, code, msg
            ) from exc

        # Invoke API call succeeded — read the streamed payload + metadata.
        body = resp.get("Payload")
        payload_bytes = body.read() if body is not None else b""
        return InvokeResult(
            status_code=resp.get("StatusCode"),
            function_error=resp.get("FunctionError"),
            executed_version=resp.get("ExecutedVersion"),
            payload=payload_bytes,
        )

    # Unreachable: the loop returns on success or raises on the last attempt.
    raise LambdaInvokeError(
        label or function_name, max_attempts, last_code, last_msg
    )  # pragma: no cover


class LambdaEnvMergeError(RuntimeError):
    """Raised when a Lambda environment merge could not be completed."""


def merge_lambda_environment(
    function_name: str,
    updates: "dict[str, str]",
    *,
    region: "str | None" = None,
    client: "object | None" = None,
) -> int:
    """Merge *updates* into a Lambda's environment. Returns the total var count.

    **Read-modify-write, never replace.** ``update-function-configuration
    --environment`` replaces the WHOLE variable map, and the fleet's
    functions carry provider keys, database URLs and operator-set flags
    that exist only on the live function and are codified nowhere. A
    deploy script that writes a fresh map deletes every one of them.

    **Nothing is echoed.** The current map is read into memory and merged
    in-process; only the resulting variable COUNT and the merged KEY names
    are printed. The values never reach stdout, a command line, or a shell
    trace.

    Why this lives in krepis rather than in each ``deploy.sh``: three
    alpha-engine repos need the same merge to turn on cost telemetry
    (``alpha-engine-config-I7179``), and the Bash+heredoc form of it had
    already been written once in crucible-research. A second and third copy
    is the drift mechanism ``shared-code-policy`` names, and this one is
    re-expressible as a Python CLI entry, which is where the fleet rule
    stops permitting a mirrored Bash primitive.
    """
    if not updates:
        raise LambdaEnvMergeError("merge_lambda_environment called with no updates")
    for key, value in updates.items():
        if not key or not isinstance(value, str) or value == "":
            raise LambdaEnvMergeError(
                f"refusing to merge an empty value for {key!r} — an empty "
                f"environment variable is indistinguishable from an unset "
                f"one at the reader and silently disables whatever it gates"
            )
    if client is None:
        import boto3  # imported lazily so `import krepis.aws` costs nothing

        client = boto3.client("lambda", region_name=region) if region else boto3.client("lambda")

    try:
        client.get_waiter("function_updated").wait(FunctionName=function_name)
    except Exception as exc:  # noqa: BLE001 — duck-typed boto errors
        # DELIBERATE non-raising degradation, with rationale:
        # (a) swallowed: the waiter failed (function has never been
        #     deployed, or the waiter is unavailable on a stubbed client).
        # (b) the primary deliverable survives because the update below is
        #     the load-bearing call and raises on its own failure — this
        #     wait only avoids a ResourceConflictException race.
        # (c) recorded here, and by the update's own exception if the race
        #     does bite.
        _DEFAULT_LOGGER.debug("function_updated waiter skipped for %s: %s", function_name, exc)

    try:
        config = client.get_function_configuration(FunctionName=function_name)
    except Exception as exc:  # noqa: BLE001
        raise LambdaEnvMergeError(
            f"could not read the current environment of {function_name}: {exc}"
        ) from exc
    variables = dict((config.get("Environment") or {}).get("Variables") or {})
    variables.update(updates)

    try:
        client.update_function_configuration(
            FunctionName=function_name,
            Environment={"Variables": variables},
        )
    except Exception as exc:  # noqa: BLE001
        raise LambdaEnvMergeError(
            f"could not write the merged environment of {function_name}: {exc}"
        ) from exc
    try:
        client.get_waiter("function_updated").wait(FunctionName=function_name)
    except Exception as exc:  # noqa: BLE001 — same rationale as above
        _DEFAULT_LOGGER.debug("post-update waiter skipped for %s: %s", function_name, exc)
    return len(variables)


def main(argv: "list[str] | None" = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="python -m krepis.aws",
        description=(
            "AWS Lambda helpers for deploy scripts. Bash-callable; mirrors "
            "krepis.alerts. Exit 0 once the invoke API call succeeds (the "
            "function's own status is the caller's to judge); non-zero on a "
            "non-throttle boto error or exhausted retries on the "
            "throttle/concurrency class."
        ),
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    inv = subparsers.add_parser(
        "invoke-canary",
        help=(
            "Invoke a Lambda with bounded retry on the throttle / "
            "reserved-concurrency-limit class."
        ),
    )
    inv.add_argument(
        "--function-name",
        required=True,
        help="Function name, name:alias, or name:version (e.g. my-fn:live).",
    )
    inv.add_argument(
        "--payload",
        required=True,
        help='JSON payload string, e.g. \'{"dry_run": true}\'.',
    )
    inv.add_argument(
        "--out",
        required=True,
        help="File to write the response payload bytes to (the caller parses it).",
    )
    inv.add_argument(
        "--region",
        default=None,
        help="AWS region (defaults to the ambient boto3/AWS_REGION config).",
    )
    inv.add_argument(
        "--max-attempts",
        type=int,
        default=_DEFAULT_MAX_ATTEMPTS,
        help=f"Max invoke attempts (default: {_DEFAULT_MAX_ATTEMPTS}).",
    )
    inv.add_argument(
        "--label",
        default="",
        help="Optional label for log/error context (defaults to the function name).",
    )

    env = subparsers.add_parser(
        "merge-lambda-env",
        help=(
            "Merge KEY=VALUE pairs into a Lambda's environment, preserving "
            "every variable already set on the live function. Values are "
            "never echoed."
        ),
    )
    env.add_argument(
        "--function-name",
        required=True,
        help="Function name (no alias — configuration is version-independent).",
    )
    env.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        required=True,
        help="Variable to merge. Repeatable. VALUE may not be empty.",
    )
    env.add_argument(
        "--region",
        default=None,
        help="AWS region (defaults to the ambient boto3/AWS_REGION config).",
    )

    args = parser.parse_args(argv)

    if args.cmd == "merge-lambda-env":
        _logging.basicConfig(
            level=_logging.WARNING, format="%(message)s", stream=sys.stderr
        )
        updates = {}
        for pair in args.set:
            if "=" not in pair:
                print(f"ERROR: --set expects KEY=VALUE, got {pair!r}", file=sys.stderr)
                return 2
            key, value = pair.split("=", 1)
            updates[key.strip()] = value
        try:
            total = merge_lambda_environment(
                args.function_name, updates, region=args.region
            )
        except LambdaEnvMergeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        # Keys, never values.
        print(
            f"merged {len(updates)} variable(s) [{', '.join(sorted(updates))}] "
            f"into {args.function_name} ({total} total; values not shown)"
        )
        return 0

    if args.cmd == "invoke-canary":
        # Surface the backoff WARNINGs to the deploy log (stderr).
        _logging.basicConfig(
            level=_logging.WARNING, format="%(message)s", stream=sys.stderr
        )
        try:
            result = invoke_lambda_with_retry(
                args.function_name,
                args.payload,
                region=args.region,
                max_attempts=args.max_attempts,
                label=args.label,
            )
        except LambdaInvokeError as exc:
            print(f"ERROR: canary invoke could not complete — {exc}", file=sys.stderr)
            return 1
        with open(args.out, "wb") as fh:
            fh.write(result.payload)
        # Metadata → stdout for the Bash caller (mirrors `aws lambda invoke`
        # stdout; predictor parses FunctionError from here).
        print(result.metadata_json())
        return 0

    return 2  # pragma: no cover — argparse requires a subcommand


if __name__ == "__main__":
    import sys

    sys.exit(main())
