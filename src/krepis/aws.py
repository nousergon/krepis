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

        if not region:
            from krepis.aws_region import resolve_region

            region = resolve_region()
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

        if not region:
            from krepis.aws_region import resolve_region

            region = resolve_region()
        client = boto3.client("lambda", region_name=region)

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


class LambdaAliasPinnedError(LambdaEnvMergeError):
    """Raised when an environment edit would be a silent no-op on the alias
    that actually serves traffic.

    ``update-function-configuration`` mutates ``$LATEST`` only. A function
    invoked through an alias pinned to a PUBLISHED version keeps serving the
    frozen environment of that version, so the edit appears to succeed and
    changes nothing — the L4497 footgun, documented at length in
    ``crucible-predictor/config.py`` after an env flip silently did nothing.
    An edit that cannot take effect is a failure, not a success, so this is
    raised rather than logged (fleet rule: fail loud, no silent swallows).
    """


def remove_lambda_environment_keys(
    function_name: str,
    keys: "Iterable[str]",
    *,
    region: "str | None" = None,
    client: "object | None" = None,
    promote_aliases: "Iterable[str] | None" = None,
    missing_ok: bool = False,
) -> "tuple[int, str | None]":
    """Delete *keys* from a Lambda's environment. Returns ``(remaining, version)``.

    The removal counterpart of :func:`merge_lambda_environment`, and subject to
    the same two rules.

    **Read-modify-write, never replace.** Only the named keys are removed;
    every other variable on the live function — provider keys, database URLs,
    operator-set flags that are codified nowhere — survives untouched.

    **Nothing is echoed.** Values never reach stdout, a command line, or a
    shell trace. Only key NAMES and counts are returned or logged.

    Two behaviours this helper adds over a bare CLI call, both of them
    fail-loud conversions of a silent no-op:

    * A key named for removal that is **not present** raises, unless
      ``missing_ok=True``. Asking to remove something that is not there means
      the caller's model of the function is wrong, and on a credential that is
      the difference between "revoked" and "still live somewhere else".
    * If any alias is pinned to a published version, the edit to ``$LATEST``
      would not reach traffic. With ``promote_aliases`` given, the full correct
      procedure runs — ``update-function-configuration`` → ``publish-version``
      → ``update-alias`` for each named alias — and the new version is
      returned. Without it, :class:`LambdaAliasPinnedError` is raised naming
      the pinned aliases, rather than returning a success that changed nothing.

    Reverting is ``update-alias <name> --function-version <prior>``; the prior
    version is not deleted here, so the rollback target always exists.
    """
    names = [k for k in keys]
    if not names:
        raise LambdaEnvMergeError(
            "remove_lambda_environment_keys called with no keys"
        )
    for key in names:
        if not key or not isinstance(key, str):
            raise LambdaEnvMergeError(f"not a usable environment key: {key!r}")

    if client is None:
        import boto3  # imported lazily so `import krepis.aws` costs nothing

        if not region:
            from krepis.aws_region import resolve_region

            region = resolve_region()
        client = boto3.client("lambda", region_name=region)

    try:
        client.get_waiter("function_updated").wait(FunctionName=function_name)
    except Exception as exc:  # noqa: BLE001 — duck-typed boto errors
        # DELIBERATE non-raising degradation, with rationale:
        # (a) swallowed: the waiter failed (function never deployed, or the
        #     waiter is unavailable on a stubbed client).
        # (b) the primary deliverable survives because the update below is the
        #     load-bearing call and raises on its own failure — this wait only
        #     avoids a ResourceConflictException race.
        # (c) recorded here, and by the update's own exception if the race bites.
        _DEFAULT_LOGGER.debug(
            "function_updated waiter skipped for %s: %s", function_name, exc
        )

    try:
        config = client.get_function_configuration(FunctionName=function_name)
    except Exception as exc:  # noqa: BLE001
        raise LambdaEnvMergeError(
            f"could not read the current environment of {function_name}: {exc}"
        ) from exc
    variables = dict((config.get("Environment") or {}).get("Variables") or {})

    absent = sorted(k for k in names if k not in variables)
    if absent and not missing_ok:
        raise LambdaEnvMergeError(
            f"refusing to remove keys that are not set on {function_name}: "
            f"{', '.join(absent)} — the caller's model of this function is "
            f"wrong, and a no-op reported as a removal is how a credential "
            f"stays live somewhere nobody is looking"
        )
    for key in names:
        variables.pop(key, None)

    pinned = _pinned_aliases(client, function_name)
    wanted = sorted(set(promote_aliases or ()))
    if pinned and promote_aliases is None:
        raise LambdaAliasPinnedError(
            f"{function_name} serves traffic through alias(es) "
            f"{', '.join(sorted(pinned))} pinned to a published version. "
            f"update-function-configuration mutates $LATEST only, so this "
            f"edit would not reach traffic. Re-call with "
            f"promote_aliases=[...] to run the full procedure "
            f"(update-function-configuration -> publish-version -> "
            f"update-alias), or move the alias yourself."
        )
    unknown = sorted(set(wanted) - set(pinned)) if wanted else []
    if unknown:
        raise LambdaEnvMergeError(
            f"cannot promote alias(es) that do not exist on {function_name}: "
            f"{', '.join(unknown)}"
        )

    try:
        client.update_function_configuration(
            FunctionName=function_name,
            Environment={"Variables": variables},
        )
    except Exception as exc:  # noqa: BLE001
        raise LambdaEnvMergeError(
            f"could not write the reduced environment of {function_name}: {exc}"
        ) from exc
    try:
        client.get_waiter("function_updated").wait(FunctionName=function_name)
    except Exception as exc:  # noqa: BLE001 — same rationale as above
        _DEFAULT_LOGGER.debug(
            "post-update waiter skipped for %s: %s", function_name, exc
        )

    published: "str | None" = None
    if wanted:
        try:
            published = client.publish_version(FunctionName=function_name)["Version"]
        except Exception as exc:  # noqa: BLE001
            raise LambdaEnvMergeError(
                f"environment written but publish-version failed for "
                f"{function_name}: {exc} — $LATEST no longer carries "
                f"{', '.join(sorted(names))} while every alias still serves "
                f"the prior version"
            ) from exc
        for alias in wanted:
            try:
                client.update_alias(
                    FunctionName=function_name,
                    Name=alias,
                    FunctionVersion=published,
                )
            except Exception as exc:  # noqa: BLE001
                raise LambdaEnvMergeError(
                    f"published {function_name} version {published} but could "
                    f"not move alias {alias} onto it: {exc}"
                ) from exc

    return len(variables), published


def _pinned_aliases(client: "object", function_name: str) -> "list[str]":
    """Alias names on *function_name* pinned to a published version.

    An alias pointing at ``$LATEST`` is not pinned — an environment edit
    reaches it immediately and needs no publish/promote. Returns ``[]`` when
    the client cannot enumerate aliases; that is not swallowed silently, it
    raises, because guessing "no aliases" is what turns this into the silent
    no-op the whole helper exists to prevent.
    """
    lister = getattr(client, "list_aliases", None)
    if lister is None:
        raise LambdaEnvMergeError(
            "the Lambda client cannot enumerate aliases, so whether this edit "
            "reaches traffic is unknowable — refusing to report a success"
        )
    try:
        aliases = lister(FunctionName=function_name).get("Aliases") or []
    except Exception as exc:  # noqa: BLE001
        raise LambdaEnvMergeError(
            f"could not list aliases of {function_name}: {exc}"
        ) from exc
    return [
        a["Name"]
        for a in aliases
        if str(a.get("FunctionVersion", "$LATEST")) != "$LATEST"
    ]



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

    rm = subparsers.add_parser(
        "remove-lambda-env",
        help=(
            "Remove named variables from a Lambda's environment, preserving "
            "every other variable set on the live function. Values are never "
            "echoed. Refuses a key that is not set, and refuses to report "
            "success when an alias pinned to a published version would not "
            "see the change."
        ),
    )
    rm.add_argument(
        "--function-name",
        required=True,
        help="Function name (no alias — configuration is version-independent).",
    )
    rm.add_argument(
        "--unset",
        action="append",
        default=[],
        metavar="KEY",
        required=True,
        help="Variable name to remove. Repeatable.",
    )
    rm.add_argument(
        "--promote-alias",
        action="append",
        default=[],
        metavar="NAME",
        help=(
            "Alias to move onto the newly published version (e.g. live). "
            "Repeatable. Required when any alias is pinned to a published "
            "version, because $LATEST-only edits never reach that traffic."
        ),
    )
    rm.add_argument(
        "--missing-ok",
        action="store_true",
        help="Treat an already-absent key as done instead of an error.",
    )
    rm.add_argument(
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

    if args.cmd == "remove-lambda-env":
        _logging.basicConfig(
            level=_logging.WARNING, format="%(message)s", stream=sys.stderr
        )
        keys = [k.strip() for k in args.unset]
        try:
            remaining, published = remove_lambda_environment_keys(
                args.function_name,
                keys,
                region=args.region,
                promote_aliases=args.promote_alias or None,
                missing_ok=args.missing_ok,
            )
        except LambdaEnvMergeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        # Keys, never values.
        promoted = (
            f"; published version {published} and moved "
            f"{', '.join(sorted(args.promote_alias))}"
            if published
            else ""
        )
        print(
            f"removed {len(keys)} variable(s) [{', '.join(sorted(keys))}] "
            f"from {args.function_name} ({remaining} remain; values not "
            f"shown){promoted}"
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
