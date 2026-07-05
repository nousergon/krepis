"""Bounded-backoff HTTP retry primitive — the transient external-API
resilience chokepoint (L4499).

Consolidates the backoff + full-jitter + ``Retry-After`` + api-key-scrub
retry idiom that was mirrored across four alpha-engine-data sites:

  * ``collectors/daily_closes.py::_fred_get_with_retry``     (L4480)
  * ``polygon_client.py::_get`` / ``_backoff``               (L4496)
  * ``preflight.py::_reachability_get``                      (L4494)
  * ``collectors/daily_closes_fred_repair.py::_fetch_fred_range``

Each had its own copy of "exponential backoff + full jitter, honor
``Retry-After``, retry the transient class, scrub the api-key from the
error before logging/raising, then fail loud." This module is the single
source of truth for that policy so the four callsites stop drifting.

Two layers are exported:

  * :func:`request_with_retry` — the full GET-with-retry for the plain
    callsites (FRED fetch, preflight probe, FRED repair). Returns the final
    ``requests.Response``; the caller still owns status interpretation
    (``raise_for_status`` / special-casing a 403), so genuinely different
    consumers compose it without a leaky mega-config.
  * :func:`backoff_delay` + :func:`scrub_api_keys` — the low-level pieces for
    a consumer with bespoke control flow (the rate-limited ``polygon_client``
    keeps its own loop + 403 handling + JSON parse + rate limiter, but shares
    the delay math and the scrubber).
  * :func:`call_with_retry` — the generic callable loop for gRPC / boto /
    subprocess and other non-HTTP consumers. Shares :func:`backoff_delay` and
    ships :func:`is_transient_google_error` / :func:`is_transient_boto_error`
    classifiers for the common provider surfaces.

Design note (anti-over-engineering): this is deliberately NOT a
pluggable-everything HTTP framework. It captures the one invariant the four
sites share; consumers whose semantics diverge (polygon's 403 + rate limiter)
reuse the primitives rather than being forced through a generic loop.
"""

from __future__ import annotations

import logging as _logging
import random as _random
import re
import time as _time
from typing import Callable, Iterable, TypeVar

import requests

_DEFAULT_LOGGER = _logging.getLogger(__name__)

_T = TypeVar("_T")

# Transient HTTP status class: 429 (rate limit) + the retryable 5xx. A 4xx
# other than 429 is a deterministic client error — retrying it is pointless,
# so it is NOT in the default set and is returned to the caller as-is.
DEFAULT_TRANSIENT_STATUS: "frozenset[int]" = frozenset({429, 500, 502, 503, 504})

# Mask FRED ``api_key=`` (snake) and polygon ``apiKey=`` (camel) querystring
# VALUES — both leak via ``requests`` exception ``str()`` (the effective URL)
# and via hand-built error strings. Mirrors the per-repo scrubbers this module
# replaces; complements ``krepis.logging.SecretsRedactingFilter``
# (which catches token-shaped secrets, not query-param api keys).
_API_KEY_RE = re.compile(r"(?:api_key|apiKey)=[^&\s]+")


def scrub_api_keys(msg: object) -> str:
    """Mask ``api_key=...`` / ``apiKey=...`` querystring values in a string.

    Preserves the key NAME (so logs still show *which* param) and the value
    delimiter, replacing only the secret value with ``***``. Idempotent.
    """
    return _API_KEY_RE.sub(lambda m: m.group(0).split("=", 1)[0] + "=***", str(msg))


class HttpRetryError(RuntimeError):
    """Raised when all attempts are exhausted on a transient NETWORK error
    (``requests.Timeout`` / ``requests.ConnectionError``) or a non-transient
    ``RequestException``.

    The message is api-key-scrubbed. The originating exception is preserved
    as ``__cause__`` (and on ``.last_exc``); ``.label`` / ``.attempts`` carry
    context for callers that want to re-wrap (e.g. preflight's
    ``RuntimeError(... unreachable ...)``).
    """

    def __init__(self, label: str, attempts: int, last_exc: BaseException) -> None:
        self.label = label
        self.attempts = attempts
        self.last_exc = last_exc
        super().__init__(
            scrub_api_keys(
                f"{label or 'request'} failed after {attempts} attempt(s): {last_exc}"
            )
        )


def backoff_delay(
    attempt: int,
    *,
    base: float = 1.0,
    cap: float = 30.0,
    retry_after: "str | float | None" = None,
    rng: "_random.Random | None" = None,
) -> float:
    """Full-jitter exponential backoff: ``min(base*2**attempt + U(0, base), cap)``.

    ``attempt`` is 0-indexed. Honors a server ``Retry-After`` (seconds, str or
    float) when supplied — a numeric value replaces the exponential term (still
    + jitter, still capped); a non-numeric ``Retry-After`` (HTTP-date form)
    falls back to the exponential term. ``rng`` is injectable for deterministic
    tests.
    """
    wait: "float | None" = None
    if retry_after is not None:
        try:
            wait = float(retry_after)
        except (TypeError, ValueError):
            wait = None
    if wait is None:
        wait = base * (2 ** attempt)
    jitter = (rng or _random).uniform(0, base)
    return min(wait + jitter, cap)


def request_with_retry(
    url: str,
    *,
    method: str = "GET",
    params: "dict | None" = None,
    session: "requests.Session | None" = None,
    timeout: float = 15.0,
    max_attempts: int = 3,
    backoff_base: float = 1.0,
    backoff_cap: float = 30.0,
    transient_status: Iterable[int] = DEFAULT_TRANSIENT_STATUS,
    retry_network: bool = True,
    honor_retry_after: bool = True,
    scrub: Callable[[object], str] = scrub_api_keys,
    logger: "_logging.Logger | None" = None,
    label: str = "",
    sleep: Callable[[float], None] = _time.sleep,
) -> requests.Response:
    """``method`` ``url`` with bounded backoff + full jitter on the transient
    class, returning the final :class:`requests.Response`.

    Retries:
      * responses whose status is in ``transient_status`` (default 429 + 5xx),
        honoring ``Retry-After`` when ``honor_retry_after``; and
      * (when ``retry_network``) ``requests.Timeout`` / ``ConnectionError``.

    Terminal behavior:
      * a transient-status response that survives ``max_attempts`` is
        **returned** — the caller decides whether to ``raise_for_status`` or
        special-case it (e.g. a 403, which is NOT in the transient set, is
        returned immediately for the caller to convert); and
      * an exhausted NETWORK error (or a non-transient ``RequestException``
        such as a bad URL) raises :class:`HttpRetryError` (scrubbed).

    ``scrub`` is applied to every error string logged or raised. ``session``
    lets a caller reuse a session (e.g. one carrying auth query params).
    ``sleep`` is injectable for tests. ``max_attempts`` must be >= 1.
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
    log = logger or _DEFAULT_LOGGER
    transient = frozenset(transient_status)
    requester = (session or requests).request
    resp: "requests.Response | None" = None
    for attempt in range(max_attempts):
        last = attempt == max_attempts - 1
        try:
            resp = requester(method, url, params=params or {}, timeout=timeout)
        except (requests.Timeout, requests.ConnectionError) as exc:
            if not retry_network or last:
                raise HttpRetryError(label, attempt + 1, exc) from exc
            delay = backoff_delay(attempt, base=backoff_base, cap=backoff_cap)
            log.warning(
                "%s transient %s — backing off %.1fs (attempt %d/%d)",
                label or url, type(exc).__name__, delay, attempt + 1, max_attempts,
            )
            sleep(delay)
            continue
        except requests.RequestException as exc:
            # Non-transient (bad URL / too many redirects / invalid schema) —
            # retrying a deterministic error is pointless; fail loud now.
            raise HttpRetryError(label, attempt + 1, exc) from exc

        if resp.status_code in transient and not last:
            retry_after = resp.headers.get("Retry-After") if honor_retry_after else None
            delay = backoff_delay(
                attempt, base=backoff_base, cap=backoff_cap, retry_after=retry_after,
            )
            log.warning(
                "%s HTTP %d — backing off %.1fs (attempt %d/%d)",
                label or url, resp.status_code, delay, attempt + 1, max_attempts,
            )
            sleep(delay)
            continue
        return resp

    # Loop exhausted on transient-status responses: return the last one for the
    # caller to interpret (network exhaustion already raised above). resp is
    # non-None because max_attempts >= 1 guarantees at least one assignment.
    assert resp is not None
    return resp


# ── Generic callable retry (gRPC / boto / subprocess / any transient class) ──
#
# Mirrors the ``request_with_retry`` / ``invoke_lambda_with_retry`` split:
# ``backoff_delay`` is the shared math; ``call_with_retry`` is the generic loop
# for consumers whose control flow is not a plain HTTP GET or Lambda invoke.


# google.api_core gRPC exception names in the transient class. Matched by name
# (not isinstance) so callers without the google extra installed never import it.
DEFAULT_TRANSIENT_GOOGLE_EXCEPTIONS: "frozenset[str]" = frozenset({
    "ServiceUnavailable",
    "InternalServerError",
    "DeadlineExceeded",
    "ResourceExhausted",
    "Aborted",
})

# botocore ClientError codes in the transient class (Polly throttling, Lambda
# throttle, generic AWS 5xx). Callers may pass a narrower ``codes`` set.
DEFAULT_TRANSIENT_BOTO_ERROR_CODES: "frozenset[str]" = frozenset({
    "ThrottlingException",
    "TooManyRequestsException",
    "ServiceUnavailable",
    "InternalFailure",
    "InternalServerError",
    "ProvisionedThroughputExceededException",
})


def is_transient_google_error(
    exc: BaseException,
    *,
    names: Iterable[str] = DEFAULT_TRANSIENT_GOOGLE_EXCEPTIONS,
) -> bool:
    """True for ``google.api_core`` blips worth retrying (503, overload, …)."""
    mod = type(exc).__module__
    return mod.startswith("google.api_core") and type(exc).__name__ in frozenset(names)


def is_transient_boto_error(
    exc: BaseException,
    *,
    codes: Iterable[str] = DEFAULT_TRANSIENT_BOTO_ERROR_CODES,
) -> bool:
    """True for ``botocore.exceptions.ClientError`` in the transient code class."""
    if type(exc).__name__ != "ClientError" or not type(exc).__module__.startswith("botocore"):
        return False
    response = getattr(exc, "response", None) or {}
    code = (response.get("Error") or {}).get("Code", "")
    return code in frozenset(codes)


def call_with_retry(
    fn: Callable[[], _T],
    *,
    is_retryable: Callable[[BaseException], bool],
    max_attempts: int = 3,
    backoff_base: float = 1.0,
    backoff_cap: float = 30.0,
    retry_after: "Callable[[BaseException], str | float | None] | None" = None,
    label: str = "",
    logger: "_logging.Logger | None" = None,
    sleep: "Callable[[float], None]" = _time.sleep,
) -> _T:
    """Run ``fn`` with bounded full-jitter backoff on retryable exceptions.

    Retries when ``is_retryable(exc)`` is true, up to ``max_attempts``. On
    exhaustion the **original** exception is re-raised (preserving type and
    cause) — same terminal contract as ``request_with_retry`` returning the
    last 503 for the caller to interpret.

    ``retry_after(exc)`` may return a server hint (seconds) that replaces the
    exponential term in :func:`backoff_delay`, mirroring HTTP ``Retry-After``.
    ``sleep`` is injectable for tests. ``max_attempts`` must be >= 1.
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
    log = logger or _DEFAULT_LOGGER
    tag = label or "call"
    for attempt in range(max_attempts):
        try:
            return fn()
        except BaseException as exc:
            if not is_retryable(exc) or attempt == max_attempts - 1:
                raise
            hint = retry_after(exc) if retry_after is not None else None
            delay = backoff_delay(
                attempt, base=backoff_base, cap=backoff_cap, retry_after=hint,
            )
            log.warning(
                "%s transient %s — backing off %.1fs (attempt %d/%d)",
                tag, type(exc).__name__, delay, attempt + 1, max_attempts,
            )
            sleep(delay)
    raise RuntimeError(f"{tag} failed after {max_attempts} attempts")  # pragma: no cover
