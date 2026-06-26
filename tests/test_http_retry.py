"""Tests for ``krepis.http_retry`` — the consolidated transient
external-API retry primitive (L4499)."""

from __future__ import annotations

import random
from unittest.mock import MagicMock, patch

import pytest
import requests

from krepis import http_retry
from krepis.http_retry import (
    HttpRetryError,
    backoff_delay,
    request_with_retry,
    scrub_api_keys,
)


# ── scrub_api_keys ───────────────────────────────────────────────────────────


def test_scrub_masks_both_styles():
    assert scrub_api_keys("x?api_key=SECRET&y=1") == "x?api_key=***&y=1"
    assert scrub_api_keys("x?apiKey=SECRET&y=1") == "x?apiKey=***&y=1"


def test_scrub_terminates_at_ampersand():
    assert scrub_api_keys("u?apiKey=SECRET&file_type=json") == "u?apiKey=***&file_type=json"


def test_scrub_passthrough_and_idempotent():
    assert scrub_api_keys("no key here") == "no key here"
    once = scrub_api_keys("?api_key=SECRET")
    assert scrub_api_keys(once) == once == "?api_key=***"


def test_scrub_accepts_exception_object():
    exc = requests.HTTPError("500 for url: https://x/?apiKey=LEAKED&a=1")
    out = scrub_api_keys(exc)
    assert "LEAKED" not in out and "apiKey=***" in out


# ── backoff_delay ────────────────────────────────────────────────────────────


def test_backoff_grows_exponentially_and_caps():
    rng = MagicMock()
    rng.uniform.return_value = 0.0  # zero jitter → deterministic
    assert backoff_delay(0, base=1.0, cap=30.0, rng=rng) == 1.0
    assert backoff_delay(1, base=1.0, cap=30.0, rng=rng) == 2.0
    assert backoff_delay(2, base=1.0, cap=30.0, rng=rng) == 4.0
    assert backoff_delay(10, base=1.0, cap=30.0, rng=rng) == 30.0  # capped


def test_backoff_jitter_is_bounded():
    for attempt in range(4):
        d = backoff_delay(attempt, base=1.0, cap=100.0, rng=random.Random(attempt))
        base_term = 1.0 * (2 ** attempt)
        assert base_term <= d <= base_term + 1.0


def test_backoff_honors_numeric_retry_after():
    rng = MagicMock(); rng.uniform.return_value = 0.0
    # Retry-After replaces the exponential term.
    assert backoff_delay(3, base=1.0, cap=100.0, retry_after="12", rng=rng) == 12.0
    assert backoff_delay(3, base=1.0, cap=100.0, retry_after=7.5, rng=rng) == 7.5


def test_backoff_non_numeric_retry_after_falls_back():
    rng = MagicMock(); rng.uniform.return_value = 0.0
    # HTTP-date form → not parseable → exponential term (2**1 = 2).
    assert backoff_delay(1, base=1.0, cap=100.0, retry_after="Wed, 21 Oct 2026 07:28:00 GMT", rng=rng) == 2.0


# ── request_with_retry ───────────────────────────────────────────────────────


def _resp(status: int, headers: "dict | None" = None) -> MagicMock:
    r = MagicMock(spec=requests.Response)
    r.status_code = status
    r.headers = headers or {}
    return r


def _session(*side_effect):
    s = MagicMock()
    s.request.side_effect = list(side_effect)
    return s


_NOSLEEP = lambda *_a, **_k: None  # noqa: E731


def test_success_first_try_returns_response():
    s = _session(_resp(200))
    out = request_with_retry("https://x", session=s, sleep=_NOSLEEP, label="x")
    assert out.status_code == 200
    assert s.request.call_count == 1


def test_5xx_retries_then_succeeds():
    s = _session(_resp(500), _resp(200))
    out = request_with_retry("https://x", session=s, max_attempts=3, sleep=_NOSLEEP)
    assert out.status_code == 200
    assert s.request.call_count == 2


def test_5xx_exhausted_returns_last_response():
    # A persistent transient status is RETURNED (caller does raise_for_status).
    s = _session(_resp(503), _resp(503), _resp(503))
    out = request_with_retry("https://x", session=s, max_attempts=3, sleep=_NOSLEEP)
    assert out.status_code == 503
    assert s.request.call_count == 3


def test_429_honors_retry_after_header():
    delays = []
    s = _session(_resp(429, {"Retry-After": "5"}), _resp(200))
    request_with_retry(
        "https://x", session=s, max_attempts=3,
        sleep=lambda d: delays.append(d),
    )
    # The single backoff used Retry-After=5 (+ jitter in [0,1)).
    assert len(delays) == 1 and 5.0 <= delays[0] < 6.0


def test_non_transient_status_returned_immediately():
    # 403 is not in the transient set → returned at once for the caller (e.g.
    # polygon's PolygonForbiddenError conversion), no retry.
    s = _session(_resp(403))
    out = request_with_retry("https://x", session=s, max_attempts=3, sleep=_NOSLEEP)
    assert out.status_code == 403
    assert s.request.call_count == 1


def test_network_error_retries_then_succeeds():
    s = _session(requests.ConnectionError("boom"), _resp(200))
    out = request_with_retry("https://x", session=s, max_attempts=3, sleep=_NOSLEEP)
    assert out.status_code == 200
    assert s.request.call_count == 2


def test_network_error_exhausted_raises_scrubbed():
    s = _session(requests.Timeout("read timed out ?apiKey=LEAKED"),
                 requests.Timeout("read timed out ?apiKey=LEAKED"))
    with pytest.raises(HttpRetryError) as ei:
        request_with_retry("https://x", session=s, max_attempts=2, sleep=_NOSLEEP, label="polygon")
    assert "LEAKED" not in str(ei.value)
    assert ei.value.attempts == 2
    assert isinstance(ei.value.last_exc, requests.Timeout)
    assert "polygon" in str(ei.value)


def test_retry_network_false_raises_on_first_network_error():
    s = _session(requests.ConnectionError("boom"))
    with pytest.raises(HttpRetryError):
        request_with_retry("https://x", session=s, retry_network=False, sleep=_NOSLEEP)
    assert s.request.call_count == 1


def test_non_transient_request_exception_raises_immediately():
    s = _session(requests.TooManyRedirects("loop"))
    with pytest.raises(HttpRetryError):
        request_with_retry("https://x", session=s, max_attempts=3, sleep=_NOSLEEP)
    assert s.request.call_count == 1  # no retry on a deterministic error


def test_max_attempts_must_be_positive():
    with pytest.raises(ValueError, match="max_attempts must be >= 1"):
        request_with_retry("https://x", session=_session(_resp(200)), max_attempts=0)


def test_default_session_uses_requests_module():
    # session=None path uses the module-level requests.request.
    with patch.object(http_retry.requests, "request", return_value=_resp(200)) as m:
        out = request_with_retry("https://x", sleep=_NOSLEEP)
    assert out.status_code == 200
    m.assert_called_once()
    # method + url threaded positionally; params/timeout as kwargs.
    args, kwargs = m.call_args
    assert args[0] == "GET" and args[1] == "https://x"
    assert "timeout" in kwargs


def test_custom_transient_status_set():
    # Caller can widen/narrow the retry class; 418 retried here, 500 not.
    s = _session(_resp(418), _resp(200))
    out = request_with_retry(
        "https://x", session=s, transient_status={418}, max_attempts=3, sleep=_NOSLEEP,
    )
    assert out.status_code == 200 and s.request.call_count == 2

    s2 = _session(_resp(500))
    out2 = request_with_retry(
        "https://x", session=s2, transient_status={418}, max_attempts=3, sleep=_NOSLEEP,
    )
    assert out2.status_code == 500 and s2.request.call_count == 1  # 500 not transient here
