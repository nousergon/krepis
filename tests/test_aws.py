"""Tests for krepis.aws — the Lambda invoke-with-throttle-retry chokepoint."""

from __future__ import annotations

import io
import json

import pytest

from krepis.aws import (
    DEFAULT_RETRYABLE_INVOKE_CODES,
    InvokeResult,
    LambdaInvokeError,
    invoke_lambda_with_retry,
)

_NOSLEEP = lambda _d: None  # noqa: E731


def _client_error(code: str, message: str = "boom"):
    from botocore.exceptions import ClientError

    return ClientError(
        error_response={"Error": {"Code": code, "Message": message}},
        operation_name="Invoke",
    )


class _FakeLambda:
    """A boto3 lambda client stand-in driven by a scripted sequence of
    behaviors: each element is either a ClientError to raise or a dict to
    return from ``invoke``. Records every call for assertions."""

    def __init__(self, script):
        self._script = list(script)
        self.calls = []

    def invoke(self, **kwargs):
        self.calls.append(kwargs)
        behavior = self._script.pop(0)
        if isinstance(behavior, Exception):
            raise behavior
        return behavior


def _ok_response(status=200, payload=b'{"status": "OK"}', function_error=None):
    resp = {"StatusCode": status, "Payload": io.BytesIO(payload)}
    if function_error is not None:
        resp["FunctionError"] = function_error
    resp["ExecutedVersion"] = "42"
    return resp


def test_success_first_try_returns_metadata_and_payload():
    client = _FakeLambda([_ok_response()])
    result = invoke_lambda_with_retry(
        "fn:live", '{"dry_run": true}', client=client, sleep=_NOSLEEP
    )
    assert isinstance(result, InvokeResult)
    assert result.status_code == 200
    assert result.function_error is None
    assert result.executed_version == "42"
    assert json.loads(result.payload)["status"] == "OK"
    assert len(client.calls) == 1
    # Payload is passed as bytes.
    assert client.calls[0]["Payload"] == b'{"dry_run": true}'


def test_metadata_json_shape_matches_aws_cli_stdout():
    result = InvokeResult(200, None, "42", b"{}")
    meta = json.loads(result.metadata_json())
    assert meta == {"StatusCode": 200, "FunctionError": "", "ExecutedVersion": "42"}


def test_function_error_is_surfaced_not_raised():
    # An in-function unhandled exception sets FunctionError on the metadata but
    # the invoke API call SUCCEEDED — we return it for the caller to judge, we
    # do NOT raise (that's a bad-status, not a failed invoke).
    client = _FakeLambda(
        [_ok_response(payload=b'{"errorMessage": "boom"}', function_error="Unhandled")]
    )
    result = invoke_lambda_with_retry("fn:live", "{}", client=client, sleep=_NOSLEEP)
    assert result.function_error == "Unhandled"
    assert len(client.calls) == 1


def test_throttle_then_success_retries():
    client = _FakeLambda(
        [
            _client_error("TooManyRequestsException", "Rate Exceeded"),
            _client_error("TooManyRequestsException", "Rate Exceeded"),
            _ok_response(),
        ]
    )
    delays = []
    result = invoke_lambda_with_retry(
        "fn:live", "{}", client=client, sleep=delays.append, max_attempts=6
    )
    assert result.status_code == 200
    assert len(client.calls) == 3
    assert len(delays) == 2  # slept before each retry, not after success
    assert all(d > 0 for d in delays)


def test_non_throttle_error_fails_loud_immediately_no_retry():
    client = _FakeLambda([_client_error("AccessDeniedException", "nope")])
    with pytest.raises(LambdaInvokeError) as ei:
        invoke_lambda_with_retry("fn:live", "{}", client=client, sleep=_NOSLEEP)
    assert ei.value.code == "AccessDeniedException"
    assert ei.value.attempts == 1
    assert len(client.calls) == 1  # NOT retried


def test_persistent_throttle_exhausts_and_fails_loud():
    client = _FakeLambda(
        [_client_error("TooManyRequestsException", "Rate Exceeded")] * 6
    )
    with pytest.raises(LambdaInvokeError) as ei:
        invoke_lambda_with_retry(
            "fn:live", "{}", client=client, sleep=_NOSLEEP, max_attempts=6
        )
    assert ei.value.code == "TooManyRequestsException"
    assert ei.value.attempts == 6
    assert len(client.calls) == 6  # exactly max_attempts


def test_max_attempts_must_be_positive():
    client = _FakeLambda([_ok_response()])
    with pytest.raises(ValueError):
        invoke_lambda_with_retry("fn:live", "{}", client=client, max_attempts=0)


def test_reserved_concurrency_code_is_default_retryable():
    assert "TooManyRequestsException" in DEFAULT_RETRYABLE_INVOKE_CODES


def test_cli_invoke_canary_writes_payload_and_prints_metadata(tmp_path, monkeypatch, capsys):
    from krepis import aws

    out_file = tmp_path / "canary.json"
    fake = _FakeLambda([_ok_response(payload=b'{"status": "SKIPPED"}')])
    monkeypatch.setattr(
        aws, "invoke_lambda_with_retry", _passthrough_using(fake)
    )
    rc = aws.main(
        [
            "invoke-canary",
            "--function-name",
            "fn:live",
            "--payload",
            '{"dry_run": true}',
            "--out",
            str(out_file),
        ]
    )
    assert rc == 0
    assert json.loads(out_file.read_bytes())["status"] == "SKIPPED"
    meta = json.loads(capsys.readouterr().out.strip())
    assert meta["StatusCode"] == 200
    assert meta["FunctionError"] == ""


def test_cli_invoke_canary_returns_1_on_uninvokable(tmp_path, monkeypatch):
    from krepis import aws

    def _raise(*_a, **_k):
        raise LambdaInvokeError("fn:live", 6, "TooManyRequestsException", "Rate Exceeded")

    monkeypatch.setattr(aws, "invoke_lambda_with_retry", _raise)
    rc = aws.main(
        [
            "invoke-canary",
            "--function-name",
            "fn:live",
            "--payload",
            "{}",
            "--out",
            str(tmp_path / "x.json"),
        ]
    )
    assert rc == 1


def _passthrough_using(fake_client):
    """Return an invoke_lambda_with_retry that forces the fake client (so the
    CLI test exercises real invoke logic without boto3/AWS)."""

    def _inner(function_name, payload, **kwargs):
        kwargs.pop("client", None)
        kwargs.pop("region", None)
        return invoke_lambda_with_retry(
            function_name, payload, client=fake_client, sleep=_NOSLEEP,
            **{k: v for k, v in kwargs.items() if k in {"max_attempts", "label"}},
        )

    return _inner
