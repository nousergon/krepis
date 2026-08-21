"""Tests for krepis.aws — the Lambda invoke-with-throttle-retry chokepoint."""

from __future__ import annotations

import io
import json

import pytest

from krepis.aws import (
    DEFAULT_RETRYABLE_INVOKE_CODES,
    InvokeResult,
    LambdaEnvMergeError,
    LambdaInvokeError,
    invoke_lambda_with_retry,
    main,
    merge_lambda_environment,
    remove_lambda_environment_keys,
    LambdaAliasPinnedError,
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


class FakeWaiter:
    def __init__(self, log, name):
        self.log, self.name = log, name

    def wait(self, **kw):
        self.log.append(("wait", self.name, kw.get("FunctionName")))


class FakeLambda:
    """Duck-typed boto3 lambda client with a declared starting environment."""

    def __init__(self, variables=None, read_error=None, write_error=None):
        self.variables = dict(variables or {})
        self.read_error = read_error
        self.write_error = write_error
        self.calls = []
        self.aliases: "list[dict]" = []
        self.next_version = "518"

    def get_waiter(self, name):
        return FakeWaiter(self.calls, name)

    def get_function_configuration(self, FunctionName):  # noqa: N803
        if self.read_error:
            raise self.read_error
        self.calls.append(("read", FunctionName))
        return {"Environment": {"Variables": dict(self.variables)}}

    def update_function_configuration(self, FunctionName, Environment):  # noqa: N803
        if self.write_error:
            raise self.write_error
        self.calls.append(("write", FunctionName))
        self.variables = dict(Environment["Variables"])

    # ── alias / version surface (remove_lambda_environment_keys) ──────────
    def list_aliases(self, FunctionName):  # noqa: N803
        self.calls.append(("list_aliases", FunctionName))
        return {"Aliases": list(self.aliases)}

    def publish_version(self, FunctionName):  # noqa: N803
        self.calls.append(("publish", FunctionName))
        return {"Version": self.next_version}

    def update_alias(self, FunctionName, Name, FunctionVersion):  # noqa: N803
        self.calls.append(("update_alias", Name, FunctionVersion))
        for a in self.aliases:
            if a["Name"] == Name:
                a["FunctionVersion"] = FunctionVersion


class TestMergeLambdaEnvironment:
    """alpha-engine-config-I7179: turning cost telemetry on is an environment
    edit across three repos' Lambdas, and the merge that does it may not
    delete the live-only variables it does not know about."""

    def test_existing_variables_survive(self):
        client = FakeLambda({"ANTHROPIC_API_KEY": "live-only", "RAG_DATABASE_URL": "x"})
        total = merge_lambda_environment(
            "alpha-engine-replay-concordance",
            {"KREPIS_COST_SINK_BUCKET": "alpha-engine-research"},
            client=client,
        )
        assert client.variables["ANTHROPIC_API_KEY"] == "live-only"
        assert client.variables["RAG_DATABASE_URL"] == "x"
        assert client.variables["KREPIS_COST_SINK_BUCKET"] == "alpha-engine-research"
        assert total == 3

    def test_merge_overwrites_only_the_named_keys(self):
        client = FakeLambda({"KREPIS_COST_SINK_PREFIX": "old"})
        merge_lambda_environment("f", {"KREPIS_COST_SINK_PREFIX": "new"}, client=client)
        assert client.variables == {"KREPIS_COST_SINK_PREFIX": "new"}

    def test_absent_environment_block_is_not_a_crash(self):
        class Bare(FakeLambda):
            def get_function_configuration(self, FunctionName):  # noqa: N803
                return {}

        client = Bare()
        merge_lambda_environment("f", {"A": "1"}, client=client)
        assert client.variables == {"A": "1"}

    @pytest.mark.parametrize("value", ["", None])
    def test_empty_value_is_refused(self, value):
        """An empty variable reads as unset at the consumer and silently
        disables whatever it gates — the failure this whole issue is
        about, one layer down."""
        with pytest.raises(LambdaEnvMergeError):
            merge_lambda_environment("f", {"K": value}, client=FakeLambda())

    def test_no_updates_is_refused(self):
        with pytest.raises(LambdaEnvMergeError):
            merge_lambda_environment("f", {}, client=FakeLambda())

    def test_read_failure_fails_loud_and_does_not_write(self):
        client = FakeLambda(read_error=RuntimeError("denied"))
        with pytest.raises(LambdaEnvMergeError):
            merge_lambda_environment("f", {"A": "1"}, client=client)
        assert not any(c[0] == "write" for c in client.calls)

    def test_write_failure_fails_loud(self):
        client = FakeLambda(write_error=RuntimeError("conflict"))
        with pytest.raises(LambdaEnvMergeError):
            merge_lambda_environment("f", {"A": "1"}, client=client)


class TestMergeLambdaEnvCli:
    def test_values_are_never_printed(self, capsys, monkeypatch):
        client = FakeLambda()
        monkeypatch.setattr(
            "krepis.aws.merge_lambda_environment",
            lambda fn, updates, region=None: len(updates),
        )
        rc = main([
            "merge-lambda-env",
            "--function-name", "alpha-engine-evaluator-director",
            "--set", "KREPIS_COST_SINK_BUCKET=alpha-engine-research",
        ])
        out = capsys.readouterr().out
        assert rc == 0
        assert "KREPIS_COST_SINK_BUCKET" in out
        assert "alpha-engine-research" not in out
        assert client.calls == []

    def test_malformed_pair_is_rejected(self):
        assert main([
            "merge-lambda-env", "--function-name", "f", "--set", "NOEQUALS",
        ]) == 2


class TestRemoveLambdaEnvironmentKeys:
    """alpha-engine-config-I7925: an expired /alpha-engine/GITHUB_TOKEN sat in
    the predictor Lambda's environment, was read from site-packages by a
    first-party dependency, and halted the 2026-08-21 preopen. Removing a
    credential from a live function is the removal counterpart of
    I7179's merge, and carries the same two invariants plus the L4497
    alias-pinning footgun."""

    def _unpinned(self, variables):
        client = FakeLambda(variables)
        client.aliases = [{"Name": "live", "FunctionVersion": "$LATEST"}]
        return client

    def test_only_the_named_keys_are_removed(self):
        client = self._unpinned(
            {"GITHUB_TOKEN": "dead", "ANTHROPIC_API_KEY": "live-only", "FMP_API_KEY": "x"}
        )
        remaining, published = remove_lambda_environment_keys(
            "alpha-engine-predictor-inference", ["GITHUB_TOKEN"], client=client
        )
        assert "GITHUB_TOKEN" not in client.variables
        assert client.variables["ANTHROPIC_API_KEY"] == "live-only"
        assert client.variables["FMP_API_KEY"] == "x"
        assert (remaining, published) == (2, None)

    def test_absent_key_is_refused(self):
        """A no-op reported as a removal is how a credential stays live
        somewhere nobody is looking."""
        client = self._unpinned({"FMP_API_KEY": "x"})
        with pytest.raises(LambdaEnvMergeError, match="not set"):
            remove_lambda_environment_keys("f", ["GITHUB_TOKEN"], client=client)
        assert not any(c[0] == "write" for c in client.calls)

    def test_absent_key_is_allowed_when_missing_ok(self):
        client = self._unpinned({"FMP_API_KEY": "x"})
        remaining, _ = remove_lambda_environment_keys(
            "f", ["GITHUB_TOKEN"], client=client, missing_ok=True
        )
        assert remaining == 1

    def test_no_keys_is_refused(self):
        with pytest.raises(LambdaEnvMergeError):
            remove_lambda_environment_keys("f", [], client=FakeLambda())

    def test_pinned_alias_without_promotion_raises_instead_of_no_op(self):
        """L4497: update-function-configuration mutates $LATEST only. An edit
        that cannot reach traffic is a failure, not a success."""
        client = FakeLambda({"GITHUB_TOKEN": "dead"})
        client.aliases = [{"Name": "live", "FunctionVersion": "517"}]
        with pytest.raises(LambdaAliasPinnedError, match="live"):
            remove_lambda_environment_keys("f", ["GITHUB_TOKEN"], client=client)
        assert not any(c[0] == "write" for c in client.calls)

    def test_pinned_alias_with_promotion_runs_the_full_procedure(self):
        client = FakeLambda({"GITHUB_TOKEN": "dead", "FMP_API_KEY": "x"})
        client.aliases = [{"Name": "live", "FunctionVersion": "517"}]
        remaining, published = remove_lambda_environment_keys(
            "f", ["GITHUB_TOKEN"], client=client, promote_aliases=["live"]
        )
        assert (remaining, published) == (1, "518")
        order = [c[0] for c in client.calls if c[0] in {"write", "publish", "update_alias"}]
        assert order == ["write", "publish", "update_alias"]
        assert client.aliases[0]["FunctionVersion"] == "518"

    def test_defer_publish_permits_a_latest_only_edit_on_a_pinned_function(self):
        """A deploy script publishes and moves the alias itself a few steps
        later. That is a claim about the CALLER, so it is spelled out rather
        than inferred from an empty promote list."""
        client = FakeLambda({"GITHUB_TOKEN": "dead", "FMP_API_KEY": "x"})
        client.aliases = [{"Name": "live", "FunctionVersion": "517"}]
        remaining, published = remove_lambda_environment_keys(
            "f", ["GITHUB_TOKEN"], client=client, defer_publish=True
        )
        assert (remaining, published) == (1, None)
        assert "GITHUB_TOKEN" not in client.variables
        assert not any(c[0] in {"publish", "update_alias"} for c in client.calls)
        assert client.aliases[0]["FunctionVersion"] == "517"

    def test_defer_publish_and_promote_together_are_refused(self):
        client = FakeLambda({"GITHUB_TOKEN": "dead"})
        client.aliases = [{"Name": "live", "FunctionVersion": "517"}]
        with pytest.raises(LambdaEnvMergeError, match="mutually exclusive"):
            remove_lambda_environment_keys(
                "f",
                ["GITHUB_TOKEN"],
                client=client,
                promote_aliases=["live"],
                defer_publish=True,
            )
        assert not any(c[0] == "write" for c in client.calls)

    def test_promoting_an_alias_that_does_not_exist_is_refused(self):
        client = FakeLambda({"GITHUB_TOKEN": "dead"})
        client.aliases = [{"Name": "live", "FunctionVersion": "517"}]
        with pytest.raises(LambdaEnvMergeError, match="do not exist"):
            remove_lambda_environment_keys(
                "f", ["GITHUB_TOKEN"], client=client, promote_aliases=["staging"]
            )

    def test_a_client_that_cannot_list_aliases_refuses_to_claim_success(self):
        class NoAliases(FakeLambda):
            list_aliases = None

        client = NoAliases({"GITHUB_TOKEN": "dead"})
        with pytest.raises(LambdaEnvMergeError, match="unknowable"):
            remove_lambda_environment_keys("f", ["GITHUB_TOKEN"], client=client)

    def test_read_failure_fails_loud_and_does_not_write(self):
        client = FakeLambda({"GITHUB_TOKEN": "dead"}, read_error=RuntimeError("denied"))
        with pytest.raises(LambdaEnvMergeError):
            remove_lambda_environment_keys("f", ["GITHUB_TOKEN"], client=client)
        assert not any(c[0] == "write" for c in client.calls)

    def test_write_failure_fails_loud(self):
        client = self._unpinned({"GITHUB_TOKEN": "dead"})
        client.write_error = RuntimeError("conflict")
        with pytest.raises(LambdaEnvMergeError):
            remove_lambda_environment_keys("f", ["GITHUB_TOKEN"], client=client)


class TestRemoveLambdaEnvCli:
    def test_values_are_never_printed_and_keys_are(self, capsys, monkeypatch):
        monkeypatch.setattr(
            "krepis.aws.remove_lambda_environment_keys",
            lambda fn, keys, region=None, promote_aliases=None, missing_ok=False, defer_publish=False: (
                13,
                "518" if promote_aliases else None,
            ),
        )
        rc = main([
            "remove-lambda-env",
            "--function-name", "alpha-engine-predictor-inference",
            "--unset", "GITHUB_TOKEN",
            "--promote-alias", "live",
        ])
        out = capsys.readouterr().out
        assert rc == 0
        assert "GITHUB_TOKEN" in out
        assert "518" in out
        assert "13 remain" in out

    def test_a_refusal_exits_non_zero(self, capsys, monkeypatch):
        def _boom(fn, keys, region=None, promote_aliases=None, missing_ok=False, defer_publish=False):
            raise LambdaAliasPinnedError("alias live is pinned")

        monkeypatch.setattr("krepis.aws.remove_lambda_environment_keys", _boom)
        rc = main([
            "remove-lambda-env",
            "--function-name", "f",
            "--unset", "GITHUB_TOKEN",
        ])
        assert rc == 1
        assert "pinned" in capsys.readouterr().err
