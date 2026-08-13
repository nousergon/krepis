"""Tests for :mod:`krepis.cost_sink`.

These lock the properties that make cost telemetry survivable: it batches
rather than PUTting per call, it partitions by when the CALL happened, a
second flush cannot clobber the first, and a broken sink never takes down
the work it was measuring.
"""

from __future__ import annotations

import json

import pytest

from krepis.cost_sink import (
    BUCKET_ENV_VAR,
    DEFAULT_FLUSH_THRESHOLD,
    PREFIX_ENV_VAR,
    CostSinkConfigError,
    S3JsonlCostSink,
    default_sink_from_env,
    reset_default_sink_for_tests,
    resolve_run_id,
)


class FakeS3:
    def __init__(self, fail=False):
        self.puts = []
        self.fail = fail

    def put_object(self, **kwargs):
        if self.fail:
            raise RuntimeError("s3 down")
        self.puts.append(kwargs)


def _record(callsite_id="my-callsite", ts="2026-07-28T09:15:00+00:00", cost=0.01):
    return {
        "ts": ts,
        "callsite_id": callsite_id,
        "provider": "openrouter",
        "model": "deepseek-v4-flash",
        "input_tokens": 100,
        "output_tokens": 20,
        "cache_read_tokens": 0,
        "cost_usd": cost,
    }


def _sink(s3, **kw):
    kw.setdefault("bucket", "b")
    kw.setdefault("prefix", "cost_raw")
    kw.setdefault("run_id", "run-1")
    kw.setdefault("register_atexit", False)
    return S3JsonlCostSink(s3_client=s3, **kw)


class TestBatching:
    def test_no_put_until_flush(self):
        """Buffering is the point — an agentic loop must not issue one PUT
        per call."""
        s3 = FakeS3()
        sink = _sink(s3)
        for _ in range(10):
            sink(_record())
        assert s3.puts == [], "records must not PUT individually"
        assert sink.flush() == 1
        assert len(s3.puts) == 1, "ten records, one object"

    def test_one_object_per_group_not_per_record(self):
        s3 = FakeS3()
        sink = _sink(s3)
        sink(_record(callsite_id="a"))
        sink(_record(callsite_id="a"))
        sink(_record(callsite_id="b"))
        sink.flush()
        assert len(s3.puts) == 2, "grouped by callsite_id"

    def test_body_is_one_json_object_per_line(self):
        s3 = FakeS3()
        sink = _sink(s3)
        sink(_record(cost=0.01))
        sink(_record(cost=0.02))
        sink.flush()
        lines = s3.puts[0]["Body"].decode().splitlines()
        assert len(lines) == 2
        assert [json.loads(x)["cost_usd"] for x in lines] == [0.01, 0.02]

    def test_auto_flush_at_threshold_bounds_loss(self):
        """Buffering everything until close would turn any hard kill into
        total loss for the run."""
        s3 = FakeS3()
        sink = _sink(s3, flush_threshold=3)
        for _ in range(3):
            sink(_record())
        assert len(s3.puts) == 1, "auto-flushed at the threshold"

    def test_default_threshold_keeps_a_normal_loop_to_one_put(self):
        s3 = FakeS3()
        sink = _sink(s3)
        for _ in range(DEFAULT_FLUSH_THRESHOLD - 1):
            sink(_record())
        assert s3.puts == []


class TestKeyLayout:
    def test_key_uses_record_date_not_flush_time(self):
        """A record belongs in the partition describing when the CALL
        happened."""
        s3 = FakeS3()
        sink = _sink(s3)
        sink(_record(ts="2026-07-19T23:59:00+00:00"))
        sink.flush()
        assert "/2026-07-19/" in s3.puts[0]["Key"]

    def test_key_shape(self):
        s3 = FakeS3()
        sink = _sink(s3, run_id="run-xyz")
        sink(_record(callsite_id="evaljudge-sync"))
        sink.flush()
        assert s3.puts[0]["Key"] == (
            "cost_raw/2026-07-28/run-xyz/evaljudge-sync.0.jsonl"
        )
        assert s3.puts[0]["ContentType"] == "application/x-ndjson"

    def test_second_flush_does_not_overwrite_the_first(self):
        """Sequence numbers, not a fixed key — otherwise a mid-run flush is
        silently erased by the one at close."""
        s3 = FakeS3()
        sink = _sink(s3)
        sink(_record())
        sink.flush()
        sink(_record())
        sink.flush()
        keys = [p["Key"] for p in s3.puts]
        assert len(set(keys)) == 2, f"keys collided: {keys}"
        assert keys[0].endswith(".0.jsonl") and keys[1].endswith(".1.jsonl")

    def test_missing_ts_does_not_silently_become_today(self):
        """Substituting today's date files the record under a partition it
        does not belong to — a plausible wrong answer is worse than an
        obviously wrong one."""
        s3 = FakeS3()
        sink = _sink(s3)
        sink({"callsite_id": "x", "cost_usd": 0.1})
        sink.flush()
        assert "/unknown-date/" in s3.puts[0]["Key"]

    def test_missing_callsite_id_is_labelled_not_dropped(self):
        s3 = FakeS3()
        sink = _sink(s3)
        sink({"ts": "2026-07-28T00:00:00+00:00", "cost_usd": 0.1})
        sink.flush()
        assert s3.puts[0]["Key"].endswith("unknown.0.jsonl")


class TestFailureHandling:
    def test_put_failure_does_not_raise(self):
        """A telemetry fault must never take down the work it measures."""
        s3 = FakeS3(fail=True)
        sink = _sink(s3)
        sink(_record())
        sink.flush()  # must not raise
        assert sink.flush_errors == 1
        assert sink.records_written == 0

    def test_failure_is_counted_per_failed_group(self):
        s3 = FakeS3(fail=True)
        sink = _sink(s3)
        sink(_record(callsite_id="a"))
        sink(_record(callsite_id="b"))
        sink.flush()
        assert sink.flush_errors == 2

    def test_records_written_tracks_success(self):
        s3 = FakeS3()
        sink = _sink(s3)
        sink(_record())
        sink(_record())
        sink.flush()
        assert sink.records_written == 2
        assert sink.flush_errors == 0


class TestLifecycle:
    def test_context_manager_flushes_on_exit(self):
        s3 = FakeS3()
        with _sink(s3) as sink:
            sink(_record())
            assert s3.puts == []
        assert len(s3.puts) == 1

    def test_flush_is_idempotent_when_empty(self):
        s3 = FakeS3()
        sink = _sink(s3)
        assert sink.flush() == 0
        assert sink.flush() == 0
        assert s3.puts == []

    @pytest.mark.parametrize("bad", ["", None])
    def test_bucket_and_prefix_must_be_present(self, bad):
        with pytest.raises(ValueError):
            S3JsonlCostSink(bucket=bad, prefix="p", register_atexit=False)
        with pytest.raises(ValueError):
            S3JsonlCostSink(bucket="b", prefix=bad, register_atexit=False)


class TestRunId:
    def test_env_var_wins(self, monkeypatch):
        """An orchestrator's run id is the join key — inventing a second one
        destroys the join."""
        monkeypatch.setenv("KREPIS_RUN_ID", "sf-exec-abc")
        assert resolve_run_id() == "sf-exec-abc"

    def test_falls_back_when_unset(self, monkeypatch):
        monkeypatch.delenv("KREPIS_RUN_ID", raising=False)
        assert resolve_run_id().startswith("krepis-")

    def test_blank_env_var_is_not_used(self, monkeypatch):
        monkeypatch.setenv("KREPIS_RUN_ID", "   ")
        assert resolve_run_id().startswith("krepis-")

    def test_no_format_is_imposed_on_run_id(self):
        """alpha-engine-config-I5206: an aggregator requiring ISO-date run
        ids discarded 100% of production rows for 17 days when a producer's
        format changed. Run ids are opaque here, deliberately."""
        s3 = FakeS3()
        sink = _sink(s3, run_id="276a5be44c7c-EXEL-v5")
        sink(_record())
        sink.flush()
        assert "276a5be44c7c-EXEL-v5" in s3.puts[0]["Key"]


class TestDefaultSinkFromEnv:
    """The default sink is what makes emission a property of the ENVIRONMENT
    rather than of whoever wrote the call site.

    alpha-engine-config-I7179: with the sink an opt-in constructor argument,
    coverage equalled the set of authors who remembered to pass one — which
    on 2026-08-13 was a single process, while every LLM-calling stage of the
    weekly pipeline emitted nothing.
    """

    @pytest.fixture(autouse=True)
    def _clean(self, monkeypatch):
        monkeypatch.delenv(BUCKET_ENV_VAR, raising=False)
        monkeypatch.delenv(PREFIX_ENV_VAR, raising=False)
        monkeypatch.delenv("KREPIS_RUN_ID", raising=False)
        reset_default_sink_for_tests()
        yield
        reset_default_sink_for_tests()

    def test_unconfigured_returns_none(self):
        """A public consumer that never asked for cost telemetry pays
        nothing."""
        assert default_sink_from_env() is None

    def test_both_set_builds_a_sink_writing_where_told(self, monkeypatch):
        monkeypatch.setenv(BUCKET_ENV_VAR, "alpha-engine-research")
        monkeypatch.setenv(PREFIX_ENV_VAR, "decision_artifacts/_cost_raw")
        monkeypatch.setenv("KREPIS_RUN_ID", "sf-exec-1")
        s3 = FakeS3()
        sink = default_sink_from_env(s3_client=s3)
        assert isinstance(sink, S3JsonlCostSink)
        sink(_record(callsite_id="replay-concordance"))
        sink.flush()
        assert s3.puts[0]["Bucket"] == "alpha-engine-research"
        assert s3.puts[0]["Key"] == (
            "decision_artifacts/_cost_raw/2026-07-28/sf-exec-1/"
            "replay-concordance.0.jsonl"
        )

    def test_one_sink_per_process(self, monkeypatch):
        """A lane building a fresh client per request must not build a fresh
        buffer per request — that is one PUT per call, which is the shape
        S3JsonlCostSink exists to avoid."""
        monkeypatch.setenv(BUCKET_ENV_VAR, "b")
        monkeypatch.setenv(PREFIX_ENV_VAR, "p")
        first = default_sink_from_env(s3_client=FakeS3())
        second = default_sink_from_env(s3_client=FakeS3())
        assert first is second

    @pytest.mark.parametrize(
        "set_var,missing_var",
        [(BUCKET_ENV_VAR, PREFIX_ENV_VAR), (PREFIX_ENV_VAR, BUCKET_ENV_VAR)],
    )
    def test_half_configured_raises(self, monkeypatch, set_var, missing_var):
        """Falling back to "no sink" here is how a deploy-time typo becomes
        months of unattributed spend: the destination prefix simply keeps
        not growing, which reads exactly like a quiet week."""
        monkeypatch.setenv(set_var, "x")
        with pytest.raises(CostSinkConfigError) as exc:
            default_sink_from_env()
        assert missing_var in str(exc.value)

    def test_blank_values_are_treated_as_unset(self, monkeypatch):
        monkeypatch.setenv(BUCKET_ENV_VAR, "   ")
        monkeypatch.setenv(PREFIX_ENV_VAR, "   ")
        assert default_sink_from_env() is None


class TestLLMClientDefaultsToTheEnvironmentSink:
    """The wiring, asserted at the client rather than at the factory — a
    default nothing consults is not a default."""

    @pytest.fixture(autouse=True)
    def _clean(self, monkeypatch):
        monkeypatch.delenv(BUCKET_ENV_VAR, raising=False)
        monkeypatch.delenv(PREFIX_ENV_VAR, raising=False)
        reset_default_sink_for_tests()
        yield
        reset_default_sink_for_tests()

    def _spec(self):
        from krepis.llm_config import ModelSpec

        return ModelSpec(provider="openrouter", model="deepseek-v4-flash")

    def test_client_with_no_sink_argument_emits_when_env_is_set(self, monkeypatch):
        monkeypatch.setenv(BUCKET_ENV_VAR, "b")
        monkeypatch.setenv(PREFIX_ENV_VAR, "p")
        from krepis.llm import LLMClient

        client = LLMClient(self._spec(), callsite_id="replay-concordance")
        assert client._cost_sink is not None

    def test_client_with_no_sink_argument_stays_silent_when_env_is_absent(self):
        from krepis.llm import LLMClient

        client = LLMClient(self._spec(), callsite_id="replay-concordance")
        assert client._cost_sink is None

    def test_explicit_sink_wins_over_the_environment(self, monkeypatch):
        monkeypatch.setenv(BUCKET_ENV_VAR, "b")
        monkeypatch.setenv(PREFIX_ENV_VAR, "p")
        from krepis.llm import LLMClient

        injected: list = []

        def sink(record):
            injected.append(record)

        client = LLMClient(self._spec(), callsite_id="c", cost_sink=sink)
        assert client._cost_sink is sink

    def test_half_configured_environment_fails_at_construction(self, monkeypatch):
        """Before the first billable call, not after seventeen days of
        them."""
        monkeypatch.setenv(BUCKET_ENV_VAR, "b")
        from krepis.llm import LLMClient

        with pytest.raises(CostSinkConfigError):
            LLMClient(self._spec(), callsite_id="c")
