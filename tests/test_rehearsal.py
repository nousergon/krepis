"""A rehearsal's bookkeeping never lands at a live key (alpha-engine-config-I12351).

`rehearsal-2026-10-16-2` wrote 19 live ``_stage_coverage/<date>/<Stage>.json``
verdicts and six live ``ops/checks/ae-rss-*/latest.json`` console rows. These
tests pin the two chokepoints that wrote them, plus the pure helper both use.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from krepis import rehearsal
from krepis import rss_budget as rb
from krepis import stage_coverage as sc

TOKEN = "rehearsal-2026-10-16-2"


@pytest.fixture(autouse=True)
def _no_ambient_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(rehearsal.RUN_TOKEN_ENV, raising=False)
    monkeypatch.delenv(rehearsal.PIPELINE_ROLE_ENV, raising=False)


# ── krepis.rehearsal ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name",
    [
        TOKEN,
        f"arn:aws:states:us-east-1:111111111111:execution:pipe:{TOKEN}",
    ],
)
def test_an_explicit_rehearsal_name_or_arn_is_a_rehearsal(name: str) -> None:
    assert rehearsal.rehearsal_token(name, env={}) == TOKEN


@pytest.mark.parametrize(
    "name",
    [
        "f4b49786-4d8b-793d-cfa7-419f26714770_32a27b1a",
        "weekly-rerun-2026-10-10-1",
        "watch-rerun-2026-10-02-3",
        "",
        None,
    ],
)
def test_a_production_or_absent_name_is_not_a_rehearsal(name: str | None) -> None:
    assert rehearsal.rehearsal_token(name, env={}) is None
    assert rehearsal.is_rehearsal(name, env={}) is False


def test_run_token_decides_when_no_name_is_given() -> None:
    assert rehearsal.rehearsal_token(env={"RUN_TOKEN": TOKEN}) == TOKEN
    assert rehearsal.rehearsal_token(env={"RUN_TOKEN": "f4b49786"}) is None


def test_an_explicit_production_name_outranks_a_stray_run_token() -> None:
    """The name a caller passes is the execution it belongs to."""
    env = {"RUN_TOKEN": TOKEN}
    assert rehearsal.rehearsal_token("f4b49786-4d8b", env=env) is None


def test_the_explicit_role_override_marks_an_unnamed_process() -> None:
    env = {"AE_PIPELINE_ROLE": "Rehearsal"}
    assert rehearsal.rehearsal_token(env=env) == rehearsal.UNNAMED_TOKEN


def test_the_token_is_one_safe_path_segment() -> None:
    assert rehearsal.rehearsal_token("rehearsal-x/../y z", env={}) == "rehearsal-x..yz"


def test_namespaced_key_inserts_below_the_top_level_prefix() -> None:
    assert (
        rehearsal.namespaced_key("_stage_coverage/2026-10-15/Backtester.json", TOKEN)
        == f"_stage_coverage/.rehearsal/{TOKEN}/2026-10-15/Backtester.json"
    )
    assert rehearsal.namespaced_key("top.json", TOKEN) == f".rehearsal/{TOKEN}/top.json"


def test_namespaced_key_is_the_identity_for_production() -> None:
    key = "_stage_coverage/2026-10-16/Backtester.json"
    assert rehearsal.namespaced_key(key, None) == key


# ── krepis.stage_coverage ────────────────────────────────────────────────────


class _S3:
    def __init__(self) -> None:
        self.puts: list[dict] = []

    def put_object(self, **kwargs) -> dict:
        self.puts.append(kwargs)
        return {}


class _CW:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def put_metric_data(self, **kwargs) -> None:
        self.calls.append(kwargs)


def _verdict(status: str = sc.STATUS_MISSING) -> sc.StageVerdict:
    return sc.StageVerdict(
        stage="Backtester",
        status=status,
        reason="test",
        run_date="2026-10-15",
        recorded_at=datetime(2026, 10, 16, 21, 59, tzinfo=timezone.utc).isoformat(),
    )


def test_a_production_verdict_keeps_its_live_key_and_metric() -> None:
    s3, cw = _S3(), _CW()
    sc.record_verdict(_verdict(), s3_client=s3, cloudwatch_client=cw)
    assert [p["Key"] for p in s3.puts] == ["_stage_coverage/2026-10-15/Backtester.json"]
    assert len(cw.calls) == 1


def test_a_rehearsal_verdict_goes_to_its_namespace_and_publishes_no_metric() -> None:
    s3, cw = _S3(), _CW()
    sc.record_verdict(
        _verdict(), s3_client=s3, cloudwatch_client=cw, execution_name=TOKEN
    )
    assert [p["Key"] for p in s3.puts] == [
        f"_stage_coverage/.rehearsal/{TOKEN}/2026-10-15/Backtester.json"
    ]
    assert json.loads(s3.puts[0]["Body"])["stage"] == "Backtester"
    assert cw.calls == []


def test_run_token_alone_isolates_a_launcher_verdict(monkeypatch: pytest.MonkeyPatch) -> None:
    """Spot launchers call the CLI under ssm_log_capture, which exports RUN_TOKEN."""
    monkeypatch.setenv("RUN_TOKEN", TOKEN)
    s3, cw = _S3(), _CW()
    sc.record_verdict(_verdict(), s3_client=s3, cloudwatch_client=cw)
    assert s3.puts[0]["Key"].startswith(f"_stage_coverage/.rehearsal/{TOKEN}/")
    assert cw.calls == []


def test_assert_stage_coverage_threads_the_execution_name() -> None:
    s3, cw = _S3(), _CW()

    class _Registry:
        def get_object(self, Bucket, Key):  # noqa: N803
            raise RuntimeError("registry unreachable")

        put_object = s3.put_object

    out = sc.assert_stage_coverage(
        "Backtester",
        run_date="2026-10-15",
        s3_client=_Registry(),
        cloudwatch_client=cw,
        execution_name=f"arn:aws:states:us-east-1:1:execution:p:{TOKEN}",
    )
    assert out["status"] == sc.STATUS_UNMEASURED
    assert [p["Key"] for p in s3.puts] == [
        f"_stage_coverage/.rehearsal/{TOKEN}/2026-10-15/Backtester.json"
    ]
    assert cw.calls == []


def test_the_cli_takes_the_execution_name(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}

    def _fake(stage, **kw):
        seen.update(kw)
        return {"status": sc.STATUS_COVERED, "reason": "ok"}

    monkeypatch.setattr(sc, "assert_stage_coverage", _fake)
    rc = sc.main(
        ["assert", "--stage", "Backtester", "--run-date", "2026-10-15",
         "--execution-name", TOKEN]
    )
    assert rc == 0
    assert seen["execution_name"] == TOKEN


# ── krepis.rss_budget ────────────────────────────────────────────────────────


class _RssS3:
    def __init__(self, objects: dict | None = None) -> None:
        self.objects = dict(objects or {})
        self.gets: list[str] = []
        self.puts: dict = {}

    def get_object(self, Bucket, Key):  # noqa: N803
        self.gets.append(Key)
        if Key not in self.objects:
            raise RuntimeError("NoSuchKey")

        class _Body:
            def __init__(self, data: bytes) -> None:
                self._data = data

            def read(self) -> bytes:
                return self._data

        return {"Body": _Body(json.dumps(self.objects[Key]).encode())}

    def put_object(self, Bucket, Key, Body, ContentType=None):  # noqa: N803
        self.puts[Key] = json.loads(Body.decode())


def _stdout() -> str:
    reading = {
        "measured": True,
        "peak_rss_kb": 1_000_000,
        "mem_total_kb": 4_000_000,
        "instance_type": "c5.large",
        "run_token": TOKEN,
    }
    return f"out\n{rb.SENTINEL} {json.dumps(reading)}\n"


LIVE_ROW = "ops/checks/ae-rss-backtester/latest.json"


def test_a_rehearsal_rss_row_never_touches_the_live_console_row() -> None:
    s3 = _RssS3({LIVE_ROW: {"history": [], "status": "ok"}})
    body = rb.publish(
        bucket="b",
        description="backtester: backtester",
        instance_id="i-1",
        stdout=_stdout(),
        correlation_id=TOKEN,
        s3_client=s3,
    )
    want = f"ops/.rehearsal/{TOKEN}/checks/ae-rss-backtester/latest.json"
    assert body is not None
    assert list(s3.puts) == [want]
    assert s3.gets == [want]
    assert s3.objects[LIVE_ROW] == {"history": [], "status": "ok"}


def test_a_production_rss_row_is_unchanged() -> None:
    s3 = _RssS3()
    rb.publish(
        bucket="b",
        description="backtester: backtester",
        instance_id="i-1",
        stdout=_stdout(),
        correlation_id="f4b49786-4d8b-793d",
        s3_client=s3,
    )
    assert list(s3.puts) == [LIVE_ROW]
