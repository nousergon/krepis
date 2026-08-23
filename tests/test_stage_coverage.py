"""Tests for :mod:`krepis.stage_coverage` — the per-stage output assertion.

Every status has a test that PRODUCES it, in both polarities where the status
has two. This fleet has shipped three detectors that could not fail; these are
the proof this one can — and, just as importantly, that it cannot fail the
stage it observes.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pytest

from krepis import stage_coverage as sc

NOW = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc)
CYCLE = date(2026, 8, 14)
WINDOW = datetime(2026, 8, 15, 9, 0, tzinfo=timezone.utc)


# ── Fakes ────────────────────────────────────────────────────────────────────


class _ClientError(Exception):
    """Minimal stand-in for botocore's ClientError shape."""

    def __init__(self, code: str, status: int) -> None:
        super().__init__(code)
        self.response = {
            "Error": {"Code": code},
            "ResponseMetadata": {"HTTPStatusCode": status},
        }


class FakeS3:
    def __init__(self, objects: dict[str, Any] | None = None) -> None:
        # key -> LastModified datetime, or an exception to raise
        self.objects = objects or {}
        self.puts: list[dict[str, Any]] = []

    def head_object(self, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
        entry = self.objects.get(Key)
        if entry is None:
            raise _ClientError("404", 404)
        if isinstance(entry, Exception):
            raise entry
        return {"LastModified": entry}

    def put_object(self, **kwargs: Any) -> dict[str, Any]:
        self.puts.append(kwargs)
        return {}

    def get_object(self, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
        raise AssertionError("registry should be injected in these tests")


class FakeCW:
    def __init__(self, fail: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail = fail

    def put_metric_data(self, **kwargs: Any) -> None:
        if self.fail:
            raise RuntimeError("cloudwatch down")
        self.calls.append(kwargs)


def registry(
    *,
    stage_rows: list[dict[str, Any]] | None = None,
    artifacts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "pipeline_stages": stage_rows
        if stage_rows is not None
        else [
            {
                "stage": "MorningEnrich",
                "stage_class": "product",
                "output": "registered",
                "artifacts": ["daily_closes_parquet"],
            },
            {
                "stage": "WeeklyPreflight",
                "stage_class": "infrastructure",
                "output": "none",
                "reason": "a gate; its verdict is its output",
            },
        ],
        "artifacts": artifacts
        if artifacts is not None
        else [
            {
                "artifact_id": "daily_closes_parquet",
                "s3_bucket": "alpha-engine-research",
                "s3_key_template": "prices/daily/{date}/closes.parquet",
            }
        ],
    }


KEY = "prices/daily/2026-08-14/closes.parquet"


def _evaluate(s3: FakeS3, **kw: Any) -> sc.StageVerdict:
    params: dict[str, Any] = {
        "s3_client": s3,
        "now": NOW,
        "run_date": "2026-08-15",
        "cycle_date": CYCLE,
        "window_start": WINDOW,
    }
    params.update(kw)
    reg = params.pop("registry", None) or registry()
    stage = params.pop("stage", "MorningEnrich")
    return sc.evaluate_stage(reg, stage, **params)


# ── COVERED ──────────────────────────────────────────────────────────────────


def test_covered_when_the_declared_artifact_was_written_in_window() -> None:
    verdict = _evaluate(FakeS3({KEY: WINDOW + timedelta(minutes=30)}))
    assert verdict.status == sc.STATUS_COVERED
    assert verdict.covered == ["daily_closes_parquet"]
    assert verdict.missing == [] and verdict.stale == []
    assert verdict.is_finding is False


def test_covered_requires_the_cycle_date_not_the_run_date() -> None:
    """`{date}` is the CYCLE tick, not the execution's run_date.

    Measured 2026-08-13: the 08-08 execution carries run_date=2026-08-08 while
    every artifact it produced landed under 2026-08-07. Resolving with run_date
    produced 28 confident and entirely false `missing` verdicts.
    """
    s3 = FakeS3({KEY: WINDOW + timedelta(minutes=30)})
    assert _evaluate(s3).status == sc.STATUS_COVERED
    # The run_date-keyed key does not exist; if the module ever substituted it,
    # this stage would read MISSING.
    wrong = _evaluate(s3, cycle_date=date(2026, 8, 15))
    assert wrong.status == sc.STATUS_MISSING


# ── COVERED_NO_OUTPUT ────────────────────────────────────────────────────────


def test_a_stage_declaring_no_output_is_covered_by_having_said_so() -> None:
    verdict = _evaluate(FakeS3(), stage="WeeklyPreflight")
    assert verdict.status == sc.STATUS_COVERED_NO_OUTPUT
    assert verdict.declared_output == "none"
    assert verdict.is_finding is False


def test_declaring_no_output_still_produces_a_recorded_verdict() -> None:
    """ "Declares nothing" and "was never considered" must not be one absence."""
    s3, cw = FakeS3(), FakeCW()
    verdict = _evaluate(FakeS3(), stage="WeeklyPreflight")
    sc.record_verdict(verdict, s3_client=s3, cloudwatch_client=cw)
    assert len(s3.puts) == 1
    body = json.loads(s3.puts[0]["Body"])
    assert body["stage"] == "WeeklyPreflight"
    assert body["status"] == sc.STATUS_COVERED_NO_OUTPUT
    assert cw.calls[0]["MetricData"][0]["Value"] == 1.0


# ── MISSING ──────────────────────────────────────────────────────────────────


def test_missing_when_the_declared_artifact_does_not_exist() -> None:
    verdict = _evaluate(FakeS3())
    assert verdict.status == sc.STATUS_MISSING
    assert verdict.missing == ["daily_closes_parquet"]
    assert verdict.is_finding is True


def test_missing_is_still_measurable_without_a_window() -> None:
    """Absence does not need a window; only staleness does."""
    verdict = _evaluate(FakeS3(), window_start=None)
    assert verdict.status == sc.STATUS_MISSING


def test_a_real_absence_outranks_a_sibling_probe_fault() -> None:
    """A harness fault beside a confirmed finding must not mask the finding."""
    reg = registry(
        stage_rows=[
            {
                "stage": "MorningEnrich",
                "stage_class": "product",
                "output": "registered",
                "artifacts": ["daily_closes_parquet", "other"],
            }
        ],
        artifacts=[
            {
                "artifact_id": "daily_closes_parquet",
                "s3_key_template": "prices/daily/{date}/closes.parquet",
            },
            {"artifact_id": "other", "s3_key_template": "other/{date}.json"},
        ],
    )
    s3 = FakeS3({"other/2026-08-14.json": _ClientError("403", 403)})
    verdict = _evaluate(s3, registry=reg)
    assert verdict.status == sc.STATUS_MISSING
    assert verdict.missing == ["daily_closes_parquet"]
    assert [pair[0] for pair in verdict.unmeasured] == ["other"]


# ── STALE ────────────────────────────────────────────────────────────────────


def test_stale_when_the_artifact_predates_this_runs_window() -> None:
    """A leftover satisfies every existence-only probe while the consumer
    reads last week's belief. `missing` and `stale` are different defects."""
    verdict = _evaluate(FakeS3({KEY: WINDOW - timedelta(days=7)}))
    assert verdict.status == sc.STATUS_STALE
    assert verdict.stale == ["daily_closes_parquet"]


def test_wholly_stale_is_always_a_finding() -> None:
    """`ReplayConcordance` shape, measured 2026-08-22: it declares exactly
    ONE artifact and that artifact is stale — zero covered. STALE with zero
    covered is indistinguishable from the stage not having run at all and is
    a finding unconditionally, full stop (`alpha-engine-config-I8166`)."""
    verdict = _evaluate(FakeS3({KEY: WINDOW - timedelta(days=7)}))
    assert verdict.status == sc.STATUS_STALE
    assert verdict.covered == []
    assert verdict.stale == ["daily_closes_parquet"]
    assert verdict.is_finding is True
    assert "WHOLLY STALE" in verdict.reason


_MULTI_KEY_COVERED = "champions/2026-08-14/covered.json"
_MULTI_KEY_STALE = "champions/2026-08-14/predictor_meta_champion_weights.json"


def _multi_registry(*, refresh: str | None = None, refresh_reason: str | None = None) -> dict[str, Any]:
    """`PredictorTraining` shape, measured 2026-08-22: 7 covered, 1 stale
    (`predictor_meta_champion_weights` — the artifact champion selection
    writes)."""
    artifact_row: dict[str, Any] = {
        "artifact_id": "predictor_meta_champion_weights",
        "s3_bucket": "alpha-engine-research",
        "s3_key_template": "champions/{date}/predictor_meta_champion_weights.json",
    }
    if refresh is not None:
        artifact_row["refresh"] = refresh
    if refresh_reason is not None:
        artifact_row["refresh_reason"] = refresh_reason
    return registry(
        stage_rows=[
            {
                "stage": "PredictorTraining",
                "stage_class": "product",
                "output": "registered",
                "artifacts": ["covered_champion", "predictor_meta_champion_weights"],
            }
        ],
        artifacts=[
            {
                "artifact_id": "covered_champion",
                "s3_bucket": "alpha-engine-research",
                "s3_key_template": "champions/{date}/covered.json",
            },
            artifact_row,
        ],
    )


def test_partially_stale_with_no_conditional_declaration_is_a_finding() -> None:
    """`PredictorTraining` shape, measured 2026-08-22: `covered=7, stale=1`
    (`predictor_meta_champion_weights`), no `refresh: conditional` row — a
    promotion decision nobody can confirm happened. A finding."""
    s3 = FakeS3(
        {
            _MULTI_KEY_COVERED: WINDOW + timedelta(minutes=30),
            _MULTI_KEY_STALE: WINDOW - timedelta(days=7),
        }
    )
    verdict = _evaluate(s3, registry=_multi_registry(), stage="PredictorTraining")
    assert verdict.status == sc.STATUS_STALE
    assert verdict.covered == ["covered_champion"]
    assert verdict.stale == ["predictor_meta_champion_weights"]
    assert verdict.stale_conditional == []
    assert verdict.is_finding is True


def test_partially_stale_with_valid_conditional_declaration_is_not_a_finding() -> None:
    """A stale artifact whose registry row declares `refresh: conditional`
    with a `refresh_reason` is the legitimate case: no new grades this week,
    no promotion this week. Suppressed, and the suppression is named."""
    s3 = FakeS3(
        {
            _MULTI_KEY_COVERED: WINDOW + timedelta(minutes=30),
            _MULTI_KEY_STALE: WINDOW - timedelta(days=7),
        }
    )
    reg = _multi_registry(
        refresh="conditional", refresh_reason="only refreshed on a promotion week"
    )
    verdict = _evaluate(s3, registry=reg, stage="PredictorTraining")
    assert verdict.status == sc.STATUS_STALE
    assert verdict.stale_conditional == ["predictor_meta_champion_weights"]
    assert verdict.is_finding is False
    assert "suppressed by a valid `refresh: conditional`" in verdict.reason
    assert "predictor_meta_champion_weights" in verdict.reason


def test_conditional_without_refresh_reason_does_not_suppress() -> None:
    """`refresh: conditional` with no `refresh_reason` is NOT a valid
    declaration — same shape as `COVERED_NO_OUTPUT`: an exception to the
    default must be SAID. It must not suppress, and the reason must name the
    invalid attempt rather than silently drop it."""
    s3 = FakeS3(
        {
            _MULTI_KEY_COVERED: WINDOW + timedelta(minutes=30),
            _MULTI_KEY_STALE: WINDOW - timedelta(days=7),
        }
    )
    reg = _multi_registry(refresh="conditional")  # no refresh_reason
    verdict = _evaluate(s3, registry=reg, stage="PredictorTraining")
    assert verdict.status == sc.STATUS_STALE
    assert verdict.stale_conditional == []
    assert verdict.is_finding is True
    assert "INVALID" in verdict.reason
    assert "predictor_meta_champion_weights" in verdict.reason


def test_a_naive_last_modified_is_treated_as_utc_not_as_stale() -> None:
    naive = (WINDOW + timedelta(minutes=5)).replace(tzinfo=None)
    verdict = _evaluate(FakeS3({KEY: naive}))
    assert verdict.status == sc.STATUS_COVERED


# ── UNMEASURED — never a pass, never a finding ───────────────────────────────


def test_a_stage_absent_from_the_registry_is_unmeasured_not_covered() -> None:
    verdict = _evaluate(FakeS3(), stage="NoSuchStage")
    assert verdict.status == sc.STATUS_UNMEASURED
    assert verdict.is_finding is False
    assert "has declared nothing" in verdict.reason


def test_a_non_404_probe_failure_is_unmeasured_not_missing() -> None:
    """A detector reporting its own harness fault AS the defect is this
    fleet's most-repeated detector bug, and it always errs alarming."""
    verdict = _evaluate(FakeS3({KEY: _ClientError("403", 403)}))
    assert verdict.status == sc.STATUS_UNMEASURED
    assert verdict.missing == []
    assert verdict.is_finding is False


def test_a_present_key_with_no_window_is_unmeasured_not_covered() -> None:
    """Degrading it to healthy is how a stage that stopped writing hides
    behind last cycle's object."""
    verdict = _evaluate(FakeS3({KEY: WINDOW}), window_start=None)
    assert verdict.status == sc.STATUS_UNMEASURED
    assert verdict.covered == []


def test_an_unresolvable_cycle_date_is_unmeasured_not_a_guess() -> None:
    verdict = _evaluate(FakeS3({KEY: WINDOW}), cycle_date=None)
    assert verdict.status == sc.STATUS_UNMEASURED
    assert "refusing to guess" in verdict.reason


def test_registered_with_an_empty_artifact_list_is_unmeasured() -> None:
    reg = registry(
        stage_rows=[{"stage": "MorningEnrich", "output": "registered", "artifacts": []}]
    )
    verdict = _evaluate(FakeS3(), registry=reg)
    assert verdict.status == sc.STATUS_UNMEASURED
    assert "self-contradictory" in verdict.reason


def test_an_artifact_id_with_no_registry_row_is_unmeasured() -> None:
    reg = registry(artifacts=[])
    verdict = _evaluate(FakeS3(), registry=reg)
    assert verdict.status == sc.STATUS_UNMEASURED
    assert verdict.unmeasured == [
        ("daily_closes_parquet", "no artifacts: row for this id")
    ]


def test_a_partial_measurement_is_unmeasured_and_names_the_split() -> None:
    reg = registry(
        stage_rows=[
            {
                "stage": "MorningEnrich",
                "output": "registered",
                "artifacts": ["daily_closes_parquet", "other"],
            }
        ],
        artifacts=[
            {
                "artifact_id": "daily_closes_parquet",
                "s3_key_template": "prices/daily/{date}/closes.parquet",
            },
            {"artifact_id": "other", "s3_key_template": "other/{date}.json"},
        ],
    )
    s3 = FakeS3(
        {
            KEY: WINDOW + timedelta(minutes=1),
            "other/2026-08-14.json": _ClientError("503", 503),
        }
    )
    verdict = _evaluate(s3, registry=reg)
    assert verdict.status == sc.STATUS_UNMEASURED
    assert verdict.covered == ["daily_closes_parquet"]
    assert "1 of 2 artifacts confirmed" in verdict.reason


# ── Key formatting ───────────────────────────────────────────────────────────


def test_format_key_substitutes_date_and_trading_day_alike() -> None:
    assert sc.format_key("a/{date}/b", CYCLE) == "a/2026-08-14/b"
    assert sc.format_key("a/{trading_day}.json", CYCLE) == "a/2026-08-14.json"


def test_format_key_leaves_an_unknown_placeholder_intact() -> None:
    """Fail loud on a key nobody can resolve rather than probe a wrong one."""
    assert sc.format_key("a/{quarter}/b", CYCLE) == "a/{quarter}/b"


def test_format_key_passes_a_pointer_template_through_unchanged() -> None:
    assert sc.format_key("latest_weekly.json", CYCLE) == "latest_weekly.json"


# ── The assertion can never fail the stage it observes ───────────────────────


def test_assert_stage_coverage_never_raises_on_a_dead_registry() -> None:
    class Dead:
        def get_object(self, **_: Any) -> dict[str, Any]:
            raise RuntimeError("s3 down")

        def put_object(self, **_: Any) -> dict[str, Any]:
            return {}

    out = sc.assert_stage_coverage(
        "MorningEnrich", run_date="2026-08-15", s3_client=Dead(), cloudwatch_client=FakeCW(), now=NOW
    )
    assert out["status"] == sc.STATUS_UNMEASURED
    assert "registry unreadable" in out["reason"]


def test_assert_stage_coverage_never_raises_when_everything_fails() -> None:
    class Hostile:
        def __getattr__(self, _name: str) -> Any:
            raise RuntimeError("boom")

    out = sc.assert_stage_coverage(
        "MorningEnrich", run_date="2026-08-15", s3_client=Hostile(), cloudwatch_client=Hostile(), now=NOW
    )
    assert out["status"] == sc.STATUS_UNMEASURED


def test_assert_stage_coverage_builds_clients_with_no_region_in_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for alpha-engine-config-I7428.

    SSM ``AWS-RunShellScript`` runs with neither ``AWS_REGION`` nor
    ``AWS_DEFAULT_REGION`` set and no per-user boto profile — reproduced here
    by clearing both. Before the fix, the internal ``boto3.client("s3")`` /
    ``boto3.client("cloudwatch")`` construction (no clients injected, so the
    module builds its own) raised ``NoRegionError`` on the CloudWatch client
    and the whole assertion degraded to UNMEASURED on every single box run —
    never a genuine COVERED/MISSING/STALE verdict. Patching
    ``krepis.aws_region._from_botocore_session`` and ``_from_imds`` to ``None``
    forces the chain all the way to :data:`krepis.aws_region.DEFAULT_REGION`,
    matching a bare EC2/SSM box with no profile and no IMDS reachable in the
    test sandbox.
    """
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    from krepis import aws_region

    monkeypatch.setattr(aws_region, "_from_botocore_session", lambda: None)
    monkeypatch.setattr(aws_region, "_from_imds", lambda: None)

    captured: dict[str, Any] = {}

    def _spy_client(service_name: str, **kwargs: Any) -> Any:
        captured[service_name] = kwargs.get("region_name")
        # Don't actually touch the network — return a fake that satisfies
        # the calls this test path makes.
        if service_name == "s3":
            return FakeS3({})
        return FakeCW()

    monkeypatch.setattr("boto3.client", _spy_client)

    out = sc.assert_stage_coverage(
        "SaturdayHealthCheck",
        run_date="2026-08-15",
        now=NOW,
        registry_local_path=None,
    )

    assert captured.get("s3") == aws_region.DEFAULT_REGION
    assert captured.get("cloudwatch") == aws_region.DEFAULT_REGION
    # The registry read still fails (FakeS3({}) has no matching key), so the
    # verdict is UNMEASURED — but for a REGISTRY reason, never a region one.
    assert out["status"] == sc.STATUS_UNMEASURED
    assert "NoRegionError" not in out["reason"]
    assert "REGION_RESOLVER_BUG" not in out["reason"]


def test_a_genuine_no_region_error_is_marked_as_a_resolver_bug(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If ``resolve_region()`` itself regresses to returning something boto3
    rejects, the resulting NoRegionError must be distinguishable — on the log
    surface and in the verdict's ``reason`` — from an ordinary environment
    UNMEASURED (unreadable registry, unresolvable cycle date), per
    alpha-engine-config-I7428 deliverable 5: a missing region is a caller
    defect, not an unmeasurable condition."""
    from botocore.exceptions import NoRegionError

    def _boom(service_name: str, **_: Any) -> Any:
        raise NoRegionError()

    monkeypatch.setattr("boto3.client", _boom)

    out = sc.assert_stage_coverage("SaturdayHealthCheck", run_date="2026-08-15", now=NOW)

    assert out["status"] == sc.STATUS_UNMEASURED
    assert out["reason"].startswith("REGION_RESOLVER_BUG:")


def test_record_verdict_is_fail_soft_on_both_surfaces(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class DeadS3:
        def put_object(self, **_: Any) -> dict[str, Any]:
            raise RuntimeError("s3 down")

    verdict = _evaluate(FakeS3({KEY: WINDOW}))
    with caplog.at_level("ERROR"):
        sc.record_verdict(
            verdict, s3_client=DeadS3(), cloudwatch_client=FakeCW(fail=True)
        )
    # Fail-soft AND loud: a silent swallow would shrink the coverage
    # denominator with no signal, which is the defect this mechanism removes.
    assert "FAILED to write verdict" in caplog.text
    assert "FAILED to publish coverage metric" in caplog.text


def test_the_verdict_key_is_per_stage_not_per_run() -> None:
    """The weekly pipeline's Parallel branches assert concurrently; a single
    per-run object would be a lost-update race between them."""
    a = sc.verdict_key(
        sc.StageVerdict(stage="Scanner", status="X", run_date="2026-08-15")
    )
    b = sc.verdict_key(
        sc.StageVerdict(stage="Director", status="X", run_date="2026-08-15")
    )
    assert a == "_stage_coverage/2026-08-15/Scanner.json"
    assert a != b


# ── Cycle-date resolution ────────────────────────────────────────────────────


def test_resolve_cycle_date_returns_none_when_the_calendar_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(_now: datetime) -> date:
        raise RuntimeError("calendar unavailable")

    monkeypatch.setattr(sc, "last_closed_trading_day", boom)
    assert sc.resolve_cycle_date(NOW) is None


def test_resolve_cycle_date_returns_a_date_on_the_happy_path() -> None:
    resolved = sc.resolve_cycle_date(NOW)
    assert isinstance(resolved, date)
    assert resolved <= NOW.date()


# ── CLI ──────────────────────────────────────────────────────────────────────


def _patch_assert(
    monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any]
) -> list[dict]:
    seen: list[dict[str, Any]] = []

    def fake(stage: str, **kw: Any) -> dict[str, Any]:
        seen.append({"stage": stage, **kw})
        return payload

    monkeypatch.setattr(sc, "assert_stage_coverage", fake)
    return seen


def test_cli_exits_zero_in_observe_mode_on_a_real_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OBSERVE MODE is the whole safety property: loud on every surface except
    the one that kills the run."""
    _patch_assert(
        monkeypatch,
        {"status": sc.STATUS_MISSING, "reason": "absent", "is_finding": True},
    )
    assert sc.main(["assert", "--stage", "MorningEnrich", "--run-date", "2026-08-15"]) == 0


def test_cli_exits_three_only_under_enforce_and_only_on_a_finding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_assert(
        monkeypatch,
        {"status": sc.STATUS_MISSING, "reason": "absent", "is_finding": True},
    )
    assert sc.main(["assert", "--stage", "X", "--run-date", "2026-08-15", "--enforce"]) == sc.EXIT_COVERAGE_FAILURE


def test_cli_enforce_exits_zero_on_a_clean_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other polarity. An absence-alarm is blind in BOTH directions and
    GREEN is the dangerous one — a detector that cannot pass is as broken as
    one that cannot fail."""
    _patch_assert(
        monkeypatch,
        {"status": sc.STATUS_COVERED, "reason": "ok", "is_finding": False},
    )
    assert sc.main(["assert", "--stage", "X", "--run-date", "2026-08-15", "--enforce"]) == 0


def test_cli_enforce_does_not_act_on_stale_or_unmeasured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for status in (sc.STATUS_STALE, sc.STATUS_UNMEASURED):
        _patch_assert(
            monkeypatch, {"status": status, "reason": "r", "is_finding": False}
        )
        assert sc.main(["assert", "--stage", "X", "--run-date", "2026-08-15", "--enforce"]) == 0


def test_cli_enforce_defaults_off_and_reads_the_env_knob(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("STAGE_COVERAGE_ENFORCE", raising=False)
    assert sc.build_parser().parse_args(["assert", "--stage", "X"]).enforce is False
    monkeypatch.setenv("STAGE_COVERAGE_ENFORCE", "1")
    assert sc.build_parser().parse_args(["assert", "--stage", "X"]).enforce is True


def test_cli_forwards_run_date_and_window_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _patch_assert(
        monkeypatch, {"status": sc.STATUS_COVERED, "reason": "", "is_finding": False}
    )
    sc.main(
        [
            "assert",
            "--stage",
            "MorningEnrich",
            "--run-date",
            "2026-08-15",
            "--window-start",
            "2026-08-15T09:00:00Z",
        ]
    )
    assert seen[0]["stage"] == "MorningEnrich"
    assert seen[0]["run_date"] == "2026-08-15"
    assert seen[0]["window_start"] == WINDOW


def test_cli_an_unparseable_window_does_not_abort_the_assertion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Absence stays measurable without a window; refusing to run at all would
    trade a measurable half for nothing."""
    seen = _patch_assert(
        monkeypatch, {"status": sc.STATUS_MISSING, "reason": "", "is_finding": True}
    )
    assert sc.main(["assert", "--stage", "X", "--run-date", "2026-08-15", "--window-start", "not-a-date"]) == 0
    assert seen[0]["window_start"] is None


def test_cli_writes_a_warning_to_stderr_for_every_non_covered_status(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_assert(
        monkeypatch,
        {"status": sc.STATUS_MISSING, "reason": "absent", "is_finding": True},
    )
    sc.main(["assert", "--stage", "MorningEnrich", "--run-date", "2026-08-15"])
    assert "WARNING: stage-coverage MorningEnrich: MISSING" in capsys.readouterr().err


def test_parse_window_accepts_z_and_offset_and_bare_forms() -> None:
    assert sc._parse_window("2026-08-15T09:00:00Z") == WINDOW
    assert sc._parse_window("2026-08-15T09:00:00+00:00") == WINDOW
    assert sc._parse_window("2026-08-15T09:00:00") == WINDOW
    assert sc._parse_window("") is None
    assert sc._parse_window(None) is None


# ── Registry access ──────────────────────────────────────────────────────────


def test_load_registry_reads_a_local_path(tmp_path: Any) -> None:
    path = tmp_path / "reg.yaml"
    path.write_text("pipeline_stages:\n  - stage: X\n    output: none\n")
    reg = sc.load_registry(None, local_path=str(path))
    assert reg["pipeline_stages"][0]["stage"] == "X"


def test_resolve_stage_row_returns_none_for_an_unknown_stage() -> None:
    assert sc.resolve_stage_row(registry(), "Nope") is None


def test_index_artifacts_keys_by_artifact_id() -> None:
    assert "daily_closes_parquet" in sc.index_artifacts(registry())


def test_the_registry_mirror_key_is_the_one_config_syncs_on_merge() -> None:
    """The consumer must be refreshed by the same event that changes the
    declaration; a hand-copied registry drifts invisibly."""
    assert sc.REGISTRY_KEY == "_freshness_monitor/ARTIFACT_REGISTRY.yaml"
    assert sc.DEFAULT_BUCKET == "alpha-engine-research"


# ── run_date is the grouping key, and it is REQUIRED (config-I8155) ──────────


def test_a_verdict_cannot_be_constructed_without_a_run_date() -> None:
    """Refusing at construction covers every path that produces a verdict."""
    with pytest.raises(sc.StageCoverageContractError, match="REQUIRED"):
        sc.StageVerdict(stage="Scanner", status=sc.STATUS_COVERED)


def test_a_blank_or_whitespace_run_date_is_refused_too() -> None:
    for blank in ("", "   ", None):
        with pytest.raises(sc.StageCoverageContractError):
            sc.StageVerdict(stage="Scanner", status=sc.STATUS_COVERED, run_date=blank)


def test_a_run_date_is_normalised_on_construction() -> None:
    verdict = sc.StageVerdict(
        stage="Scanner", status=sc.STATUS_COVERED, run_date="  2026-08-22 "
    )
    assert verdict.run_date == "2026-08-22"


def test_evaluate_stage_requires_run_date_as_a_keyword() -> None:
    with pytest.raises(TypeError):
        sc.evaluate_stage(registry(), "MorningEnrich", s3_client=FakeS3(), now=NOW)


def test_evaluate_stage_refuses_a_blank_run_date() -> None:
    with pytest.raises(sc.StageCoverageContractError, match="REQUIRED"):
        _evaluate(FakeS3(), run_date="")


def test_the_verdict_key_never_falls_back_to_the_cycle_date() -> None:
    """The 2026-08-22 split: run_date and cycle_date genuinely differ on a
    weekend, and the fall-back scattered one execution across two prefixes."""
    verdict = sc.StageVerdict(
        stage="Backtester",
        status=sc.STATUS_COVERED,
        run_date="2026-08-22",
        cycle_date="2026-08-21",
    )
    assert sc.verdict_key(verdict) == "_stage_coverage/2026-08-22/Backtester.json"


def test_one_executions_verdicts_share_one_prefix_across_a_weekend() -> None:
    saturday = datetime(2026, 8, 22, 12, 0, tzinfo=timezone.utc)
    s3 = FakeS3({KEY: NOW})
    verdicts = [
        _evaluate(s3, stage=stage, now=saturday, run_date="2026-08-22", cycle_date=date(2026, 8, 21))
        for stage in ("MorningEnrich", "WeeklyPreflight")
    ]
    prefixes = {sc.verdict_key(v).split("/")[1] for v in verdicts}
    assert prefixes == {"2026-08-22"}
    assert {v.cycle_date for v in verdicts} == {"2026-08-21"}


def test_assert_stage_coverage_raises_on_a_missing_run_date() -> None:
    """The ONE thing the front door raises on: a defect in the call site, not
    an encounter with the world. Everything else still degrades to UNMEASURED."""
    with pytest.raises(TypeError):
        sc.assert_stage_coverage("Scanner")
    with pytest.raises(sc.StageCoverageContractError, match="REQUIRED"):
        sc.assert_stage_coverage("Scanner", run_date="")


def test_assert_stage_coverage_still_never_raises_once_run_date_is_supplied() -> None:
    out = sc.assert_stage_coverage(
        "Scanner",
        run_date="2026-08-22",
        s3_client=object(),
        cloudwatch_client=FakeCW(),
    )
    assert out["status"] == sc.STATUS_UNMEASURED
    assert out["run_date"] == "2026-08-22"


def test_the_cli_exits_four_and_writes_nothing_without_a_run_date(
    monkeypatch: Any, capsys: Any
) -> None:
    """A verdict filed under the wrong prefix is worse than none: it is counted."""
    monkeypatch.delenv("RUN_DATE", raising=False)
    called: list[Any] = []
    monkeypatch.setattr(
        sc, "assert_stage_coverage", lambda *a, **k: called.append(k) or {}
    )
    rc = sc.main(["assert", "--stage", "DataPhase1"])
    assert rc == sc.EXIT_NO_RUN_DATE
    assert called == []
    assert "NO VERDICT RECORDED" in capsys.readouterr().err


def test_the_cli_run_date_exit_code_is_distinct_from_a_coverage_finding() -> None:
    assert sc.EXIT_NO_RUN_DATE != sc.EXIT_COVERAGE_FAILURE
