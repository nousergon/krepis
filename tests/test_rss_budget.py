"""Tests for the per-stage peak-RSS budget (alpha-engine-config-I7260).

Contract, not values, wherever a value is a declared knob: the thresholds are
initial values expected to move once surviving-run readings exist, so the tests
assert the BAND STRUCTURE (hard floor below warn floor, trend fires
independently of the latest reading, absence is never green) rather than
pinning 0.15/0.30 in a way that would make the re-derivation a test rewrite.
"""

from __future__ import annotations

import json

import pytest

from krepis import rss_budget as rb


# ── Identifiers ──────────────────────────────────────────────────────────────


def test_description_splits_into_stage_and_step():
    assert rb.split_description("evaluator: bootstrap") == ("evaluator", "bootstrap")
    assert rb.split_description("data-weekly: phase2") == ("data-weekly", "phase2")


def test_description_without_separator_is_a_stage_not_an_error():
    """A label shape is never something an observability path may fail on."""
    assert rb.split_description("evaluator") == ("evaluator", "")


def test_check_id_is_derived_from_the_stage_and_is_stable():
    assert rb.check_id("EvaluatorDiagnostics") == "ae-rss-evaluatordiagnostics"
    assert rb.check_id("data phase1") == rb.check_id("data-phase1")


def test_envelope_key_sits_under_the_console_checks_prefix():
    """The console reads the whole ops/checks/ prefix, so a new stage's row
    appears with no console change and nothing hand-listed."""
    assert rb.envelope_key("evaluator") == "ops/checks/ae-rss-evaluator/latest.json"


# ── Which steps publish ──────────────────────────────────────────────────────


@pytest.mark.parametrize("step", sorted(rb.INFRASTRUCTURE_STEPS))
def test_infrastructure_steps_do_not_publish(step):
    assert rb.is_publishable_step(step) is False


def test_preflight_only_is_excluded_so_the_daily_sweep_cannot_manufacture_headroom():
    """The daily all-stage sweep (I7249) drives every launcher with
    --preflight-only against a smoke workload. Publishing those readings would
    flood every row with rosy headroom from a workload that never touches the
    ~900-ticker universe."""
    assert rb.is_publishable_step("preflight-only") is False


def test_the_stage_workload_publishes():
    assert rb.is_publishable_step("evaluator") is True
    assert rb.is_publishable_step("full-training") is True


# ── The wrapper ──────────────────────────────────────────────────────────────


def test_wrapper_preserves_a_zero_exit(tmp_path):
    out = _run_wrapped("exit 0\n", tmp_path)
    assert out.returncode == 0


def test_wrapper_preserves_a_nonzero_exit(tmp_path):
    """The invariant that makes this safe to enable by default: the stage's
    own exit status propagates verbatim through the measurement wrapper."""
    out = _run_wrapped("exit 42\n", tmp_path)
    assert out.returncode == 42


def test_wrapper_preserves_the_bodys_stdout(tmp_path):
    out = _run_wrapped("echo hello-from-the-stage\n", tmp_path)
    assert "hello-from-the-stage" in out.stdout


def test_wrapper_body_may_contain_heredocs_and_quotes(tmp_path):
    """The body is carried as base64, so no delimiter, quote or heredoc in a
    stage payload can break the wrapper."""
    body = "cat <<'EOF'\nit's a \"quoted\" $dollar 'heredoc'\nEOF\n"
    out = _run_wrapped(body, tmp_path)
    assert out.returncode == 0
    assert "it's a \"quoted\" $dollar 'heredoc'" in out.stdout


def test_wrapper_emits_a_parseable_sentinel(tmp_path):
    out = _run_wrapped("true\n", tmp_path)
    reading = rb.parse_reading(out.stdout)
    assert reading is not None
    assert "measured" in reading


def _run_wrapped(body, tmp_path):
    import subprocess

    path = tmp_path / "wrapped.sh"
    path.write_text(rb.wrap_script(body))
    return subprocess.run(["bash", str(path)], capture_output=True, text=True)


# ── Sentinel parsing ─────────────────────────────────────────────────────────


def test_missing_sentinel_parses_to_none_not_to_an_unmeasured_reading():
    """`None` (nothing arrived) and `{"measured": false}` (the harness ran and
    could not measure) are different conditions and render different reasons."""
    assert rb.parse_reading("just some workload output\n") is None


def test_last_sentinel_wins_over_one_echoed_by_the_workload():
    stdout = (
        f'{rb.SENTINEL} {{"measured": true, "peak_rss_kb": 1, "mem_total_kb": 2}}\n'
        "...workload...\n"
        f'{rb.SENTINEL} {{"measured": true, "peak_rss_kb": 99, "mem_total_kb": 100}}\n'
    )
    assert rb.parse_reading(stdout)["peak_rss_kb"] == 99


def test_a_malformed_sentinel_does_not_raise():
    assert rb.parse_reading(f"{rb.SENTINEL} {{not json}}\n") is None


# ── Headroom ─────────────────────────────────────────────────────────────────


def test_headroom_is_the_fraction_left_free():
    assert rb.headroom(4 * 1024 * 1024, 16 * 1024 * 1024) == pytest.approx(0.75)


def test_headroom_is_clamped_at_zero_when_peak_exceeds_memtotal():
    """ru_maxrss can exceed MemTotal with swap or a brief overcommit; "zero
    free" is what that means operationally."""
    assert rb.headroom(20, 16) == 0.0


def test_headroom_refuses_a_nonpositive_denominator():
    with pytest.raises(ValueError):
        rb.headroom(100, 0)


# ── The warn band ────────────────────────────────────────────────────────────


def test_the_hard_floor_sits_below_the_warn_floor():
    """Structural invariant. If this ever inverts, the early warning fires
    after the breach it was supposed to precede."""
    assert 0 < rb.HARD_HEADROOM_FLOOR < rb.WARN_HEADROOM_FLOOR < 1


def test_comfortable_headroom_is_ok():
    status, _ = rb.classify(_reading(headroom=0.60), [])
    assert status == rb.ENVELOPE_OK


def test_the_warn_band_fires_before_the_hard_floor():
    """The induction the issue asks for: walking a stage down toward the wall
    must produce attention BEFORE error."""
    warn, _ = rb.classify(
        _reading(headroom=(rb.HARD_HEADROOM_FLOOR + rb.WARN_HEADROOM_FLOOR) / 2), []
    )
    hard, _ = rb.classify(_reading(headroom=rb.HARD_HEADROOM_FLOOR / 2), [])
    assert warn == rb.ENVELOPE_ATTENTION
    assert hard == rb.ENVELOPE_ERROR


def test_a_declining_trend_warns_even_when_this_run_cleared_the_floor():
    """Alarm on the trend, not on the kill (sf-pipeline-policy §1.2)."""
    history = [{"headroom": h} for h in (0.28, 0.26, 0.24)]
    status, summary = rb.classify(_reading(headroom=0.45), history)
    assert status == rb.ENVELOPE_ATTENTION
    assert "eroding" in summary


def test_a_trend_is_never_fabricated_from_too_few_points():
    history = [{"headroom": 0.01}] * (rb.MIN_TREND_SAMPLES - 1)
    assert rb.trend_median(history) is None
    status, _ = rb.classify(_reading(headroom=0.60), history)
    assert status == rb.ENVELOPE_OK


# ── Absence renders UNOBSERVED, never healthy ────────────────────────────────


def test_no_reading_is_attention_not_ok():
    """principles.md §2.7 — a stage with no reading is unobserved, not
    healthy."""
    status, summary = rb.classify(None, [])
    assert status == rb.ENVELOPE_ATTENTION
    assert "UNOBSERVED" in summary


def test_no_reading_is_not_an_error_either():
    """A failed measurement on a stage that did its work is not a finding
    about the pipeline; routing it to `error` would page an operator about an
    observability fault wearing a production fault's clothes."""
    status, _ = rb.classify(None, [])
    assert status != rb.ENVELOPE_ERROR


def test_an_unmeasured_reading_carries_its_reason_onto_the_surface():
    status, summary = rb.classify(
        {"measured": False, "reason": "python3 absent on instance"}, []
    )
    assert status == rb.ENVELOPE_ATTENTION
    assert "python3 absent on instance" in summary


# ── Envelope shape and merge rules ───────────────────────────────────────────


def test_envelope_carries_the_four_field_row_contract():
    body = rb.build_envelope(
        stage="evaluator",
        step="evaluator",
        reading=_reading(headroom=0.5),
        previous=None,
        instance_id="i-1",
    )
    for field in ("check_id", "status", "ran_at", "cadence_minutes", "summary"):
        assert field in body
    assert body["check_id"] == "ae-rss-evaluator"


def test_envelope_declares_its_thresholds_as_initial_values():
    """The honesty requirement: the surface must not present a guessed
    threshold as a measured one."""
    body = rb.build_envelope(
        stage="evaluator",
        step="evaluator",
        reading=_reading(headroom=0.5),
        previous=None,
        instance_id="i-1",
    )
    assert "INITIAL VALUES" in body["thresholds_basis"]
    assert "I7260" in body["thresholds_basis"]


def test_a_later_lighter_step_on_the_same_box_never_lowers_the_peak():
    """What makes INFRASTRUCTURE_STEPS a noise filter rather than a
    correctness dependency."""
    heavy = rb.build_envelope(
        stage="evaluator",
        step="evaluator",
        reading=_reading(headroom=0.10),
        previous=None,
        instance_id="i-1",
    )
    light = rb.build_envelope(
        stage="evaluator",
        step="tail-step",
        reading=_reading(headroom=0.90),
        previous=heavy,
        instance_id="i-1",
    )
    assert light["peak_rss_kb"] == heavy["peak_rss_kb"]
    assert light["status"] == rb.ENVELOPE_ERROR


def test_a_new_instance_retires_the_previous_reading_into_the_trend():
    prev = rb.build_envelope(
        stage="evaluator",
        step="evaluator",
        reading=_reading(headroom=0.40),
        previous=None,
        instance_id="i-1",
    )
    assert prev["history"] == []
    nxt = rb.build_envelope(
        stage="evaluator",
        step="evaluator",
        reading=_reading(headroom=0.38),
        previous=prev,
        instance_id="i-2",
    )
    assert len(nxt["history"]) == 1
    assert nxt["history"][0]["headroom"] == prev["headroom"]


def test_history_is_bounded():
    body = None
    for n in range(rb.HISTORY_LIMIT + 5):
        body = rb.build_envelope(
            stage="evaluator",
            step="evaluator",
            reading=_reading(headroom=0.5),
            previous=body,
            instance_id=f"i-{n}",
        )
    assert len(body["history"]) == rb.HISTORY_LIMIT


def test_an_unmeasured_run_does_not_erase_a_measured_reading_from_the_same_box():
    measured = rb.build_envelope(
        stage="evaluator",
        step="evaluator",
        reading=_reading(headroom=0.20),
        previous=None,
        instance_id="i-1",
    )
    after = rb.build_envelope(
        stage="evaluator",
        step="tail",
        reading=None,
        previous=measured,
        instance_id="i-1",
    )
    assert after["measured"] is True
    assert after["peak_rss_kb"] == measured["peak_rss_kb"]


# ── publish() — never raises, never publishes an infrastructure step ─────────


class _FakeS3:
    def __init__(self, objects=None, fail_put=False, fail_get=False):
        self.objects = dict(objects or {})
        self.puts = {}
        self.fail_put = fail_put
        self.fail_get = fail_get

    def get_object(self, Bucket, Key):  # noqa: N803 - boto3 kwarg names
        if self.fail_get or Key not in self.objects:
            raise RuntimeError("NoSuchKey")

        class _Body:
            def __init__(self, data):
                self._data = data

            def read(self):
                return self._data

        return {"Body": _Body(json.dumps(self.objects[Key]).encode())}

    def put_object(self, Bucket, Key, Body, ContentType=None):  # noqa: N803
        if self.fail_put:
            raise RuntimeError("AccessDenied")
        self.puts[Key] = json.loads(Body.decode())
        self.objects[Key] = self.puts[Key]


def _stdout_with(reading):
    return f"work output\n{rb.SENTINEL} {json.dumps(reading)}\n"


def test_publish_writes_the_stage_row():
    s3 = _FakeS3()
    body = rb.publish(
        bucket="b",
        description="evaluator: evaluator",
        instance_id="i-1",
        stdout=_stdout_with(_reading(headroom=0.5)),
        s3_client=s3,
    )
    assert body is not None
    assert "ops/checks/ae-rss-evaluator/latest.json" in s3.puts


def test_publish_skips_infrastructure_steps():
    s3 = _FakeS3()
    assert (
        rb.publish(
            bucket="b",
            description="evaluator: bootstrap",
            instance_id="i-1",
            stdout=_stdout_with(_reading(headroom=0.5)),
            s3_client=s3,
        )
        is None
    )
    assert s3.puts == {}


def test_publish_swallows_an_s3_failure_and_returns_none():
    """(a) the swallowed failure mode, (b) the stage's exit code is computed by
    ssm_dispatcher.run from the SSM terminal status and never from this."""
    s3 = _FakeS3(fail_put=True)
    assert (
        rb.publish(
            bucket="b",
            description="evaluator: evaluator",
            instance_id="i-1",
            stdout=_stdout_with(_reading(headroom=0.5)),
            s3_client=s3,
        )
        is None
    )


def test_publish_still_writes_a_row_when_no_reading_arrived():
    """A stage that ran and reported nothing must still occupy a row — an
    omitted row is indistinguishable from a stage that does not exist."""
    s3 = _FakeS3()
    body = rb.publish(
        bucket="b",
        description="evaluator: evaluator",
        instance_id="i-1",
        stdout="workload output with no sentinel\n",
        s3_client=s3,
    )
    assert body["status"] == rb.ENVELOPE_ATTENTION
    assert body["measured"] is False


def test_publish_folds_the_previous_row_into_the_trend():
    key = "ops/checks/ae-rss-evaluator/latest.json"
    first = rb.build_envelope(
        stage="evaluator",
        step="evaluator",
        reading=_reading(headroom=0.4),
        previous=None,
        instance_id="i-1",
    )
    s3 = _FakeS3({key: first})
    body = rb.publish(
        bucket="b",
        description="evaluator: evaluator",
        instance_id="i-2",
        stdout=_stdout_with(_reading(headroom=0.35)),
        s3_client=s3,
    )
    assert len(body["history"]) == 1


def test_publish_survives_an_unreadable_previous_row():
    s3 = _FakeS3(fail_get=True)
    body = rb.publish(
        bucket="b",
        description="evaluator: evaluator",
        instance_id="i-1",
        stdout=_stdout_with(_reading(headroom=0.5)),
        s3_client=s3,
    )
    assert body is not None
    assert body["trend_samples"] == 0


# ── Rehearsal CLI ────────────────────────────────────────────────────────────


def test_rehearse_cli_prints_a_verdict(capsys):
    rc = rb.main(
        [
            "rehearse",
            "--peak-rss-kb",
            str(15 * 1024 * 1024),
            "--mem-total-kb",
            str(16 * 1024 * 1024),
        ]
    )
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["status"] == rb.ENVELOPE_ERROR


def test_selftest_returns_the_bodys_exit_code(capsys):
    assert rb.main(["selftest"]) == 0
    capsys.readouterr()


def _reading(*, headroom, total_kb=16 * 1024 * 1024, instance_type="m5.xlarge"):
    return {
        "measured": True,
        "peak_rss_kb": int(total_kb * (1.0 - headroom)),
        "mem_total_kb": total_kb,
        "instance_type": instance_type,
    }
