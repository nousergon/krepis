"""Contract tests for the fleet's single resource-kill classifier.

Referenced by `nous-ergon-ops/governance/policy-clauses.d/sf-pipeline-policy.yaml`
clause `SFP-3-resource-kill-halts-and-is-named` as the `kind: external` check
that makes the clause assertable. The clause requires a resource kill be named
as OOM or TIMEOUT, with the stage, the limit and the observed value, in the
failure cause the operator reads.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from krepis import resource_kill


class TestClassifyFromReturnCode:
    @pytest.mark.parametrize("rc", [137, -9])
    def test_sigkill_codes_are_oom(self, rc):
        assert resource_kill.classify(returncode=rc) == resource_kill.OOM

    @pytest.mark.parametrize("rc", [124, 143, -14, -15])
    def test_timeout_codes_are_timeout(self, rc):
        assert resource_kill.classify(returncode=rc) == resource_kill.TIMEOUT

    @pytest.mark.parametrize("rc", [0, 1, 2, 127])
    def test_ordinary_failure_codes_are_not_resource_kills(self, rc):
        assert resource_kill.classify(returncode=rc) is None


class TestClassifyFromText:
    def test_bash_job_control_kill_line_is_oom(self):
        """The 2026-08-13 and 2026-08-15 shape, verbatim."""
        text = (
            "2026-08-15 12:42:50,552 INFO [backtest] feature_maps: loaded\n"
            "bash: line 16: 26748 Killed                  python -u backtest.py\n"
        )
        assert resource_kill.classify(text=text) == resource_kill.OOM

    def test_kill_line_wins_over_a_laundered_exit_code(self):
        """The exact I7442 failure: rc laundered to 1, kill line survives.

        A launcher's `if ! cmd; then echo "ERROR: X failed"; exit 1; fi`
        destroys the 137. Without the text signal the run classifies as an
        ordinary failure — which is what happened on 2026-08-15.
        """
        text = "ERROR: backtest.py failed\nbash: line 16: 26748 Killed  python\n"
        assert resource_kill.classify(returncode=1, text=text) == resource_kill.OOM

    def test_oom_killer_banner_is_oom(self):
        assert (
            resource_kill.classify(text="kernel: Out of memory: Killed process 1")
            == resource_kill.OOM
        )

    def test_watchdog_unit_is_timeout(self):
        assert (
            resource_kill.classify(text="Starting alpha-engine-watchdog ...")
            == resource_kill.TIMEOUT
        )

    def test_a_retried_http_timeout_is_not_a_resource_kill(self):
        """A false resource kill suppresses a legitimate relaunch (§3)."""
        text = (
            "WARNING [http_retry] GET https://example/x timed out; retrying (1/3)\n"
            "INFO [http_retry] GET https://example/x ok\n"
            "ERROR: collector failed: no rows returned\n"
        )
        assert resource_kill.classify(returncode=1, text=text) is None

    def test_kill_line_is_taken_from_the_tail_not_the_head(self):
        text = "bash: line 3: 11 Killed  early\nlater noise\nbash: line 9: 22 Killed  late\n"
        assert "late" in resource_kill.find_kill_line(text)

    def test_no_text_and_no_code_classifies_nothing(self):
        assert resource_kill.classify() is None
        assert resource_kill.find_kill_line("") is None
        assert resource_kill.find_kill_line(None) is None


class TestClassifyFromSsmStatus:
    def test_ssm_timedout_status_is_timeout(self):
        assert (
            resource_kill.classify(status="TimedOut", returncode=None)
            == resource_kill.TIMEOUT
        )

    def test_oom_wins_a_tie_with_a_timeout_signal(self):
        assert (
            resource_kill.classify(
                status="TimedOut", text="bash: line 1: 5 Killed  python"
            )
            == resource_kill.OOM
        )


class TestFormatCause:
    """sf-pipeline-policy §3 obligation 3 — the operator-readable form."""

    def test_names_classification_stage_limit_and_observed(self):
        cause = resource_kill.format_cause(
            classification=resource_kill.OOM,
            stage="predictor-backtest",
            returncode=137,
            kill_line="bash: line 16: 26748 Killed  python -u backtest.py",
            limit="instance-type=c5.large, executionTimeout=14400s",
        )
        assert cause.startswith("RESOURCE KILL (OOM)")
        assert "stage=predictor-backtest" in cause
        assert "limit=instance-type=c5.large" in cause
        assert "observed=" in cause
        assert "Killed" in cause

    def test_absent_values_are_rendered_not_omitted(self):
        """A missing field reads as a satisfied obligation; 'unknown' does not."""
        cause = resource_kill.format_cause(
            classification=resource_kill.TIMEOUT, stage="evaluator"
        )
        assert "limit=unknown" in cause
        assert "observed=unknown" in cause
        assert "rc=unknown" in cause

    def test_oom_observed_states_why_it_is_unavailable(self):
        cause = resource_kill.format_cause(
            classification=resource_kill.OOM, stage="s", returncode=137
        )
        assert "reports no peak RSS" in cause

    def test_missing_kill_line_says_so(self):
        cause = resource_kill.format_cause(
            classification=resource_kill.OOM, stage="s", returncode=137
        )
        assert "no kill line survived" in cause

    def test_a_runaway_kill_line_cannot_evict_the_fields_beside_it(self):
        cause = resource_kill.format_cause(
            classification=resource_kill.OOM,
            stage="s",
            returncode=137,
            kill_line="x" * 5000,
        )
        assert len(cause) < 1000
        assert "stage=s" in cause


class TestCli:
    def _run(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "krepis.resource_kill", "classify", *args],
            capture_output=True,
            text=True,
        )

    def test_exit_zero_and_cause_on_stdout_when_classified(self):
        proc = self._run("--stage", "backtester", "--rc", "137")
        assert proc.returncode == 0
        assert proc.stdout.startswith("RESOURCE KILL (OOM)")

    def test_exit_three_when_not_a_resource_kill(self):
        proc = self._run("--stage", "backtester", "--rc", "1")
        assert proc.returncode == 3
        assert proc.stdout.strip() == ""

    def test_scans_a_log_tail(self, tmp_path):
        log = tmp_path / "x.log"
        log.write_text("noise\n" * 100 + "bash: line 4: 9 Killed  python\n")
        proc = self._run("--stage", "s", "--rc", "1", "--log", str(log))
        assert proc.returncode == 0
        assert "OOM" in proc.stdout

    def test_json_shape(self, tmp_path):
        proc = self._run("--stage", "s", "--rc", "137", "--json")
        payload = json.loads(proc.stdout)
        assert payload["classification"] == "OOM"
        assert payload["stage"] == "s"
        assert payload["cause"].startswith("RESOURCE KILL (OOM)")

    def test_an_unreadable_log_does_not_fail_the_classification(self):
        proc = self._run("--stage", "s", "--rc", "137", "--log", "/nope/missing.log")
        assert proc.returncode == 0
        assert "OOM" in proc.stdout


class TestNoSecondDefinition:
    """policy-shared-code: one definition, and the old call sites delegate."""

    def test_ssm_log_capture_delegates(self):
        from krepis import ssm_log_capture

        assert ssm_log_capture._OOM_RETURNCODES is resource_kill.OOM_RETURNCODES
        assert ssm_log_capture._RESOURCE_KILL_RE is resource_kill.KILL_LINE_RE

    def test_ssm_dispatcher_delegates(self):
        from krepis import ssm_dispatcher

        assert (
            ssm_dispatcher._classify_terminal_failure("Failed", 137)
            == resource_kill.OOM
        )
        assert (
            ssm_dispatcher._classify_terminal_failure("TimedOut", None)
            == resource_kill.TIMEOUT
        )


class TestCliInProcess:
    """Same surface as TestCli, called in-process so it is measured."""

    def test_classified_prints_cause_and_returns_zero(self, capsys):
        rc = resource_kill.main(["classify", "--stage", "s", "--rc", "137"])
        assert rc == 0
        assert "RESOURCE KILL (OOM)" in capsys.readouterr().out

    def test_unclassified_returns_three(self, capsys):
        assert resource_kill.main(["classify", "--stage", "s", "--rc", "1"]) == 3
        assert capsys.readouterr().out == ""

    def test_json_unclassified_returns_three_with_null_cause(self, capsys):
        rc = resource_kill.main(["classify", "--stage", "s", "--rc", "1", "--json"])
        assert rc == 3
        payload = json.loads(capsys.readouterr().out)
        assert payload["classification"] is None and payload["cause"] is None

    def test_reads_stdin(self, capsys, monkeypatch):
        import io as _io

        monkeypatch.setattr(
            "sys.stdin", _io.StringIO("bash: line 1: 2 Killed  python\n")
        )
        rc = resource_kill.main(["classify", "--stage", "s", "--rc", "1", "--log", "-"])
        assert rc == 0
        assert "OOM" in capsys.readouterr().out

    def test_short_log_is_read_whole_without_seek_error(self, tmp_path, capsys):
        log = tmp_path / "s.log"
        log.write_text("bash: line 1: 2 Killed  python\n")
        rc = resource_kill.main(
            ["classify", "--stage", "s", "--rc", "1", "--log", str(log)]
        )
        assert rc == 0
        assert "OOM" in capsys.readouterr().out

    def test_unreadable_log_warns_and_still_classifies(self, capsys):
        rc = resource_kill.main(
            ["classify", "--stage", "s", "--rc", "137", "--log", "/nope/x.log"]
        )
        assert rc == 0
        captured = capsys.readouterr()
        assert "could not read" in captured.err
        assert "OOM" in captured.out
