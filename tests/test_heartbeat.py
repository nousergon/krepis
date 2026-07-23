"""
Tests for ``krepis.heartbeat`` — §116 rule 5 heartbeat cadence chokepoint.

Pins the institutional contract:
* :func:`emit_heartbeat` produces a structured ``HEARTBEAT [{slug}] alive at ...`` message.
* The interval constant is 300s by default.
* :func:`emit_heartbeat_if_elapsed` respects the interval and returns the
  updated ``last_heartbeat_at``.
* The CLI loops until killed.
"""

from __future__ import annotations

import re
import sys

import pytest

from krepis import heartbeat


class TestConstants:
    def test_default_interval_is_300(self):
        assert heartbeat.HEARTBEAT_INTERVAL_S == 300


class TestEmitHeartbeat:
    def test_message_format(self, capsys):
        heartbeat.emit_heartbeat("spot-train", interval_s=300)
        captured = capsys.readouterr()
        assert "HEARTBEAT [spot-train]" in captured.err
        assert "alive at" in captured.err
        assert "interval=300s" in captured.err

    def test_includes_iso_timestamp(self, capsys):
        heartbeat.emit_heartbeat("test", interval_s=60)
        captured = capsys.readouterr()
        # Match ISO 8601 timestamp like 2026-07-23T12:00:00+00:00
        assert re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", captured.err)

    def test_different_slug_appears(self, capsys):
        heartbeat.emit_heartbeat("my-custom-phase", interval_s=120)
        captured = capsys.readouterr()
        assert "HEARTBEAT [my-custom-phase]" in captured.err

    def test_custom_interval_in_message(self, capsys):
        heartbeat.emit_heartbeat("x", interval_s=600)
        captured = capsys.readouterr()
        assert "interval=600s" in captured.err


class TestEmitHeartbeatIfElapsed:
    def test_emits_on_first_call(self, capsys):
        result = heartbeat.emit_heartbeat_if_elapsed("test", interval_s=60)
        captured = capsys.readouterr()
        assert "HEARTBEAT [test]" in captured.err
        assert result > 0  # returns the monotonic time

    def test_does_not_emit_before_interval(self, capsys):
        # First call (always emits)
        result = heartbeat.emit_heartbeat_if_elapsed("test", interval_s=60)
        captured = capsys.readouterr()
        assert "HEARTBEAT" in captured.err

        # Second call immediately after — should NOT emit
        result2 = heartbeat.emit_heartbeat_if_elapsed(
            "test", interval_s=60, last_heartbeat_at=result, now=result
        )
        captured2 = capsys.readouterr()
        assert captured2.err == ""  # no new output
        assert result2 == result  # unchanged

    def test_emits_after_interval(self, capsys):
        past = 0.0
        future = 100.0  # well past the 30s interval

        # First call (always emits)
        heartbeat.emit_heartbeat_if_elapsed(
            "test", interval_s=30, last_heartbeat_at=past, now=future
        )
        captured = capsys.readouterr()
        assert "HEARTBEAT [test]" in captured.err

    def test_returns_updated_time_on_emit(self):
        now = 1234.0
        result = heartbeat.emit_heartbeat_if_elapsed(
            "test", interval_s=60, last_heartbeat_at=1000.0, now=now
        )
        assert result == now

    def test_returns_original_when_no_emit(self):
        last = 1000.0
        now = 1020.0  # only 20s elapsed, interval is 60s
        result = heartbeat.emit_heartbeat_if_elapsed(
            "test", interval_s=60, last_heartbeat_at=last, now=now
        )
        assert result == last  # unchanged, no emit

    def test_none_last_emits_always(self, capsys):
        result = heartbeat.emit_heartbeat_if_elapsed(
            "test", interval_s=60, last_heartbeat_at=None, now=500.0
        )
        captured = capsys.readouterr()
        assert "HEARTBEAT" in captured.err
        assert result == 500.0


class TestCli:
    def test_emit_subcommand_runs_until_killed(self):
        """The CLI starts a heartbeat loop. We can't test the infinite
        loop directly, but we can verify the subcommand parses correctly
        and the module entrypoint works."""
        import subprocess
        import time

        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "krepis.heartbeat",
                "emit",
                "--slug",
                "test-loop",
                "--interval",
                "1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(1.5)
        proc.terminate()
        proc.wait(timeout=5)
        stderr = proc.stderr.read().decode("utf-8") if proc.stderr else ""
        assert "HEARTBEAT" in stderr
        assert "[test-loop]" in stderr

    def test_emit_subcommand_custom_interval(self):
        import subprocess
        import time

        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "krepis.heartbeat",
                "emit",
                "--slug",
                "fast-loop",
                "--interval",
                "1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(1.5)
        proc.terminate()
        proc.wait(timeout=5)
        stderr = proc.stderr.read().decode("utf-8") if proc.stderr else ""
        # Should have received multiple heartbeats in ~1.5s with interval=1
        assert stderr.count("HEARTBEAT") >= 1

    def test_missing_slug_errors(self):
        with pytest.raises(SystemExit):
            heartbeat.main(["emit"])

    def test_help_exits_clean(self):
        with pytest.raises(SystemExit) as exc:
            heartbeat.main(["--help"])
        assert exc.value.code == 0
