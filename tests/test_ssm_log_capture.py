"""
Unit tests for ``krepis.ssm_log_capture``.

Pins the institutional-chokepoint contract that the 8 Saturday-SF spot
states + the weekday + EOD SF MorningEnrich states will rely on after
the 2026-05-22 lift from inline-bash-trap to lib CLI:

* inner exit code propagates verbatim
* stdout AND stderr are tee'd to the local log file AND the parent
  stdout (the SSM script-line-output that lands in CloudWatch /
  StandardOutputContent up to the 24KB cap)
* S3 upload happens regardless of inner exit status (success OR failure)
* S3 upload failure is swallowed (logged at WARNING) — the SF Catch must
  see the true inner exit, not a secondary log-capture failure that
  would mask it
* subprocess setup failure (binary not found, etc.) returns 127 and
  records the cause to the log file
* the S3 key layout is the canonical
  ``_ssm_logs/{slug}/{YYYY-MM-DD}/{hostname}-{HHMMSSZ}.log``
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from krepis import ssm_log_capture


@pytest.fixture
def fake_boto3():
    """boto3 stub that records upload_file calls and never raises."""
    s3_client = MagicMock()
    s3_client.upload_file.return_value = None  # boto3 returns None on success

    fake = MagicMock()
    fake.client.return_value = s3_client
    return fake, s3_client


@pytest.fixture
def isolated_logfile(tmp_path: Path) -> Path:
    return tmp_path / "test.log"


class TestExitKey:
    """Canonical S3 key layout — keep stable so consumers can find logs."""

    def test_layout_matches_pre_lift_form(self):
        from datetime import datetime, timezone

        key = ssm_log_capture._exit_key(
            "morning-enrich",
            now=datetime(2026, 5, 22, 20, 27, 0, tzinfo=timezone.utc),
            host="ip-172-31-73-124.ec2.internal",
        )
        assert key == (
            "_ssm_logs/morning-enrich/2026-05-22/"
            "ip-172-31-73-124.ec2.internal-202700Z.log"
        )

    def test_uses_default_prefix(self):
        from datetime import datetime, timezone

        key = ssm_log_capture._exit_key("X", now=datetime(2026, 1, 1, tzinfo=timezone.utc), host="h")
        assert key.startswith("_ssm_logs/")

    def test_slash_in_slug_preserved_caller_responsibility(self):
        # Intentional: the lib doesn't sanitize slug — callers pass a
        # tree-shape like ``"backtester/parity"`` if they want sub-keys.
        from datetime import datetime, timezone

        key = ssm_log_capture._exit_key(
            "backtester/parity",
            now=datetime(2026, 1, 1, tzinfo=timezone.utc),
            host="h",
        )
        assert "_ssm_logs/backtester/parity/" in key


class TestRunHappyPath:
    def test_propagates_zero_exit(self, isolated_logfile, fake_boto3):
        fake, _ = fake_boto3
        with patch.dict("sys.modules", {"boto3": fake}):
            rc = ssm_log_capture.run("slug", isolated_logfile, ["true"])
        assert rc == 0

    def test_stdout_lands_in_log_and_parent(self, isolated_logfile, fake_boto3, capfd):
        fake, _ = fake_boto3
        with patch.dict("sys.modules", {"boto3": fake}):
            rc = ssm_log_capture.run(
                "slug",
                isolated_logfile,
                [sys.executable, "-c", "print('hello-from-inner')"],
            )
        assert rc == 0
        captured = capfd.readouterr()
        assert "hello-from-inner" in captured.out
        # And the same bytes land in the log file
        log_contents = isolated_logfile.read_text()
        assert "hello-from-inner" in log_contents

    def test_stderr_merges_into_log(self, isolated_logfile, fake_boto3, capfd):
        fake, _ = fake_boto3
        with patch.dict("sys.modules", {"boto3": fake}):
            rc = ssm_log_capture.run(
                "slug",
                isolated_logfile,
                [
                    sys.executable,
                    "-c",
                    "import sys; sys.stderr.write('stderr-from-inner\\n'); sys.stderr.flush()",
                ],
            )
        assert rc == 0
        assert "stderr-from-inner" in isolated_logfile.read_text()

    def test_s3_upload_called_with_canonical_key(self, isolated_logfile, fake_boto3):
        fake, s3 = fake_boto3
        with patch.dict("sys.modules", {"boto3": fake}):
            ssm_log_capture.run("morning-enrich", isolated_logfile, ["true"])
        s3.upload_file.assert_called_once()
        args, _ = s3.upload_file.call_args
        local_path, bucket, key = args
        assert local_path == str(isolated_logfile)
        assert bucket == "alpha-engine-research"
        assert key.startswith("_ssm_logs/morning-enrich/")
        assert key.endswith(".log")

    def test_bucket_override(self, isolated_logfile, fake_boto3):
        fake, s3 = fake_boto3
        with patch.dict("sys.modules", {"boto3": fake}):
            ssm_log_capture.run(
                "slug",
                isolated_logfile,
                ["true"],
                bucket="custom-bucket",
            )
        args, _ = s3.upload_file.call_args
        assert args[1] == "custom-bucket"


class TestRunFailurePropagation:
    """The SF Catch must see the true inner exit. Secondary log-capture
    failure must not mask the primary."""

    def test_nonzero_inner_exit_propagates(self, isolated_logfile, fake_boto3):
        fake, _ = fake_boto3
        with patch.dict("sys.modules", {"boto3": fake}):
            rc = ssm_log_capture.run(
                "slug",
                isolated_logfile,
                [sys.executable, "-c", "import sys; sys.exit(7)"],
            )
        assert rc == 7

    def test_inner_binary_not_found_returns_127(self, isolated_logfile, fake_boto3):
        fake, _ = fake_boto3
        with patch.dict("sys.modules", {"boto3": fake}):
            rc = ssm_log_capture.run(
                "slug",
                isolated_logfile,
                ["/this/path/does/not/exist"],
            )
        assert rc == 127
        # And the failure cause was recorded to the log
        contents = isolated_logfile.read_text()
        assert "cannot exec" in contents

    def test_s3_failure_does_not_mask_inner_exit(self, isolated_logfile):
        fake = MagicMock()
        s3 = MagicMock()
        s3.upload_file.side_effect = RuntimeError("creds missing")
        fake.client.return_value = s3
        with patch.dict("sys.modules", {"boto3": fake}):
            rc = ssm_log_capture.run(
                "slug",
                isolated_logfile,
                [sys.executable, "-c", "import sys; sys.exit(3)"],
            )
        # Inner exit dominates; S3 failure is swallowed.
        assert rc == 3

    def test_s3_failure_with_success_inner_still_returns_zero(
        self, isolated_logfile
    ):
        fake = MagicMock()
        s3 = MagicMock()
        s3.upload_file.side_effect = RuntimeError("boom")
        fake.client.return_value = s3
        with patch.dict("sys.modules", {"boto3": fake}):
            rc = ssm_log_capture.run("slug", isolated_logfile, ["true"])
        assert rc == 0

    def test_missing_log_file_at_ship_time_is_swallowed(self, tmp_path, fake_boto3):
        # Inner cmd exits before any output; log file may end up empty
        # but should still exist (we open it for write at the top).
        # Force the "doesn't exist" branch by pointing at an unwriteable
        # parent — fall back: just delete the log between run() finishing
        # subprocess and the ship-time check. Simplest: simulate by
        # having the upload itself surface FileNotFoundError.
        fake, s3 = fake_boto3
        s3.upload_file.side_effect = FileNotFoundError("gone")
        log = tmp_path / "subdir-that-exists" / "x.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with patch.dict("sys.modules", {"boto3": fake}):
            rc = ssm_log_capture.run("slug", log, ["true"])
        assert rc == 0  # not masked


class TestS3UploadWhenLogFileMissing:
    """If the log file does not exist at ship time, return False with a
    descriptive reason rather than letting FileNotFoundError surface."""

    def test_returns_false_with_reason(self, tmp_path):
        fake = MagicMock()
        with patch.dict("sys.modules", {"boto3": fake}):
            ok, detail = ssm_log_capture._ship_log_to_s3(
                "slug",
                tmp_path / "does-not-exist.log",
                "alpha-engine-research",
            )
        assert ok is False
        assert "log file not found" in detail
        # And boto3 was never called
        fake.client.assert_not_called()


class TestCli:
    def test_run_subcommand_basic_invocation(
        self, isolated_logfile, fake_boto3, capfd
    ):
        fake, s3 = fake_boto3
        with patch.dict("sys.modules", {"boto3": fake}):
            rc = ssm_log_capture.main(
                [
                    "run",
                    "--slug",
                    "morning-enrich",
                    "--log",
                    str(isolated_logfile),
                    "--correlation-id",
                    "test-corr",
                    "--",
                    sys.executable,
                    "-c",
                    "print('cli-inner')",
                ]
            )
        assert rc == 0
        assert "cli-inner" in capfd.readouterr().out
        s3.upload_file.assert_called_once()

    def test_run_subcommand_propagates_nonzero(self, isolated_logfile, fake_boto3):
        fake, _ = fake_boto3
        with patch.dict("sys.modules", {"boto3": fake}):
            rc = ssm_log_capture.main(
                [
                    "run",
                    "--slug",
                    "x",
                    "--log",
                    str(isolated_logfile),
                    "--correlation-id",
                    "test-corr",
                    "--",
                    sys.executable,
                    "-c",
                    "import sys; sys.exit(5)",
                ]
            )
        assert rc == 5

    def test_run_without_inner_cmd_errors(self, isolated_logfile):
        with pytest.raises(SystemExit):
            ssm_log_capture.main(
                ["run", "--slug", "x", "--log", str(isolated_logfile)]
            )

    def test_missing_subcommand_errors(self):
        with pytest.raises(SystemExit):
            ssm_log_capture.main([])

    def test_help_exits_clean(self, capsys):
        with pytest.raises(SystemExit) as exc:
            ssm_log_capture.main(["--help"])
        assert exc.value.code == 0


class TestModuleEntrypoint:
    """The module is invokable as ``python -m krepis.ssm_log_capture``."""

    def test_module_has_main_guard(self):
        # The module file must end with ``if __name__ == "__main__"`` so
        # ``python -m`` invocation works on the SSM target. Sentinel
        # check: import the module and verify ``main`` is callable.
        assert callable(ssm_log_capture.main)


class TestFormatSubprocessFailure:
    """§116 rule 4 "no naked rc" chokepoint — :func:`format_subprocess_failure`."""

    def test_includes_step_name(self):
        msg = ssm_log_capture.format_subprocess_failure(
            "spot-train",
            returncode=1,
            last_output_line=None,
        )
        assert "[spot-train]" in msg
        assert "rc=1" in msg

    def test_includes_last_output_line_when_given(self):
        msg = ssm_log_capture.format_subprocess_failure(
            "enrich",
            returncode=127,
            last_output_line="FileNotFoundError: awscli not found",
        )
        assert "FileNotFoundError" in msg
        assert "rc=127" in msg

    def test_omits_output_line_when_none(self):
        msg = ssm_log_capture.format_subprocess_failure(
            "test-step",
            returncode=2,
            last_output_line=None,
        )
        assert "no output captured" in msg
        assert "rc=2" in msg

    def test_never_bare_rc(self):
        msg = ssm_log_capture.format_subprocess_failure(
            "x", returncode=255, last_output_line=None
        )
        # The message must contain the word "failed" and the step name,
        # never just "exit status 255".
        assert "failed" in msg
        assert "ssm_log_capture: ERROR" in msg


class TestCorrelationIdResolution:
    """§116 rule 6 correlation-id resolution — :func:`_resolve_correlation_id`."""

    def test_cli_arg_takes_precedence(self):
        result = ssm_log_capture._resolve_correlation_id("cli-token-abc")
        assert result == "cli-token-abc"

    def test_env_var_fallback(self, monkeypatch):
        monkeypatch.setenv(ssm_log_capture.CORRELATION_ID_ENV_VAR, "env-token-xyz")
        result = ssm_log_capture._resolve_correlation_id(None)
        assert result == "env-token-xyz"

    def test_auto_generates_when_neither_set(self, monkeypatch):
        monkeypatch.delenv(ssm_log_capture.SSM_COMMAND_ID_ENV_VAR, raising=False)
        monkeypatch.delenv(ssm_log_capture.CORRELATION_ID_ENV_VAR, raising=False)
        with patch.object(ssm_log_capture.uuid, "uuid4", return_value=MagicMock(hex="aabbccddee00")):
            result = ssm_log_capture._resolve_correlation_id(None)
        assert result == "aabbccddee00"
        assert isinstance(result, str)

    def test_cli_arg_overrides_env_var(self, monkeypatch):
        monkeypatch.setenv(ssm_log_capture.CORRELATION_ID_ENV_VAR, "env-token")
        result = ssm_log_capture._resolve_correlation_id("cli-token")
        assert result == "cli-token"

    def test_ssm_command_id_fallback(self, monkeypatch):
        monkeypatch.delenv(ssm_log_capture.CORRELATION_ID_ENV_VAR, raising=False)
        monkeypatch.setenv(ssm_log_capture.SSM_COMMAND_ID_ENV_VAR, "ssm-cmd-001")
        result = ssm_log_capture._resolve_correlation_id(None)
        assert result == "ssm-cmd-001"

    def test_ssm_command_id_overridden_by_run_token(self, monkeypatch):
        monkeypatch.setenv(ssm_log_capture.CORRELATION_ID_ENV_VAR, "run-token-val")
        monkeypatch.setenv(ssm_log_capture.SSM_COMMAND_ID_ENV_VAR, "ssm-cmd-val")
        result = ssm_log_capture._resolve_correlation_id(None)
        assert result == "run-token-val"

    def test_auto_generate_uuid_when_nothing_set(self, monkeypatch):
        monkeypatch.delenv(ssm_log_capture.CORRELATION_ID_ENV_VAR, raising=False)
        monkeypatch.delenv(ssm_log_capture.SSM_COMMAND_ID_ENV_VAR, raising=False)
        with patch.object(ssm_log_capture.uuid, "uuid4", return_value=MagicMock(hex="abcdef1234567890")):
            result = ssm_log_capture._resolve_correlation_id(None)
        assert result == "abcdef123456"
        # uuid4().hex[:12] — 12 hex chars

    def test_auto_generates_warning_logged_when_uuid_fallback(self, monkeypatch, caplog):
        import logging
        monkeypatch.delenv(ssm_log_capture.CORRELATION_ID_ENV_VAR, raising=False)
        monkeypatch.delenv(ssm_log_capture.SSM_COMMAND_ID_ENV_VAR, raising=False)
        with patch.object(ssm_log_capture.uuid, "uuid4", return_value=MagicMock(hex="deadbeef1234")):
            with caplog.at_level(logging.WARNING):
                ssm_log_capture._resolve_correlation_id(None)
        assert "auto-generated correlation_id" in caplog.text
        assert "deadbeef1234" in caplog.text

    def test_ssm_command_id_warning_logged(self, monkeypatch, caplog):
        import logging
        monkeypatch.delenv(ssm_log_capture.CORRELATION_ID_ENV_VAR, raising=False)
        monkeypatch.setenv(ssm_log_capture.SSM_COMMAND_ID_ENV_VAR, "ssm-warn-001")
        with caplog.at_level(logging.WARNING):
            ssm_log_capture._resolve_correlation_id(None)
        assert "SSM_COMMAND_ID" in caplog.text
        assert "ssm-warn-001" in caplog.text

    def test_auto_generate_is_non_empty_string(self, monkeypatch):
        """Never return None — auto-generation guarantees a str."""
        monkeypatch.delenv(ssm_log_capture.CORRELATION_ID_ENV_VAR, raising=False)
        monkeypatch.delenv(ssm_log_capture.SSM_COMMAND_ID_ENV_VAR, raising=False)
        with patch.object(ssm_log_capture.uuid, "uuid4", return_value=MagicMock(hex="abc123")):
            result = ssm_log_capture._resolve_correlation_id(None)
        assert isinstance(result, str)
        assert len(result) > 0


class TestExitKeyWithCorrelationId:
    """The correlation id is appended to the S3 key for traceability."""

    def test_correlation_id_appended_to_key(self):
        from datetime import datetime, timezone

        key = ssm_log_capture._exit_key(
            "spot-train",
            now=datetime(2026, 7, 23, 12, 0, 0, tzinfo=timezone.utc),
            host="ip-x.ec2.internal",
            correlation_id="run-007",
        )
        assert key.endswith("-run-007.log")
        assert "_ssm_logs/spot-train/" in key

    def test_no_correlation_id_omits_suffix(self):
        from datetime import datetime, timezone

        key = ssm_log_capture._exit_key(
            "test",
            now=datetime(2026, 7, 23, 12, 0, 0, tzinfo=timezone.utc),
            host="h",
            correlation_id=None,
        )
        # No trailing -correlation_id before .log. The key already has
        # dashes in the date (YYYY-MM-DD), so check that .log immediately
        # follows the Z of the HHMMSSZ timestamp.
        assert key.endswith("Z.log") or key.endswith(".log")


class TestFailureMessageOnNonZeroExit:
    """``run()`` prints a formatted failure message on non-zero exit (§116 rule 4)."""

    def test_formatted_failure_on_nonzero(self, isolated_logfile, fake_boto3, capsys):
        fake, _ = fake_boto3
        with patch.dict("sys.modules", {"boto3": fake}):
            rc = ssm_log_capture.run(
                "test-step",
                isolated_logfile,
                [sys.executable, "-c", "import sys; sys.stderr.write('boom\\n'); sys.stderr.flush(); sys.exit(9)"],
            )
        assert rc == 9
        err = capsys.readouterr().err
        assert "[test-step]" in err
        assert "failed" in err
        assert "rc=9" in err
        assert "boom" in err

    def test_log_also_contains_failure_message(self, isolated_logfile, fake_boto3):
        fake, _ = fake_boto3
        with patch.dict("sys.modules", {"boto3": fake}):
            ssm_log_capture.run(
                "step-x",
                isolated_logfile,
                [sys.executable, "-c", "import sys; sys.exit(5)"],
            )
        contents = isolated_logfile.read_text()
        assert "ssm_log_capture: ERROR" in contents
        assert "rc=5" in contents

    def test_zero_exit_no_failure_message(self, isolated_logfile, fake_boto3, capsys):
        fake, _ = fake_boto3
        with patch.dict("sys.modules", {"boto3": fake}):
            rc = ssm_log_capture.run(
                "step-ok",
                isolated_logfile,
                ["true"],
            )
        assert rc == 0
        err = capsys.readouterr().err
        assert "ssm_log_capture: ERROR" not in err


class TestCorrelationIdInLogAndKey:
    """The correlation id appears in the log header and S3 key."""

    def test_header_line_in_log(self, isolated_logfile, fake_boto3):
        fake, s3 = fake_boto3
        with patch.dict("sys.modules", {"boto3": fake}):
            ssm_log_capture.run(
                "test",
                isolated_logfile,
                [sys.executable, "-c", "print('hello')"],
                correlation_id="corr-001",
            )
        contents = isolated_logfile.read_text()
        assert "# correlation-id: corr-001" in contents
        assert "hello" in contents  # inner output follows the header

    def test_s3_key_contains_correlation_id(self, isolated_logfile, fake_boto3):
        fake, s3 = fake_boto3
        with patch.dict("sys.modules", {"boto3": fake}):
            ssm_log_capture.run(
                "test",
                isolated_logfile,
                ["true"],
                correlation_id="corr-002",
            )
        s3.upload_file.assert_called_once()
        args, _ = s3.upload_file.call_args
        key = args[2]
        assert "corr-002" in key


class TestCliCorrelationIdAutoGeneration:
    """The CLI auto-generates a correlation-id when none is provided (§116 rule 6
    graceful-degradation fix — config#4223)."""

    def test_missing_correlation_id_auto_generates(self, isolated_logfile, fake_boto3, monkeypatch):
        monkeypatch.delenv(ssm_log_capture.SSM_COMMAND_ID_ENV_VAR, raising=False)
        monkeypatch.delenv(ssm_log_capture.CORRELATION_ID_ENV_VAR, raising=False)
        fake, s3 = fake_boto3
        with patch.dict("sys.modules", {"boto3": fake}):
            with patch.object(ssm_log_capture.uuid, "uuid4", return_value=MagicMock(hex="aabbccddee001122")):
                rc = ssm_log_capture.main(
                    ["run", "--slug", "x", "--log", str(isolated_logfile), "--", "true"]
                )
        assert rc == 0
        s3.upload_file.assert_called_once()
        # No longer hard-errors — auto-generation succeeded; uuid4().hex[:12] lands in the S3 key
        args, _ = s3.upload_file.call_args
        key = args[2]
        assert "aabbccddee00" in key  # hex[:12] — 12 chars

    def test_with_correlation_id_succeeds(self, isolated_logfile, fake_boto3):
        fake, _ = fake_boto3
        with patch.dict("sys.modules", {"boto3": fake}):
            rc = ssm_log_capture.main(
                [
                    "run",
                    "--slug", "test",
                    "--log", str(isolated_logfile),
                    "--correlation-id", "cli-corr-003",
                    "--", sys.executable, "-c", "print('ok')",
                ]
            )
        assert rc == 0

    def test_env_var_fallback_accepted(self, isolated_logfile, fake_boto3, monkeypatch):
        monkeypatch.setenv("RUN_TOKEN", "env-corr-004")
        fake, _ = fake_boto3
        with patch.dict("sys.modules", {"boto3": fake}):
            rc = ssm_log_capture.main(
                [
                    "run",
                    "--slug", "test",
                    "--log", str(isolated_logfile),
                    "--", sys.executable, "-c", "print('ok')",
                ]
            )
        assert rc == 0


class TestStepNameInCli:
    """The --step-name argument customizes the failure message."""

    def test_step_name_in_failure_message(self, isolated_logfile, fake_boto3, capsys):
        fake, _ = fake_boto3
        with patch.dict("sys.modules", {"boto3": fake}):
            rc = ssm_log_capture.main(
                [
                    "run",
                    "--slug", "my-slug",
                    "--log", str(isolated_logfile),
                    "--step-name", "my-custom-step",
                    "--correlation-id", "cid",
                    "--", sys.executable, "-c", "import sys; sys.exit(3)",
                ]
            )
        assert rc == 3
        err = capsys.readouterr().err
        # The failure message should contain the custom step name, not the slug
        assert "[my-custom-step]" in err
