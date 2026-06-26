"""
Unit tests for ``krepis.ssm_dispatcher``.

Pins the institutional-chokepoint contract that the three alpha-engine
dispatcher scripts (alpha-engine-predictor ``spot_train.sh``,
alpha-engine-data ``spot_data_weekly.sh``,
alpha-engine-backtester ``spot_backtest.sh``) will rely on after the
2026-05-26 ``run_ssm`` bash-helper lift to lib:

* Success → exit 0; Failed / TimedOut / Cancelled / TerminalError →
  exit 1; unknown status → exit 1 (no silent skip per
  ``[[feedback_no_silent_fails]]``).
* StdOut streaming: deltas are written to the dispatcher's stdout as
  the inner command produces output; 24KB-cap rotation is detected and
  surfaced (without aborting the poll).
* StdErr is emitted to the dispatcher's stderr only on terminal
  non-Success, prefixed with the description so a multi-step log is
  greppable.
* InvocationDoesNotExist during the registration grace window
  (≤60s post-SendCommand) keeps polling; after the grace window it
  fails loud — closes the 2026-05-23 substrate weakness at the
  chokepoint level rather than per-SF JSON Retry block.
* Throttling exceptions during polling are transient (keep polling).
* The script body is base64-wrapped before transport so embedded
  heredocs, Python, and single quotes survive AWS-RunShellScript.
* The CLI accepts the script body from stdin or --script-file; empty
  body is refused with exit 2.
"""

from __future__ import annotations

import io
import sys
from unittest.mock import MagicMock

import pytest

from krepis import ssm_dispatcher


class _FakeClientError(Exception):
    """Stand-in for botocore.exceptions.ClientError carrying .response."""

    def __init__(self, code: str, message: str = ""):
        super().__init__(f"{code}: {message}")
        self.response = {"Error": {"Code": code, "Message": message}}


def _fake_ssm(*, send_resp=None, send_raises=None, poll_sequence=None,
              poll_raises_sequence=None):
    """Build a MagicMock ssm client driven by ``poll_sequence``.

    Args:
        send_resp: dict to return from send_command. Defaults to a
            CommandId.
        send_raises: exception to raise from send_command instead.
        poll_sequence: list of dicts returned by successive
            get_command_invocation calls.
        poll_raises_sequence: optional list of exceptions/Nones. None
            entries mean "return the next dict from poll_sequence";
            non-None means raise that exception on that call.
    """
    ssm = MagicMock()
    if send_raises is not None:
        ssm.send_command.side_effect = send_raises
    else:
        ssm.send_command.return_value = send_resp or {
            "Command": {"CommandId": "cmd-abc"}
        }
    if poll_raises_sequence:
        # Combine raises + returns so order is preserved
        def _next(*a, **k):
            try:
                exc = poll_raises_sequence.pop(0)
            except IndexError:
                exc = None
            if exc is not None:
                raise exc
            return (poll_sequence or []).pop(0)

        ssm.get_command_invocation.side_effect = _next
    elif poll_sequence is not None:
        ssm.get_command_invocation.side_effect = list(poll_sequence)
    return ssm


# ---------------------------------------------------------------------------
# Encode/transport
# ---------------------------------------------------------------------------


class TestEncodeCommandPayload:
    def test_wraps_with_base64_decode_pipe(self):
        wrapped = ssm_dispatcher._encode_command_payload("echo hello")
        assert wrapped.startswith("echo ")
        assert wrapped.endswith(" | base64 -d | bash")

    def test_round_trips_heredoc_with_quotes_unchanged(self):
        import base64

        script = "python3 - <<'PY'\nprint('hi')\nPY\n"
        wrapped = ssm_dispatcher._encode_command_payload(script)
        token = wrapped.split()[1]
        decoded = base64.b64decode(token).decode("utf-8")
        assert decoded == script

    def test_empty_script_still_encodes(self):
        wrapped = ssm_dispatcher._encode_command_payload("")
        # base64 of empty string is empty; the wrapper still emits the
        # full pipeline shape so any future caller behavior is consistent.
        assert "base64 -d | bash" in wrapped


# ---------------------------------------------------------------------------
# Happy path: Success on first/second poll
# ---------------------------------------------------------------------------


class TestRunSuccess:
    def test_success_first_poll_returns_zero(self):
        ssm = _fake_ssm(
            poll_sequence=[{"Status": "Success", "StandardOutputContent": "ok\n"}]
        )
        out = io.StringIO()
        err = io.StringIO()
        rc = ssm_dispatcher.run(
            "i-abc",
            "bootstrap",
            "echo ok",
            output_bucket="bkt",
            output_key_prefix="pfx",
            stdout_stream=out,
            stderr_stream=err,
            sleep=lambda s: None,
            boto3_client=ssm,
        )
        assert rc == 0
        assert "ok" in out.getvalue()
        # command-id line lands on stderr (operator-visible)
        assert "cmd-abc" in err.getvalue()

    def test_success_after_inprogress_streams_deltas(self):
        ssm = _fake_ssm(
            poll_sequence=[
                {"Status": "InProgress", "StandardOutputContent": "line1\n"},
                {
                    "Status": "InProgress",
                    "StandardOutputContent": "line1\nline2\n",
                },
                {
                    "Status": "Success",
                    "StandardOutputContent": "line1\nline2\nline3\n",
                },
            ]
        )
        out = io.StringIO()
        rc = ssm_dispatcher.run(
            "i-abc",
            "bootstrap",
            "x",
            stdout_stream=out,
            stderr_stream=io.StringIO(),
            sleep=lambda s: None,
            boto3_client=ssm,
        )
        assert rc == 0
        full = out.getvalue()
        assert full.count("line1") == 1  # delta-streamed, not repeated
        assert "line2" in full and "line3" in full
        # Streamed in order
        assert full.index("line1") < full.index("line2") < full.index("line3")

    def test_buffer_rotation_at_24kb_emits_cap_note(self):
        # Simulate buffer shrinking (cap hit + rotation)
        ssm = _fake_ssm(
            poll_sequence=[
                {"Status": "InProgress", "StandardOutputContent": "X" * 24000},
                {"Status": "InProgress", "StandardOutputContent": "later"},
                {
                    "Status": "Success",
                    "StandardOutputContent": "later",
                },
            ]
        )
        out = io.StringIO()
        err = io.StringIO()
        rc = ssm_dispatcher.run(
            "i",
            "ragingest",
            "x",
            output_bucket="b",
            output_key_prefix="p",
            stdout_stream=out,
            stderr_stream=err,
            sleep=lambda s: None,
            boto3_client=ssm,
        )
        assert rc == 0
        assert "24KB cap" in err.getvalue()
        assert "s3://b/p" in err.getvalue()

    def test_buffer_rotation_without_bucket_emits_configure_hint(self):
        ssm = _fake_ssm(
            poll_sequence=[
                {"Status": "InProgress", "StandardOutputContent": "X" * 24000},
                {"Status": "InProgress", "StandardOutputContent": "y"},
                {"Status": "Success", "StandardOutputContent": "y"},
            ]
        )
        err = io.StringIO()
        rc = ssm_dispatcher.run(
            "i",
            "ragingest",
            "x",
            stdout_stream=io.StringIO(),
            stderr_stream=err,
            sleep=lambda s: None,
            boto3_client=ssm,
        )
        assert rc == 0
        assert "configure --output-bucket" in err.getvalue()


# ---------------------------------------------------------------------------
# Terminal non-Success paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status", ["Failed", "TimedOut", "Cancelled", "Cancelling", "TerminalError"]
)
class TestRunTerminalFailures:
    def test_terminal_status_returns_one(self, status):
        ssm = _fake_ssm(
            poll_sequence=[
                {
                    "Status": status,
                    "StandardOutputContent": "",
                    "StandardErrorContent": "oops",
                }
            ]
        )
        err = io.StringIO()
        rc = ssm_dispatcher.run(
            "i",
            "boom",
            "x",
            stdout_stream=io.StringIO(),
            stderr_stream=err,
            sleep=lambda s: None,
            boto3_client=ssm,
        )
        assert rc == 1
        text = err.getvalue()
        assert "'boom'" in text
        assert status in text
        assert "oops" in text

    def test_terminal_without_stderr_still_returns_one(self, status):
        ssm = _fake_ssm(
            poll_sequence=[{"Status": status, "StandardOutputContent": ""}]
        )
        rc = ssm_dispatcher.run(
            "i",
            "boom",
            "x",
            stdout_stream=io.StringIO(),
            stderr_stream=io.StringIO(),
            sleep=lambda s: None,
            boto3_client=ssm,
        )
        assert rc == 1


class TestUnknownStatus:
    def test_unknown_status_fails_loud(self):
        # Per [[feedback_no_silent_fails]] — anything outside the
        # documented {Success | terminal-non-Success | pending} set
        # MUST surface, not get treated as Pending forever.
        ssm = _fake_ssm(
            poll_sequence=[{"Status": "Confused", "StandardOutputContent": ""}]
        )
        err = io.StringIO()
        rc = ssm_dispatcher.run(
            "i",
            "unknown",
            "x",
            stdout_stream=io.StringIO(),
            stderr_stream=err,
            sleep=lambda s: None,
            boto3_client=ssm,
        )
        assert rc == 1
        assert "unknown status" in err.getvalue()


# ---------------------------------------------------------------------------
# Registration race (InvocationDoesNotExist)
# ---------------------------------------------------------------------------


class TestInvocationDoesNotExistGrace:
    def test_within_grace_window_keeps_polling(self):
        # 2 IDNE in a row, then Success — must NOT fail.
        ssm = MagicMock()
        ssm.send_command.return_value = {"Command": {"CommandId": "cmd-x"}}

        polls = [
            _FakeClientError("InvocationDoesNotExist"),
            _FakeClientError("InvocationDoesNotExist"),
            {"Status": "Success", "StandardOutputContent": "ok"},
        ]

        def _next(*a, **k):
            v = polls.pop(0)
            if isinstance(v, Exception):
                raise v
            return v

        ssm.get_command_invocation.side_effect = _next

        # Fake monotonic clock — never exceeds the grace window
        t = [0.0]

        def _mono():
            return t[0]

        def _sleep(_):
            t[0] += 5.0

        rc = ssm_dispatcher.run(
            "i",
            "race",
            "x",
            stdout_stream=io.StringIO(),
            stderr_stream=io.StringIO(),
            sleep=_sleep,
            monotonic=_mono,
            boto3_client=ssm,
        )
        assert rc == 0
        # All three poll cycles consumed
        assert ssm.get_command_invocation.call_count == 3

    def test_past_grace_window_fails_loud(self):
        # IDNE persisting past 60s grace → exit 1 with named reason.
        ssm = MagicMock()
        ssm.send_command.return_value = {"Command": {"CommandId": "cmd-x"}}
        ssm.get_command_invocation.side_effect = _FakeClientError(
            "InvocationDoesNotExist"
        )

        t = [0.0]

        def _mono():
            return t[0]

        def _sleep(_):
            t[0] += 30.0  # 30s/poll → cross 60s at poll 3

        err = io.StringIO()
        rc = ssm_dispatcher.run(
            "i",
            "race",
            "x",
            stdout_stream=io.StringIO(),
            stderr_stream=err,
            sleep=_sleep,
            monotonic=_mono,
            boto3_client=ssm,
        )
        assert rc == 1
        assert "InvocationDoesNotExist" in err.getvalue()
        assert "never registered" in err.getvalue()


# ---------------------------------------------------------------------------
# Throttling: transient
# ---------------------------------------------------------------------------


class TestPollThrottling:
    def test_throttling_keeps_polling(self):
        ssm = MagicMock()
        ssm.send_command.return_value = {"Command": {"CommandId": "c"}}
        polls = [
            _FakeClientError("ThrottlingException"),
            _FakeClientError("RequestLimitExceeded"),
            {"Status": "Success", "StandardOutputContent": ""},
        ]

        def _next(*a, **k):
            v = polls.pop(0)
            if isinstance(v, Exception):
                raise v
            return v

        ssm.get_command_invocation.side_effect = _next
        rc = ssm_dispatcher.run(
            "i",
            "thr",
            "x",
            stdout_stream=io.StringIO(),
            stderr_stream=io.StringIO(),
            sleep=lambda s: None,
            boto3_client=ssm,
        )
        assert rc == 0

    def test_unexpected_poll_exception_fails_loud(self):
        ssm = MagicMock()
        ssm.send_command.return_value = {"Command": {"CommandId": "c"}}
        ssm.get_command_invocation.side_effect = RuntimeError("network gone")

        err = io.StringIO()
        rc = ssm_dispatcher.run(
            "i",
            "boom",
            "x",
            stdout_stream=io.StringIO(),
            stderr_stream=err,
            sleep=lambda s: None,
            boto3_client=ssm,
        )
        assert rc == 1
        assert "RuntimeError" in err.getvalue()
        assert "network gone" in err.getvalue()


# ---------------------------------------------------------------------------
# Send-command failures
# ---------------------------------------------------------------------------


class TestSendCommandFailure:
    def test_send_raises_returns_one_with_reason(self):
        ssm = _fake_ssm(send_raises=_FakeClientError("AccessDeniedException"))
        err = io.StringIO()
        rc = ssm_dispatcher.run(
            "i",
            "denied",
            "x",
            stdout_stream=io.StringIO(),
            stderr_stream=err,
            sleep=lambda s: None,
            boto3_client=ssm,
        )
        assert rc == 1
        assert "send_command failed" in err.getvalue()
        assert "AccessDeniedException" in err.getvalue() or "_FakeClientError" in err.getvalue()

    def test_send_returns_no_command_id_returns_one(self):
        ssm = _fake_ssm(send_resp={"Command": {}})
        err = io.StringIO()
        rc = ssm_dispatcher.run(
            "i",
            "weird",
            "x",
            stdout_stream=io.StringIO(),
            stderr_stream=err,
            sleep=lambda s: None,
            boto3_client=ssm,
        )
        assert rc == 1
        assert "no CommandId" in err.getvalue()


# ---------------------------------------------------------------------------
# SendCommand parameter shape
# ---------------------------------------------------------------------------


class TestSendCommandShape:
    def test_passes_base64_wrapped_payload(self):
        ssm = _fake_ssm(
            poll_sequence=[{"Status": "Success", "StandardOutputContent": ""}]
        )
        ssm_dispatcher.run(
            "i-123",
            "boot",
            "echo wow",
            timeout_seconds=900,
            output_bucket="bkt",
            output_key_prefix="pfx",
            region="us-east-1",
            stdout_stream=io.StringIO(),
            stderr_stream=io.StringIO(),
            sleep=lambda s: None,
            boto3_client=ssm,
        )
        assert ssm.send_command.call_count == 1
        kw = ssm.send_command.call_args.kwargs
        assert kw["InstanceIds"] == ["i-123"]
        assert kw["DocumentName"] == "AWS-RunShellScript"
        assert kw["TimeoutSeconds"] == 900
        assert kw["OutputS3BucketName"] == "bkt"
        assert kw["OutputS3KeyPrefix"] == "pfx"
        # Payload is base64-wrapped (the inner command body)
        cmds = kw["Parameters"]["commands"]
        assert len(cmds) == 1
        assert cmds[0].endswith(" | base64 -d | bash")

    def test_description_truncated_to_100_for_comment(self):
        ssm = _fake_ssm(
            poll_sequence=[{"Status": "Success", "StandardOutputContent": ""}]
        )
        long_desc = "x" * 250
        ssm_dispatcher.run(
            "i",
            long_desc,
            "echo",
            stdout_stream=io.StringIO(),
            stderr_stream=io.StringIO(),
            sleep=lambda s: None,
            boto3_client=ssm,
        )
        assert len(ssm.send_command.call_args.kwargs["Comment"]) == 100

    def test_omits_output_s3_when_not_configured(self):
        ssm = _fake_ssm(
            poll_sequence=[{"Status": "Success", "StandardOutputContent": ""}]
        )
        ssm_dispatcher.run(
            "i",
            "no-s3",
            "echo",
            stdout_stream=io.StringIO(),
            stderr_stream=io.StringIO(),
            sleep=lambda s: None,
            boto3_client=ssm,
        )
        kw = ssm.send_command.call_args.kwargs
        assert "OutputS3BucketName" not in kw
        assert "OutputS3KeyPrefix" not in kw


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli:
    def test_run_subcommand_reads_script_from_stdin(self, monkeypatch):
        captured: dict = {}

        def fake_run(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return 0

        monkeypatch.setattr(ssm_dispatcher, "run", fake_run)
        monkeypatch.setattr(sys, "stdin", io.StringIO("echo cli\n"))
        rc = ssm_dispatcher.main(
            [
                "run",
                "--instance-id",
                "i-1",
                "--description",
                "cli-test",
                "--timeout",
                "120",
                "--region",
                "us-east-1",
                "--script-stdin",
            ]
        )
        assert rc == 0
        assert captured["kwargs"]["instance_id"] == "i-1"
        assert captured["kwargs"]["script"] == "echo cli\n"
        assert captured["kwargs"]["timeout_seconds"] == 120

    def test_run_subcommand_reads_script_from_file(self, tmp_path, monkeypatch):
        script_file = tmp_path / "body.sh"
        script_file.write_text("echo from-file\n")

        captured: dict = {}

        def fake_run(*args, **kwargs):
            captured["script"] = kwargs["script"]
            return 0

        monkeypatch.setattr(ssm_dispatcher, "run", fake_run)
        rc = ssm_dispatcher.main(
            [
                "run",
                "--instance-id",
                "i-1",
                "--description",
                "cli-file",
                "--script-file",
                str(script_file),
            ]
        )
        assert rc == 0
        assert captured["script"] == "echo from-file\n"

    def test_empty_script_refused_with_exit_two(self, monkeypatch):
        monkeypatch.setattr(sys, "stdin", io.StringIO("   \n"))
        rc = ssm_dispatcher.main(
            [
                "run",
                "--instance-id",
                "i-1",
                "--description",
                "empty",
                "--script-stdin",
            ]
        )
        assert rc == 2

    def test_mutually_exclusive_script_source(self, tmp_path):
        script_file = tmp_path / "body.sh"
        script_file.write_text("x")
        with pytest.raises(SystemExit):
            ssm_dispatcher.main(
                [
                    "run",
                    "--instance-id",
                    "i",
                    "--description",
                    "x",
                    "--script-file",
                    str(script_file),
                    "--script-stdin",
                ]
            )

    def test_missing_script_source_errors(self):
        with pytest.raises(SystemExit):
            ssm_dispatcher.main(
                ["run", "--instance-id", "i", "--description", "x"]
            )

    def test_missing_subcommand_errors(self):
        with pytest.raises(SystemExit):
            ssm_dispatcher.main([])

    def test_help_exits_clean(self):
        with pytest.raises(SystemExit) as exc:
            ssm_dispatcher.main(["--help"])
        assert exc.value.code == 0


# ---------------------------------------------------------------------------
# Module entrypoint
# ---------------------------------------------------------------------------


class TestModuleEntrypoint:
    """The module is invokable as ``python -m krepis.ssm_dispatcher``."""

    def test_module_exposes_main(self):
        assert callable(ssm_dispatcher.main)

    def test_module_exposes_run(self):
        assert callable(ssm_dispatcher.run)


# ---------------------------------------------------------------------------
# L369 — diagnostics-write on terminal non-Success
# ---------------------------------------------------------------------------


class TestTailBytes:
    """``_tail_bytes`` bounds operator-facing stdout/stderr tails to a
    grep-friendly window — snaps to a line boundary so the operator
    isn't reading mid-token noise."""

    def test_short_text_returned_unchanged(self):
        assert ssm_dispatcher._tail_bytes("short\n") == "short\n"

    def test_tail_capped_at_max_bytes(self):
        text = "a" * 5000
        out = ssm_dispatcher._tail_bytes(text, max_bytes=1024)
        assert len(out.encode("utf-8")) <= 1024

    def test_tail_snaps_to_newline_boundary(self):
        # Build a payload where the byte cutoff lands mid-line; the helper
        # should drop the partial line so the tail starts cleanly.
        lines = [f"line-{i:04d}\n" for i in range(500)]
        text = "".join(lines)
        out = ssm_dispatcher._tail_bytes(text, max_bytes=200)
        assert "\n" not in out[:1]  # never starts on a bare newline
        # Every returned line is intact (no leading partial fragment)
        for piece in out.split("\n"):
            if piece:
                assert piece.startswith("line-"), (
                    f"tail returned a partial line {piece!r}"
                )

    def test_tail_returns_raw_when_no_newline_in_window(self):
        text = "x" * 5000
        out = ssm_dispatcher._tail_bytes(text, max_bytes=128)
        # All x's — no newline to snap to → raw byte-tail returned
        assert out == "x" * 128

    def test_default_cap_is_4kb(self):
        assert ssm_dispatcher.DIAGNOSTICS_TAIL_BYTES == 4 * 1024


class TestShipDiagnostics:
    """``_ship_diagnostics`` writes the failure-record JSON to S3.

    Failure-mode posture matches the sibling
    ``krepis.ssm_log_capture._ship_log_to_s3``: never raises,
    returns ``(ok, detail)``. The inner SSM exit code must always be
    preserved regardless of S3 outcome."""

    def _common_kwargs(self):
        return dict(
            bucket="alpha-engine-research",
            prefix="_spot_diagnostics/ae-data",
            status="Failed",
            command_id="cmd-abc",
            description="morning-enrich",
            exit_window_utc="2026-05-27T17:55:00+00:00",
            stdout_tail="last 4KB of stdout\n",
            stderr_tail="last 4KB of stderr\n",
            instance_id="i-09b539c844515d549",
        )

    def test_writes_json_under_date_keyed_path(self):
        s3 = MagicMock()
        err = io.StringIO()
        ok, detail = ssm_dispatcher._ship_diagnostics(
            **self._common_kwargs(),
            boto3_client=s3,
            stderr_stream=err,
        )
        assert ok is True
        assert detail.startswith("s3://alpha-engine-research/_spot_diagnostics/ae-data/")
        assert detail.endswith(".json")
        s3.put_object.assert_called_once()
        call = s3.put_object.call_args.kwargs
        assert call["Bucket"] == "alpha-engine-research"
        # Key shape: {prefix}/{YYYY-MM-DD}.json
        assert call["Key"].startswith("_spot_diagnostics/ae-data/")
        assert call["Key"].endswith(".json")
        assert call["ContentType"] == "application/json"
        # Body parses to the full payload schema
        import json as _json
        payload = _json.loads(call["Body"].decode("utf-8"))
        assert payload["status"] == "Failed"
        assert payload["command_id"] == "cmd-abc"
        assert payload["description"] == "morning-enrich"
        assert payload["exit_window_utc"] == "2026-05-27T17:55:00+00:00"
        assert payload["stdout_tail"] == "last 4KB of stdout\n"
        assert payload["stderr_tail"] == "last 4KB of stderr\n"
        assert payload["instance_id"] == "i-09b539c844515d549"

    def test_strips_trailing_slash_on_prefix(self):
        s3 = MagicMock()
        kwargs = self._common_kwargs()
        kwargs["prefix"] = "_spot_diagnostics/ae-data/"  # operator-passed
        ok, detail = ssm_dispatcher._ship_diagnostics(
            **kwargs, boto3_client=s3, stderr_stream=io.StringIO()
        )
        assert ok is True
        call = s3.put_object.call_args.kwargs
        # No double slash in the key
        assert "//" not in call["Key"], call["Key"]

    def test_s3_failure_is_swallowed_and_reported(self):
        """The whole point of best-effort: S3 outage cannot mask the inner exit."""
        s3 = MagicMock()
        s3.put_object.side_effect = RuntimeError("NoCredentialsError: blah")
        err = io.StringIO()
        ok, detail = ssm_dispatcher._ship_diagnostics(
            **self._common_kwargs(),
            boto3_client=s3,
            stderr_stream=err,
        )
        assert ok is False
        assert "RuntimeError" in detail
        # Operator-visible note on stderr explaining the swallow
        captured = err.getvalue()
        assert "diagnostics-write" in captured
        assert "swallowed" in captured
        assert "inner exit preserved" in captured


class TestRunWritesDiagnosticsOnTerminalFailure:
    """End-to-end: a Failed terminal status triggers the diagnostics-write
    when both ``diagnostics_bucket`` AND ``diagnostics_prefix`` are set."""

    def test_failed_status_writes_diagnostics_and_returns_one(self):
        ssm = _fake_ssm(
            poll_sequence=[
                {
                    "Status": "Failed",
                    "StandardOutputContent": "started\n... lots ...\n",
                    "StandardErrorContent": "boom: something\n",
                }
            ]
        )
        s3 = MagicMock()
        rc = ssm_dispatcher.run(
            "i-018eb3307a21329bf",
            "morning-enrich",
            "echo started; exit 1",
            output_bucket="bkt",
            output_key_prefix="pfx",
            diagnostics_bucket="alpha-engine-research",
            diagnostics_prefix="_spot_diagnostics/ae-data",
            stdout_stream=io.StringIO(),
            stderr_stream=io.StringIO(),
            sleep=lambda s: None,
            boto3_client=ssm,
            s3_client=s3,
        )
        assert rc == 1, "Failed terminal status must propagate exit 1"
        s3.put_object.assert_called_once()
        call = s3.put_object.call_args.kwargs
        import json as _json
        payload = _json.loads(call["Body"].decode("utf-8"))
        assert payload["status"] == "Failed"
        assert payload["command_id"] == "cmd-abc"
        assert payload["description"] == "morning-enrich"
        assert payload["instance_id"] == "i-018eb3307a21329bf"
        assert "boom" in payload["stderr_tail"]
        assert "started" in payload["stdout_tail"]
        # exit_window_utc is an ISO-formatted UTC timestamp
        assert payload["exit_window_utc"].endswith("+00:00")

    def test_success_does_not_write_diagnostics(self):
        """Diagnostics are failure-only — success path never writes."""
        ssm = _fake_ssm(
            poll_sequence=[
                {"Status": "Success", "StandardOutputContent": "ok\n"}
            ]
        )
        s3 = MagicMock()
        rc = ssm_dispatcher.run(
            "i-abc",
            "bootstrap",
            "echo ok",
            diagnostics_bucket="alpha-engine-research",
            diagnostics_prefix="_spot_diagnostics/ae-data",
            stdout_stream=io.StringIO(),
            stderr_stream=io.StringIO(),
            sleep=lambda s: None,
            boto3_client=ssm,
            s3_client=s3,
        )
        assert rc == 0
        s3.put_object.assert_not_called()

    def test_failed_without_diagnostics_config_does_not_call_s3(self):
        """Both flags must be set to trigger the write — backward-compat."""
        ssm = _fake_ssm(
            poll_sequence=[
                {
                    "Status": "Failed",
                    "StandardOutputContent": "",
                    "StandardErrorContent": "fail\n",
                }
            ]
        )
        s3 = MagicMock()
        rc = ssm_dispatcher.run(
            "i-abc",
            "bootstrap",
            "exit 1",
            # diagnostics_bucket + diagnostics_prefix both None
            stdout_stream=io.StringIO(),
            stderr_stream=io.StringIO(),
            sleep=lambda s: None,
            boto3_client=ssm,
            s3_client=s3,
        )
        assert rc == 1
        s3.put_object.assert_not_called()

    def test_partial_diagnostics_config_does_not_write(self):
        """If only one of bucket/prefix is set, no write (require both)."""
        ssm = _fake_ssm(
            poll_sequence=[
                {
                    "Status": "Failed",
                    "StandardOutputContent": "",
                    "StandardErrorContent": "fail\n",
                }
            ]
        )
        s3 = MagicMock()
        rc = ssm_dispatcher.run(
            "i-abc",
            "bootstrap",
            "exit 1",
            diagnostics_bucket="alpha-engine-research",
            diagnostics_prefix=None,
            stdout_stream=io.StringIO(),
            stderr_stream=io.StringIO(),
            sleep=lambda s: None,
            boto3_client=ssm,
            s3_client=s3,
        )
        assert rc == 1
        s3.put_object.assert_not_called()

    def test_diagnostics_s3_failure_does_not_mask_inner_exit(self):
        """S3 outage on diagnostics-write must NOT promote 1 → 0."""
        ssm = _fake_ssm(
            poll_sequence=[
                {
                    "Status": "TimedOut",
                    "StandardOutputContent": "",
                    "StandardErrorContent": "timeout\n",
                }
            ]
        )
        s3 = MagicMock()
        s3.put_object.side_effect = RuntimeError("AccessDenied")
        err = io.StringIO()
        rc = ssm_dispatcher.run(
            "i-abc",
            "bootstrap",
            "sleep 9999",
            diagnostics_bucket="alpha-engine-research",
            diagnostics_prefix="_spot_diagnostics/ae-predictor",
            stdout_stream=io.StringIO(),
            stderr_stream=err,
            sleep=lambda s: None,
            boto3_client=ssm,
            s3_client=s3,
        )
        assert rc == 1, (
            "TimedOut + diagnostics-write S3 failure must still exit 1; "
            "swallowing S3 failure to promote 1→0 would be the worst possible "
            "behavior (silently turns a real failure into a green run)"
        )
        # Operator-visible record of the diagnostics-write failure
        assert "swallowed" in err.getvalue()

    def test_diagnostics_tail_truncates_large_payloads(self):
        """A 100KB stdout should be trimmed to ≤ DIAGNOSTICS_TAIL_BYTES."""
        big_stdout = "\n".join(f"line-{i:06d}" for i in range(10000)) + "\n"
        big_stderr = "\n".join(f"err-{i:06d}" for i in range(10000)) + "\n"
        ssm = _fake_ssm(
            poll_sequence=[
                {
                    "Status": "Failed",
                    "StandardOutputContent": big_stdout,
                    "StandardErrorContent": big_stderr,
                }
            ]
        )
        s3 = MagicMock()
        rc = ssm_dispatcher.run(
            "i-abc",
            "bootstrap",
            "echo big",
            diagnostics_bucket="alpha-engine-research",
            diagnostics_prefix="_spot_diagnostics/ae-data",
            stdout_stream=io.StringIO(),
            stderr_stream=io.StringIO(),
            sleep=lambda s: None,
            boto3_client=ssm,
            s3_client=s3,
        )
        assert rc == 1
        import json as _json
        payload = _json.loads(
            s3.put_object.call_args.kwargs["Body"].decode("utf-8")
        )
        # Both tails respect the configured cap
        assert len(payload["stdout_tail"].encode("utf-8")) <= ssm_dispatcher.DIAGNOSTICS_TAIL_BYTES
        assert len(payload["stderr_tail"].encode("utf-8")) <= ssm_dispatcher.DIAGNOSTICS_TAIL_BYTES
        # Both tails preserve the LATEST lines (operator wants failure tail,
        # not arbitrary middle slice)
        assert "line-009999" in payload["stdout_tail"]
        assert "err-009999" in payload["stderr_tail"]


class TestCliDiagnosticsFlags:
    """The two new CLI flags thread through to ``run()``."""

    def test_flags_default_to_none(self, monkeypatch, capsys):
        """Without --diagnostics-* flags, the run() kwargs default None."""
        captured = {}

        def _fake_run(*args, **kwargs):
            captured.update(kwargs)
            return 0

        monkeypatch.setattr(ssm_dispatcher, "run", _fake_run)
        monkeypatch.setattr(sys, "stdin", io.StringIO("echo ok\n"))
        rc = ssm_dispatcher.main(
            [
                "run",
                "--instance-id", "i-abc",
                "--description", "smoke",
                "--script-stdin",
            ]
        )
        assert rc == 0
        assert captured["diagnostics_bucket"] is None
        assert captured["diagnostics_prefix"] is None

    def test_flags_thread_through_to_run(self, monkeypatch):
        captured = {}

        def _fake_run(*args, **kwargs):
            captured.update(kwargs)
            return 0

        monkeypatch.setattr(ssm_dispatcher, "run", _fake_run)
        monkeypatch.setattr(sys, "stdin", io.StringIO("echo ok\n"))
        rc = ssm_dispatcher.main(
            [
                "run",
                "--instance-id", "i-abc",
                "--description", "smoke",
                "--diagnostics-bucket", "alpha-engine-research",
                "--diagnostics-prefix", "_spot_diagnostics/ae-data",
                "--script-stdin",
            ]
        )
        assert rc == 0
        assert captured["diagnostics_bucket"] == "alpha-engine-research"
        assert captured["diagnostics_prefix"] == "_spot_diagnostics/ae-data"
