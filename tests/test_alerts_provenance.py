"""
Environment-reachable dry-run + ad-hoc provenance (``metron-ops-I340``).

THE DEFECT THESE PIN. Twice measured, in two repos, by two mechanisms: to
prove a detector fires, somebody forced the detector's input condition true
against the LIVE transport, and the operator was paged for a condition that
did not exist (metron-ops-I340 on 2026-09-17, a CRITICAL compliance page whose
flag reads false on all 168 recorded runs of that detector; a deploy canary's
wiring probe on 2026-08-21, whose ``log.error`` reached flow-doctor). Nothing
in either delivered page distinguished it from a real detection.

``publish(dry_run=True)`` already existed and would have prevented both — for
a caller able to edit the call site. 216 non-test files across 17 repos call
``publish``. So the fix is at this layer and has two halves, and this file
pins both plus the invariants they must not disturb:

* the dry-run is reachable from the ENVIRONMENT, one-way;
* an alert no scheduled context produced SAYS SO, in the delivered text only,
  outside the dedup identity and outside the bus event.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from krepis import alerts


@pytest.fixture
def fake_boto3():
    """boto3 stub returning mocked SNS + STS clients keyed by service."""
    sts_client = MagicMock()
    sts_client.get_caller_identity.return_value = {"Account": "711398986525"}
    sns_client = MagicMock()
    sns_client.publish.return_value = {"MessageId": "test-msg-id-abc123"}

    # The delivery-tier registry read goes to S3 and is irrelevant here. Let it
    # FAIL, which is what it does in the rest of this suite: `alert_tiers`
    # catches it, logs, and fails UPWARD to `page` — deterministic, and the
    # loudest tier, so nothing about message shape is hidden by a quiet tier.
    s3_client = MagicMock()
    s3_client.get_object.side_effect = Exception("no S3 in tests")

    fake = MagicMock()

    def _client(service: str, **kwargs):
        if service == "sts":
            return sts_client
        if service == "sns":
            return sns_client
        if service == "s3":
            return s3_client
        raise AssertionError(f"unexpected boto3 client request: {service}")

    fake.client.side_effect = _client
    return fake, sts_client, sns_client


@pytest.fixture
def ad_hoc_env(monkeypatch):
    """Make the process look like a hand-run: no scheduled-context variable.

    Clears EVERY name in ``SCHEDULED_CONTEXT_ENV_VARS`` rather than the one the
    runner happens to set, so the outcome is the same on a laptop (none set)
    and in GitHub Actions (``GITHUB_ACTIONS`` and ``CI`` both set). A test that
    passes only on one of the two is grading the runner.

    RETURNS A CALLABLE THAT THE TEST BODY MUST INVOKE, and that is not a style
    choice: pytest re-sets ``PYTEST_CURRENT_TEST`` at the start of every phase,
    so a ``delenv`` performed during fixture SETUP is undone before the test
    body runs. Clearing it from the body is the only point at which it stays
    cleared. (Measured while writing these tests — nine of them passed
    vacuously with the suffix suppressed by the very variable they had just
    deleted.)
    """
    def _apply():
        for name in alerts.SCHEDULED_CONTEXT_ENV_VARS:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(alerts.DRY_RUN_ENV_VAR, raising=False)
        # The test-env guard is keyed on PYTEST_CURRENT_TEST, just cleared;
        # the opt-in is set anyway so the guard is off either way.
        monkeypatch.setenv("ALPHA_ENGINE_ALLOW_TEST_ALERTS", "1")

    _apply()
    return _apply


@pytest.fixture
def no_side_channels(monkeypatch):
    """Neutralise SSM mute lookup and the Overseer bus emission.

    Both are network legs irrelevant to message shape; leaving them live makes
    these tests measure the laptop's credentials.
    """
    monkeypatch.setattr(alerts, "_fetch_source_mutes", lambda *_a, **_k: [])
    monkeypatch.setattr(
        alerts.fleet_events, "emit_alert_event", MagicMock(return_value=True)
    )
    monkeypatch.setattr(alerts, "_check_dedup_marker", lambda *_a, **_k: (False, ""))
    monkeypatch.setattr(alerts, "_write_dedup_marker", lambda *_a, **_k: None)


def _publish(fake, **kwargs):
    """Run a full publish with mocked SNS + Telegram; return the SNS kwargs."""
    with patch.dict("sys.modules", {"boto3": fake}):
        with patch.object(
            alerts, "_publish_telegram",
            return_value=alerts.ChannelResult(ok=True, detail="sent"),
        ) as tg:
            result = alerts.publish(**kwargs)
    return result, tg


# ── 1. The environment-reachable dry-run ───────────────────────────────────


class TestEnvDryRunParsing:
    """The value set is CLOSED, and the closure is the safety property.

    A truthiness test would read ``KREPIS_ALERT_DRY_RUN=0`` as ON and suppress
    a real alert. Suppressing a real alert is strictly worse than failing to
    suppress a synthetic one, so anything not documented is OFF.
    """

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "True", "yes", "YES", " true "])
    def test_documented_values_are_on(self, monkeypatch, value):
        monkeypatch.setenv(alerts.DRY_RUN_ENV_VAR, value)
        assert alerts._env_dry_run() is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "", "off", "maybe", "2"])
    def test_everything_else_is_off(self, monkeypatch, value):
        monkeypatch.setenv(alerts.DRY_RUN_ENV_VAR, value)
        assert alerts._env_dry_run() is False

    def test_unset_is_off(self, monkeypatch):
        monkeypatch.delenv(alerts.DRY_RUN_ENV_VAR, raising=False)
        assert alerts._env_dry_run() is False


class TestEnvDryRunReachesPublish:
    def test_env_var_suppresses_a_publish_with_no_code_change(
        self, monkeypatch, ad_hoc_env, no_side_channels, fake_boto3
    ):
        """The whole point: a caller passing NOTHING gets the dry-run."""
        ad_hoc_env()
        fake, _sts, sns = fake_boto3
        monkeypatch.setenv(alerts.DRY_RUN_ENV_VAR, "1")
        result, tg = _publish(fake, message="boom", source="metron")
        assert result.sns.ok is True
        assert "dry-run: would send" in result.sns.detail
        sns.publish.assert_not_called()
        tg.assert_not_called()
        # No boto3 client at all — not even the STS call that resolves the ARN.
        fake.client.assert_not_called()

    def test_env_var_logs_a_warning_naming_itself(
        self, monkeypatch, ad_hoc_env, no_side_channels, fake_boto3, caplog
    ):
        """A suppressed publish that logged nothing is the same defect, lower.

        Without this line a dry-run and a delivered alert are indistinguishable
        afterwards from the emitting process's own output.
        """
        ad_hoc_env()
        fake, _sts, _sns = fake_boto3
        monkeypatch.setenv(alerts.DRY_RUN_ENV_VAR, "yes")
        with caplog.at_level("WARNING", logger="krepis.alerts"):
            _publish(fake, message="boom", source="metron")
        assert any(
            alerts.DRY_RUN_ENV_VAR in r.getMessage() for r in caplog.records
        ), "a suppressed publish must say so, or it is indistinguishable from a sent one"

    def test_explicit_dry_run_true_still_wins_with_env_unset(
        self, monkeypatch, ad_hoc_env, no_side_channels, fake_boto3
    ):
        ad_hoc_env()
        fake, _sts, sns = fake_boto3
        monkeypatch.delenv(alerts.DRY_RUN_ENV_VAR, raising=False)
        result, tg = _publish(fake, message="boom", source="metron", dry_run=True)
        assert "dry-run: would send" in result.sns.detail
        sns.publish.assert_not_called()
        tg.assert_not_called()

    @pytest.mark.parametrize("value", ["0", "false", "no", ""])
    def test_the_env_var_can_never_turn_a_dry_run_OFF(
        self, monkeypatch, ad_hoc_env, no_side_channels, fake_boto3, value
    ):
        """One-way, and the direction is load-bearing.

        If the variable could override a caller's ``dry_run=True``, then a
        stray export in a shell profile would turn every deliberately-silent
        call site into a real page — which is the 09-17 incident with an extra
        step.
        """
        ad_hoc_env()
        fake, _sts, sns = fake_boto3
        monkeypatch.setenv(alerts.DRY_RUN_ENV_VAR, value)
        result, tg = _publish(fake, message="boom", source="metron", dry_run=True)
        assert "dry-run: would send" in result.sns.detail
        sns.publish.assert_not_called()
        tg.assert_not_called()


# ── 2. Provenance: the normal path is untouched ────────────────────────────


_EXPECTED_TODAY = "[ERROR] metron: boom"


class TestScheduledContextIsByteIdentical:
    """A scheduled unit's alert must not change by one byte.

    This is the constraint that keeps the change off the normal path: every
    fleet alert that matters is emitted by a systemd timer, a GitHub Actions
    job or a Lambda, and none of them may acquire a new line.
    """

    @pytest.mark.parametrize("var", list(alerts.SCHEDULED_CONTEXT_ENV_VARS))
    def test_every_declared_scheduled_var_suppresses_the_suffix(
        self, monkeypatch, ad_hoc_env, var
    ):
        ad_hoc_env()
        monkeypatch.setenv(var, "8:12345")
        assert alerts._provenance_suffix() is None
        assert alerts._scheduled_context_var() == var

    def test_an_empty_value_does_not_count_as_set(self, monkeypatch, ad_hoc_env):
        """An exported-but-empty variable is what a shell profile leaves."""
        ad_hoc_env()
        monkeypatch.setenv("JOURNAL_STREAM", "")
        assert alerts._provenance_suffix() is not None

    def test_unit_context_sns_message_is_exactly_todays_bytes(
        self, monkeypatch, ad_hoc_env, no_side_channels, fake_boto3
    ):
        ad_hoc_env()
        fake, _sts, sns = fake_boto3
        monkeypatch.setenv("JOURNAL_STREAM", "8:12345")
        _publish(fake, message="boom", source="metron")
        assert sns.publish.call_args.kwargs["Message"] == _EXPECTED_TODAY

    def test_ci_context_sns_message_is_exactly_todays_bytes(
        self, monkeypatch, ad_hoc_env, no_side_channels, fake_boto3
    ):
        ad_hoc_env()
        fake, _sts, sns = fake_boto3
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        _publish(fake, message="boom", source="metron")
        assert sns.publish.call_args.kwargs["Message"] == _EXPECTED_TODAY

    def test_unit_context_telegram_text_is_exactly_todays_bytes(
        self, monkeypatch, ad_hoc_env, no_side_channels, fake_boto3
    ):
        ad_hoc_env()
        fake, _sts, _sns = fake_boto3
        monkeypatch.setenv("JOURNAL_STREAM", "8:12345")
        _result, tg = _publish(fake, message="boom", source="metron")
        assert tg.call_args.args[0] == _EXPECTED_TODAY


# ── 3. Provenance: the ad-hoc path identifies itself ───────────────────────


class TestAdHocPublishCarriesProvenance:
    def test_suffix_names_host_and_pid_on_both_channels(
        self, ad_hoc_env, no_side_channels, fake_boto3
    ):
        ad_hoc_env()
        import os
        import socket

        fake, _sts, sns = fake_boto3
        _result, tg = _publish(fake, message="boom", source="metron")
        sent = sns.publish.call_args.kwargs["Message"]
        assert sent.startswith(_EXPECTED_TODAY)
        assert alerts.PROVENANCE_PREFIX in sent
        assert f"pid={os.getpid()}" in sent
        assert socket.gethostname()[:20].split(".")[0] in sent
        # Both channels carry the same delivered text.
        assert tg.call_args.args[0] == sent

    def test_suffix_is_its_own_line(self, ad_hoc_env, no_side_channels, fake_boto3):
        """A suffix run onto the message body reads as part of the finding."""
        ad_hoc_env()
        fake, _sts, sns = fake_boto3
        _publish(fake, message="boom", source="metron")
        lines = sns.publish.call_args.kwargs["Message"].splitlines()
        assert lines[0] == _EXPECTED_TODAY
        assert lines[-1].startswith(alerts.PROVENANCE_PREFIX)

    def test_it_marks_and_never_suppresses(
        self, ad_hoc_env, no_side_channels, fake_boto3
    ):
        """A hand-run alert is still DELIVERED. Marked, not swallowed.

        Deciding that a human's alert is unwanted is a different and much worse
        failure than an unmarked one.
        """
        ad_hoc_env()
        fake, _sts, sns = fake_boto3
        result, tg = _publish(fake, message="boom", source="metron")
        sns.publish.assert_called_once()
        tg.assert_called_once()
        assert result.sns.ok is True
        assert result.telegram.ok is True
        assert result.any_ok is True

    def test_the_suffix_carries_no_instruction_the_fleet_lint_fails(
        self, ad_hoc_env
    ):
        """ALERT001/ALERT002 shape check, held locally.

        ``nousergon-data/infrastructure/overseer/alert_message_lint.py`` fails a
        message that asks the reader for a decision (ALERT001) or tells them to
        go inspect a named surface (ALERT002). krepis does not run that lint in
        its own CI, so the property is pinned HERE rather than assumed — this
        module's text ships into 17 repos that DO run it.
        """
        ad_hoc_env()
        suffix = alerts._provenance_suffix().lower()
        for asking in ("approve", "acknowledge", "confirm", "please", "decide whether",
                       "sign-off", "if this is intended", "if expected"):
            assert asking not in suffix
        for inspecting in ("review", "check", "inspect", "consult", "examine",
                           "look at", "investigate", "triage", "diagnose",
                           "ssh ", "journalctl", "see the log", "details in"):
            assert inspecting not in suffix

    def test_host_is_sanitised_to_the_dns_label_alphabet(self, ad_hoc_env, monkeypatch):
        """A hostname lands in a Telegram body and an SNS body.

        Nothing in it may be a newline (a second line pretending to be part of
        the alert) or Telegram markup. ``_`` and ``-`` and ``.`` survive because
        they are legal in real hostnames, and the Telegram transport escapes
        Markdown itself (:func:`krepis.telegram._escape_markdown`).
        """
        ad_hoc_env()
        monkeypatch.setattr(
            alerts.socket, "gethostname", lambda: "bad\nname*with_[markup]"
        )
        suffix = alerts._provenance_suffix()
        host = suffix.split("host=", 1)[1].split(" ", 1)[0]
        assert host == "badnamewith_markup"
        assert not set(host) & set("*[]`\n")
        # Exactly one newline: the one that starts the provenance line.
        assert suffix.count("\n") == 1

    def test_gethostname_failure_never_breaks_the_alert(self, ad_hoc_env, monkeypatch):
        """Diagnostic metadata must never be why somebody's alert did not go out."""
        ad_hoc_env()
        def _boom():
            raise OSError("no hostname")

        monkeypatch.setattr(alerts.socket, "gethostname", _boom)
        suffix = alerts._provenance_suffix()
        assert "<host-unavailable>" in suffix


# ── 4. The invariants the suffix must not disturb ──────────────────────────


class TestProvenanceIsOutsideDedupIdentity:
    """THE load-bearing invariant.

    If the provenance text entered the dedup hash, every hand-run would mint
    its own episode and defeat the dedup this library exists to provide — the
    fix for one paging defect would have created a louder one.
    """

    def test_marker_key_is_computed_from_dedup_key_alone(self):
        """Structural: the key function does not take the message at all."""
        import inspect

        params = inspect.signature(alerts._dedup_marker_key).parameters
        assert list(params) == ["dedup_key"]

    def test_same_dedup_key_yields_the_same_marker_key_with_and_without_suffix(
        self, monkeypatch, ad_hoc_env, no_side_channels, fake_boto3
    ):
        ad_hoc_env()
        fake, _sts, sns = fake_boto3
        seen = {}

        def _capture(bucket, marker_key, *, dedup_key, formatted_message):
            seen[dedup_key] = (marker_key, formatted_message)

        monkeypatch.setattr(alerts, "_write_dedup_marker", _capture)
        monkeypatch.setattr(alerts, "_check_dedup_marker", lambda *_a, **_k: (False, ""))

        # Ad-hoc: suffix present.
        _publish(fake, message="boom", source="metron", dedup_key="episode-42")
        ad_hoc_key, ad_hoc_msg = seen.pop("episode-42")

        # Scheduled unit: no suffix.
        monkeypatch.setenv("JOURNAL_STREAM", "8:12345")
        _publish(fake, message="boom", source="metron", dedup_key="episode-42")
        unit_key, unit_msg = seen.pop("episode-42")

        assert ad_hoc_key == unit_key, (
            "the provenance suffix changed the dedup marker key — one standing "
            "condition would mint a fresh episode per hand-run"
        )
        assert ad_hoc_msg != unit_msg, "the two deliveries should differ in TEXT"
        assert unit_msg == _EXPECTED_TODAY

    def test_dedup_suppression_still_fires_across_the_two_contexts(
        self, monkeypatch, ad_hoc_env, no_side_channels, fake_boto3
    ):
        """The behavioural half: a hand-run does NOT escape a live window."""
        ad_hoc_env()
        fake, _sts, sns = fake_boto3
        monkeypatch.setattr(
            alerts, "_check_dedup_marker",
            lambda *_a, **_k: (True, "published 3 minutes ago"),
        )
        result, _tg = _publish(
            fake, message="boom", source="metron", dedup_key="episode-42"
        )
        assert result.dedup_skipped is True
        sns.publish.assert_not_called()


class TestProvenanceChangesNothingElse:
    def test_identity_key_severity_and_state_are_untouched(
        self, ad_hoc_env, no_side_channels, fake_boto3
    ):
        ad_hoc_env()
        fake, _sts, _sns = fake_boto3
        result, _tg = _publish(
            fake, message="boom", source="metron", dedup_key="episode-42",
        )
        assert result.identity_key == "episode-42"
        assert result.state == alerts.ALERT_STATE_OPENED

    def test_the_bus_event_carries_the_callers_RAW_message(
        self, monkeypatch, ad_hoc_env, fake_boto3
    ):
        """The drain and the console pair on the event, not on the prose.

        ``body`` must stay exactly what the caller passed, or a consumer that
        fingerprints it sees a different alert from every host.
        """
        ad_hoc_env()
        fake, _sts, _sns = fake_boto3
        monkeypatch.setattr(alerts, "_fetch_source_mutes", lambda *_a, **_k: [])
        emit = MagicMock(return_value=True)
        monkeypatch.setattr(alerts.fleet_events, "emit_alert_event", emit)
        _publish(fake, message="boom", source="metron")
        assert emit.call_args.kwargs["body"] == "boom"

    def test_the_event_schema_gains_no_field(
        self, monkeypatch, ad_hoc_env, fake_boto3
    ):
        """Additive-only would still be a schema change; this adds nothing."""
        ad_hoc_env()
        fake, _sts, _sns = fake_boto3
        monkeypatch.setattr(alerts, "_fetch_source_mutes", lambda *_a, **_k: [])
        emit = MagicMock(return_value=True)
        monkeypatch.setattr(alerts.fleet_events, "emit_alert_event", emit)
        _publish(fake, message="boom", source="metron")
        assert set(emit.call_args.kwargs) == {
            "origin", "body", "severity_raw", "source", "dedup_key", "channels",
            "state", "identity_key", "delivery_tier", "alert_class",
            "registry_drift",
        }

    def test_format_message_itself_is_unchanged(self):
        """The formatter is a pure function and stays one — the suffix is
        applied by ``publish``, so every consumer that calls ``_format_message``
        (or asserts on it) sees exactly what it saw before."""
        assert alerts._format_message("boom", "error", "metron") == _EXPECTED_TODAY
