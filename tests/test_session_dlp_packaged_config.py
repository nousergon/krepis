"""The DLP ruleset ships INSIDE krepis, so no execution context can lack it.

Why this file exists
--------------------
``krepis.session_dlp`` is the in-process DLP tier — the only outbound-content
control on any path that cannot reach the localhost egress proxy (Lambda,
ephemeral EC2 spot boxes, CI runners; ``alpha-engine-config-I4927``). It is
fail-closed by design, so an absent ruleset does not degrade the control, it
stops every outbound LLM call in that context.

Until 2026-09-09 the ruleset was resolved only from ``$KREPIS_GITLEAKS_DIR`` or
an ``/opt/*-llm-routing`` directory an operator had to provision per substrate.
Four substrates were filed as missing it and none were fixed
(``alpha-engine-config-I7913`` laptop, ``-I7719`` CI runner, ``-I9972``
crucible-v2 spot box, ``-I9407`` backtester tests); a fifth — the
data-collector flow-doctor diagnosis path, on ephemeral EC2 data boxes — ran
broken for 27 days with the failure text visible only inside alert bodies.

These tests hold the property that makes that class unrepeatable: the ruleset
is package data, it is reachable through the installed package, and the
resolver ends at it rather than at a path that may not exist.
"""

from __future__ import annotations

import os

import pytest

import krepis.session_dlp as session_dlp


class TestPackagedRuleset:
    def test_packaged_dir_resolves_and_holds_the_entry_config(self):
        d = session_dlp.packaged_gitleaks_dir()
        assert d is not None, (
            "the gitleaks ruleset is not readable through the installed "
            "krepis package — package-data declaration lost?"
        )
        assert os.path.isfile(os.path.join(d, "gitleaks-egress.toml"))

    def test_extend_target_is_a_sibling_of_the_entry_config(self):
        """gitleaks resolves ``[extend].path`` against the PROCESS CWD.

        The scan runs with ``cwd=GITLEAKS_DIR``, so the extend target must sit
        in the same directory. A chain that resolves only from a checkout is
        the exact condition that took the dashboard box's egress to zero
        (alpha-engine-config-I4451 / -I4471 / -I8267).
        """
        d = session_dlp.packaged_gitleaks_dir()
        assert session_dlp._verify_gitleaks_config_chain_at(d) is None

    def test_resolver_never_returns_a_directory_without_the_entry_config(
        self, monkeypatch, tmp_path
    ):
        """A present-but-empty operator directory must not shadow the packaged copy.

        ``/opt/llm-routing`` existing while empty is what a half-finished
        bootstrap leaves behind. The old resolver tested ``os.path.isdir`` and
        returned it, turning a provisioning slip into a total egress outage
        with a usable ruleset sitting on the same disk.
        """
        empty = tmp_path / "opt-llm-routing"
        empty.mkdir()
        monkeypatch.delenv("KREPIS_GITLEAKS_DIR", raising=False)
        monkeypatch.setattr(session_dlp, "_OPERATOR_CONFIG_DIRS", (str(empty),))
        resolved = session_dlp._gitleaks_dir()
        assert resolved == session_dlp.packaged_gitleaks_dir()

    def test_operator_directory_still_wins_when_complete(
        self, monkeypatch, tmp_path
    ):
        """The packaged copy is the floor, never an override.

        A box that manages its own ruleset keeps it — otherwise this change
        would silently replace a tightened production ruleset with the shipped
        one.
        """
        opt = tmp_path / "opt-llm-routing"
        opt.mkdir()
        (opt / "gitleaks-egress.toml").write_text(
            '[extend]\npath = "./gitleaks-custom.toml"\n', encoding="utf-8"
        )
        (opt / "gitleaks-custom.toml").write_text('title = "x"\n', encoding="utf-8")
        monkeypatch.delenv("KREPIS_GITLEAKS_DIR", raising=False)
        monkeypatch.setattr(session_dlp, "_OPERATOR_CONFIG_DIRS", (str(opt),))
        assert session_dlp._gitleaks_dir() == str(opt)

    def test_env_override_wins_over_operator_directory(self, monkeypatch, tmp_path):
        override = tmp_path / "override-routing"
        override.mkdir()
        (override / "gitleaks-egress.toml").write_text(
            'title = "e"\n', encoding="utf-8"
        )
        opt = tmp_path / "opt-llm-routing"
        opt.mkdir()
        (opt / "gitleaks-egress.toml").write_text('title = "o"\n', encoding="utf-8")
        monkeypatch.setenv("KREPIS_GITLEAKS_DIR", str(override))
        monkeypatch.setattr(session_dlp, "_OPERATOR_CONFIG_DIRS", (str(opt),))
        assert session_dlp._gitleaks_dir() == str(override)


class TestPreflight:
    """Readiness is a value you can read, not something you learn by tripping it."""

    def test_preflight_reports_the_packaged_source_when_nothing_is_provisioned(
        self, monkeypatch
    ):
        monkeypatch.delenv("KREPIS_GITLEAKS_DIR", raising=False)
        monkeypatch.setattr(session_dlp, "_OPERATOR_CONFIG_DIRS", ())
        monkeypatch.setattr(
            session_dlp, "GITLEAKS_DIR", session_dlp.packaged_gitleaks_dir()
        )
        pf = session_dlp.preflight()
        assert pf.config_source == "packaged"
        assert pf.config_error is None
        assert pf.ruleset_matches_packaged is True

    def test_administratively_disabled_is_not_ready(self, monkeypatch):
        """``KREPIS_DLP_DISABLED`` must never read as a healthy control.

        alpha-engine-config-I10001: a process with the fleet's only
        Lambda-path DLP control switched off is otherwise indistinguishable
        from one scanning cleanly.
        """
        monkeypatch.setenv("KREPIS_DLP_DISABLED", "1")
        assert session_dlp.preflight().ready is False

    def test_missing_binary_is_reported_as_not_ready_with_a_reason(
        self, monkeypatch
    ):
        """The binary is the half krepis cannot ship, so it must be SAID.

        Measured 2026-08-19 on the crucible-research thinktank spot box: a
        whole run aborted on ``gitleaks binary not found on PATH``, discovered
        by the run failing rather than by any readiness check.
        """
        monkeypatch.setattr(session_dlp.shutil, "which", lambda _n: None)
        pf = session_dlp.preflight()
        assert pf.ready is False
        assert "not found on PATH" in (pf.binary_error or "")

    def test_chain_hash_changes_when_the_extended_ruleset_changes(self, tmp_path):
        """Hashing the entry file alone would read identical across two rulesets.

        Every fleet-specific secret shape lives in the EXTENDED file, so a
        signature blind to it cannot detect the divergence it exists to
        report (alpha-engine-config-I9712).
        """
        d = tmp_path / "cfg"
        d.mkdir()
        (d / "gitleaks-egress.toml").write_text(
            '[extend]\npath = "./gitleaks-custom.toml"\n', encoding="utf-8"
        )
        custom = d / "gitleaks-custom.toml"
        custom.write_text('title = "one"\n', encoding="utf-8")
        first = session_dlp._hash_config_chain(str(d))
        custom.write_text('title = "two"\n', encoding="utf-8")
        assert session_dlp._hash_config_chain(str(d)) != first

    def test_cli_preflight_exits_nonzero_when_not_ready(self, monkeypatch, capsys):
        monkeypatch.setenv("KREPIS_DLP_DISABLED", "1")
        assert session_dlp._main(["preflight", "--json"]) == 1
        assert '"ready": false' in capsys.readouterr().out


@pytest.mark.skipif(
    not session_dlp.shutil.which("gitleaks"),
    reason="gitleaks binary not on PATH",
)
def test_a_real_scan_runs_against_the_packaged_ruleset(monkeypatch):
    """End to end on the shipped config — the property the whole class is about.

    Not a mock: the packaged ruleset must actually load in gitleaks, extend
    chain included. A packaged file that gitleaks rejects would satisfy every
    path assertion above and still fail closed in production.
    """
    packaged = session_dlp.packaged_gitleaks_dir()
    monkeypatch.setattr(session_dlp, "GITLEAKS_DIR", packaged)
    monkeypatch.setattr(
        session_dlp,
        "GITLEAKS_CONFIG",
        os.path.join(packaged, "gitleaks-egress.toml"),
    )
    monkeypatch.delenv("KREPIS_DLP_DISABLED", raising=False)
    session_dlp._cache._config_sig = None
    session_dlp._cache._clean.clear()
    verdict, reason, _ms, _ratio = session_dlp.scan_request(
        b'{"model": "m", "messages": [{"role": "user", "content": "hello world"}]}'
    )
    assert verdict == session_dlp.DLP_OK, reason
