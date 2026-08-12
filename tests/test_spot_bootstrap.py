"""The three defects that cost three weekly runs, asserted (config-I6922).

`krepis.spot_bootstrap` exists because the same Bash heredoc lived in two
repos and diverged. On 2026-08-11 `ne-weekly-freshness-pipeline` failed three
times running, each on a different defect inside that one script, each
invisible until the previous was fixed:

    run 5   rc=137 at PT5M0.0s      systemctl blocked on a Type=oneshot unit
    run 6   python3.12 not found    the script asserted an absent interpreter
    run 7   fatal: repository ''    an interpolated var that was never exported

Two of the three had already been fixed in the other repo's copy, 16 and 23
hours earlier. These tests are what makes a fourth port unnecessary: the
defects are properties of the rendered string, so they fail here in
milliseconds instead of on a live spot in three hours.
"""

from __future__ import annotations

import pytest

from krepis.spot_bootstrap import (
    PYTHON,
    ConfigCopy,
    SpotBootstrapSpec,
    main,
    render_bootstrap,
    render_install_deps,
)


def _spec(**kw) -> SpotBootstrapSpec:
    base = {
        "repo_url": "https://github.com/nousergon/nousergon-data.git",
        "checkout": "/home/ec2-user/data",
    }
    base.update(kw)
    return SpotBootstrapSpec(**base)


# ── Run 5: the watchdog unit blocked its own bootstrap ───────────────────


def test_the_watchdog_unit_is_simple_never_oneshot():
    """`systemctl start` on Type=oneshot blocks until ExecStart exits.

    ExecStart here is an endless supervision loop, and TimeoutStartSec
    defaults to infinity for oneshot — so the unit hung every bootstrap that
    enabled it until SSM killed the command at its budget.
    """
    script = render_bootstrap(_spec())
    # Scoped to the unit stanza: the guard's ERROR message legitimately names
    # "Type=oneshot" as the failure mode it exists to catch, and a blanket
    # substring check would forbid explaining the bug in the message.
    unit = script.split("[Service]", 1)[1].split("[Install]", 1)[0]
    assert "Type=simple" in unit
    assert "Type=oneshot" not in unit
    assert "RemainAfterExit" not in unit


def test_enabling_the_watchdog_is_time_bounded_and_says_why_on_failure():
    """A bounded failure with a message beats an unbounded silence.

    Run 5 consumed its whole 300s SSM budget and died under SIGKILL with no
    output past the symlink line, which read as "bootstrap is slow" rather
    than "systemctl is blocked forever".
    """
    script = render_bootstrap(_spec())
    assert "timeout 60 systemctl enable --now ec2-spot-watchdog" in script
    assert "Type=oneshot blocks systemctl start forever" in script


# ── Run 6: the interpreter was asserted, not installed ───────────────────


def test_the_interpreter_is_installed_before_it_is_asserted():
    """Order is the defect. Asserting first is a precondition on an AMI we
    do not build; the AL2023 spot image does not ship python3.12."""
    script = render_bootstrap(_spec())
    install = script.index("dnf install")
    assert_at = script.index("not found after dnf install")
    assert install < assert_at, "the assertion must be a POST-condition"


def test_the_interpreter_assertion_survives_the_fallback_install():
    """The `|| dnf install python3` fallback must not satisfy the check.

    A silent fall back to the system python3 resolves wheels against a
    different version than requirements.txt was compiled for — drift wearing
    resilience's clothes.
    """
    script = render_bootstrap(_spec())
    assert f"command -v {PYTHON} >/dev/null || {{ echo" in script


def test_install_deps_has_no_system_python_fallback():
    """render_bootstrap guarantees the interpreter, so a fallback here would
    silently undo that guarantee one SSM command later."""
    deps = render_install_deps(_spec())
    assert f"{PYTHON} -m pip install" in deps
    assert "PY=python3" not in deps


# ── Run 7: an interpolated variable that was never exported ──────────────


def test_the_repo_url_is_a_literal_not_a_shell_variable():
    """The clone ran against an empty string for a day.

    The heredoc was single-quoted, so `${REPO_URL}` expanded ON THE SPOT
    where it was unset — while a comment three lines up claimed the launcher
    exported it. Baking the URL in removes the class, not just the instance.
    """
    script = render_bootstrap(_spec())
    assert "https://github.com/nousergon/nousergon-data.git" in script
    assert "${REPO_URL}" not in script
    assert "$REPO_URL" not in script


def test_no_undeclared_shell_variable_survives_into_the_script():
    """Generalises run 7: every `${VAR}` in the output must be either
    exported by the preamble or a documented spot-side variable."""
    spec = _spec(
        exports={"S3_STAGING": "s3://bucket/prefix"},
        config_copies=(ConfigCopy(source_name="config.yaml", dest="/opt/cfg/config.yaml"),),
    )
    script = render_bootstrap(spec)
    import re

    referenced = set(re.findall(r"\$\{(\w+)\}", script))
    exported = set(spec.exports) | {"HOME", "XDG_CACHE_HOME", "AWS_REGION", "AWS_DEFAULT_REGION"}
    assert referenced <= exported, f"unexported: {sorted(referenced - exported)}"


# ── Config staging ───────────────────────────────────────────────────────


def test_a_config_copy_creates_its_destination_directory_first():
    spec = _spec(
        exports={"S3_STAGING": "s3://b/p"},
        config_copies=(ConfigCopy(source_name="config.yaml", dest="/a/b/config.yaml"),),
    )
    script = render_bootstrap(spec)
    assert script.index("mkdir -p /a/b") < script.index("aws s3 cp")


def test_multiple_config_copies_all_render():
    """The predictor stages one file to two paths; the data copy stages one
    to one. Both shapes have to work or one repo cannot cut over."""
    spec = _spec(
        exports={"S3_STAGING": "s3://b/p"},
        config_copies=(
            ConfigCopy(source_name="predictor.yaml", dest="/c/config/predictor.yaml"),
            ConfigCopy(source_name="predictor.yaml", dest="/c/experiments/reference/predictor/predictor.yaml"),
        ),
    )
    script = render_bootstrap(spec)
    assert script.count("aws s3 cp") == 2


def test_chown_is_emitted_only_when_asked():
    with_chown = render_bootstrap(
        _spec(
            exports={"S3_STAGING": "s3://b/p"},
            config_copies=(
                ConfigCopy(source_name="c.yaml", dest="/x/c.yaml", chown="/x"),
            ),
        )
    )
    without = render_bootstrap(
        _spec(
            exports={"S3_STAGING": "s3://b/p"},
            config_copies=(ConfigCopy(source_name="c.yaml", dest="/x/c.yaml"),),
        )
    )
    assert "chown -R ec2-user:ec2-user /x" in with_chown
    assert "chown" not in without


def test_a_spec_with_no_config_copies_renders_cleanly():
    """Not every workload stages config. An empty block must not leave a
    stray blank section or a dangling `mkdir`."""
    script = render_bootstrap(_spec())
    assert "aws s3 cp" not in script
    assert "\n\n\n" not in script


# ── Injection safety ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "hostile",
    ["/tmp/a b", "/tmp/a;rm -rf /", "/tmp/$(whoami)", "/tmp/a'b"],
)
def test_paths_are_shell_quoted(hostile: str):
    """These values reach a root shell on the instance. Quoting is not
    hygiene here, it is the boundary."""
    script = render_bootstrap(_spec(checkout=hostile))
    assert "rm -rf /tmp/a;rm -rf /" not in script
    assert ";rm -rf /\n" not in script


# ── Determinism ──────────────────────────────────────────────────────────


def test_rendering_is_a_pure_function_of_the_spec():
    """No clock, no environment, no AWS — which is what makes the defects
    above assertable in a unit test rather than on a live spot."""
    spec = _spec(exports={"B": "2", "A": "1"})
    assert render_bootstrap(spec) == render_bootstrap(spec)


def test_exports_are_ordered_so_the_output_is_diffable():
    script = render_bootstrap(_spec(exports={"Z": "1", "A": "2"}))
    assert script.index("A=2") < script.index("Z=1")


# ── CLI ──────────────────────────────────────────────────────────────────


def test_cli_renders_to_stdout(capsys):
    rc = main(
        [
            "render",
            "--repo-url",
            "https://github.com/nousergon/nousergon-data.git",
            "--checkout",
            "/home/ec2-user/data",
            "--export",
            "S3_STAGING=s3://b/p",
            "--config-copy",
            "config.yaml:/home/ec2-user/alpha-engine-config/data/config.yaml:/home/ec2-user/alpha-engine-config",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "Type=simple" in out
    assert "chown -R ec2-user:ec2-user /home/ec2-user/alpha-engine-config" in out


def test_cli_rejects_a_malformed_config_copy():
    with pytest.raises(SystemExit):
        main(["render", "--repo-url", "u", "--checkout", "/c", "--config-copy", "nope"])


def test_cli_rejects_a_malformed_export():
    with pytest.raises(SystemExit):
        main(["render", "--repo-url", "u", "--checkout", "/c", "--export", "nope"])


def test_cli_render_deps_emits_the_deps_script(capsys):
    rc = main(["render-deps", "--repo-url", "u", "--checkout", "/home/ec2-user/data"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "pip install" in out
    assert "cd /home/ec2-user/data" in out


# ── Parity with the two live Bash copies ─────────────────────────────────
#
# The cutover only holds if the rendered script does what the copies it
# replaces do. These read the live files from sibling checkouts and SKIP when
# absent (CI runners have no fleet checkout), so they are a laptop-side and
# fleet-CI guard, not a hard dependency.

import os  # noqa: E402
from pathlib import Path  # noqa: E402

_FLEET = Path.home() / "Development"
_COPIES = {
    "nousergon-data": _FLEET / "nousergon-data" / "infrastructure" / "_spot_common.sh",
    "crucible-predictor": _FLEET / "alpha-engine-predictor" / "infrastructure" / "_spot_common.sh",
}


@pytest.mark.parametrize("repo,path", sorted(_COPIES.items()))
def test_the_rendered_script_carries_every_hardening_the_live_copy_has(repo: str, path: Path):
    """No hardening is LOST in the move to Python.

    A consolidation that silently drops a property of one of the copies is
    the rewrite failure mode this arc has already paid for twice
    (bugclass: a rewrite drops properties the original had). Each invariant
    below is a fix somebody landed after an outage.
    """
    if not path.exists():
        pytest.skip(f"{path} not present")
    live = path.read_text()
    rendered = render_bootstrap(
        _spec(exports={"S3_STAGING": "s3://b/p"}, config_copies=(
            ConfigCopy(source_name="c.yaml", dest="/x/c.yaml"),
        ))
    )

    invariants = {
        "watchdog self-terminates on SSM stoppage": "shutdown -h now",
        "watchdog supervises amazon-ssm-agent": "systemctl is-active amazon-ssm-agent",
        "interpreter is installed": "dnf install",
        "clone is depth-1": "--depth 1",
        "clone is idempotent": "rm -rf",
    }
    for name, needle in invariants.items():
        if needle not in live:
            continue  # this copy does not have it; nothing to preserve
        assert needle in rendered, f"{repo}: rendered script drops {name!r}"


@pytest.mark.skipif(
    not (_COPIES["nousergon-data"]).exists(),
    reason="fleet checkout absent",
)
def test_the_live_copies_no_longer_carry_the_defects_this_module_encodes():
    """Both Bash copies were fixed on 2026-08-11. If either regresses before
    the cutover completes, this is where it shows — the module is not the
    only thing that has to stay correct while two implementations coexist."""
    for repo, path in _COPIES.items():
        if not path.exists():
            continue
        live = path.read_text()
        service = live.split("[Service]", 1)
        if len(service) > 1:
            unit = service[1].split("[Install]", 1)[0]
            assert "Type=oneshot" not in unit, f"{repo}: watchdog unit regressed to oneshot"
        assert "dnf install" in live, f"{repo}: interpreter install removed"


def test_an_explicit_mkdir_overrides_the_derived_parent():
    """Some destinations need a directory ABOVE the file's own parent created
    — the predictor's experiments/<id>/predictor/ tree is three levels deep
    and only the leaf is derivable from the file path."""
    copy = ConfigCopy(source_name="c.yaml", dest="/a/b/c.yaml", mkdir="/a")
    assert copy.parent() == "/a"
    script = render_bootstrap(
        _spec(exports={"S3_STAGING": "s3://b/p"}, config_copies=(copy,))
    )
    assert "mkdir -p /a\n" in script


def test_cli_json_mode_emits_a_parseable_envelope(capsys):
    """The Bash side embeds the script in a heredoc; JSON mode is for callers
    that dispatch it programmatically and need it as one field."""
    import json as _json

    rc = main(["render", "--repo-url", "u", "--checkout", "/c", "--json"])
    payload = _json.loads(capsys.readouterr().out)
    assert rc == 0
    assert "Type=simple" in payload["script"]


# ── Run 12: the deps step that discarded the evidence (config-I6949) ─────


def test_install_deps_does_not_pipe_the_install_through_tail():
    """`pip install ... 2>&1 | tail -1` left one line of pip's run-as-root
    warning as the entire record of a resolution that had quietly skipped a
    strict-mode dependency. The predictor spot smoke then died on a missing
    `flow_doctor` with nothing upstream to read."""
    deps = render_install_deps(_spec())
    assert "| tail -1" not in deps


def test_install_deps_dumps_the_log_when_pip_fails():
    deps = render_install_deps(_spec())
    assert "_pip_log" in deps
    # The failure branch must print the captured log, or preserving it buys
    # nothing — the SSM step output is the only surface anyone reads.
    assert 'tail -80 "$_pip_log" >&2' in deps
    assert "exit 1" in deps


def test_install_deps_surfaces_a_dropped_extra_and_runs_pip_check():
    """A silently-dropped extra is the exact shape of the failure this
    step is meant to make legible, and it is a pip WARNING on a zero exit."""
    deps = render_install_deps(_spec())
    assert "does not provide the extra" in deps
    assert "pip check" in deps


def test_install_deps_FAILS_on_a_dropped_extra_rather_than_only_reporting_it():
    """Surfacing a dropped extra in the log is not enough — the step must exit
    non-zero.

    pip emits `WARNING: ... does not provide the extra` on a SUCCESSFUL exit,
    so a step that only greps it passes, and the failure lands later as an
    ImportError in a different process with the install log gone. That is
    exactly how config#6963 reached production (`ModuleNotFoundError: No module
    named 'flow_doctor'` out of krepis.logging.setup_logging).

    The predictor's `_spot_common.sh` already carried this hard-fail; this
    renderer did not, which made the fleet copy WEAKER than the consumer it is
    meant to replace. config-I6922: a consumer cannot be migrated onto a shared
    implementation that drops behaviour it already has.
    """
    deps = render_install_deps(_spec())
    extra_guard = deps.split("does not provide the extra", 1)[1]
    # The guard must be a failing branch, not a bare grep: an `exit 1` has to
    # appear after the match and before `pip check`, which is deliberately
    # non-fatal and would otherwise be the next thing to run.
    before_pip_check = extra_guard.split("pip check", 1)[0]
    assert "exit 1" in before_pip_check
    assert "the environment is incomplete" in before_pip_check


def test_install_deps_keeps_pip_check_non_fatal():
    """`pip check` stays advisory — the failure it would raise is a
    pre-existing AMI-baked conflict unrelated to this checkout, which would
    otherwise fail every lane on every run. Guards the hard-fail added above
    from being widened onto it by a later edit."""
    deps = render_install_deps(_spec())
    tail = deps.split("pip check", 1)[1]
    assert "||" in tail.splitlines()[0] or "WARNING" in tail
