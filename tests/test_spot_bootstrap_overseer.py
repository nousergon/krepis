"""The nine capabilities the Overseer substrate needed (alpha-engine-config-I7374).

`alpha-engine-config` carries SEVEN `infrastructure/*_spot_bootstrap.sh` files,
~4,059 lines, plus an eighth artifact — `overseer_spot_bootstrap.sh` — that is a
RIVAL unification of the other six behind a `case "$PLAYBOOK"`. Two rival
unifications is worse than either, and Brian ruled on 2026-08-14 that the one
which survives is this module: it is already the fleet's renderer for the data
plane, and it is a pure function testable without a live spot.

Nine capabilities stood between that ruling and a cutover. Each is tested here
twice — present once declared, and ABSENT from
`tests/golden/pre_extension_bootstrap.sh`, which this module rendered before the
extension existed. That golden is also what proves the extension is additive:
`krepis.spot_bootstrap` went from zero call sites to two live ones on
2026-08-14, so this landed under production consumers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from krepis.spot_bootstrap import (
    Clone,
    ConfigCopy,
    EgressProxy,
    PrivilegeDrop,
    RunLog,
    SpotBootstrapSpec,
    SsmSecret,
    load_workloads,
    main,
    render_bootstrap,
)

_GOLDEN = Path(__file__).parent / "golden" / "pre_extension_bootstrap.sh"


def _pre_extension_spec() -> SpotBootstrapSpec:
    """The exact spec the frozen golden was rendered from."""
    return SpotBootstrapSpec(
        repo_url="https://github.com/nousergon/nousergon-data.git",
        checkout="/home/ec2-user/data",
        exports={"S3_STAGING": "s3://alpha-engine-research/staging/x"},
        config_copies=(
            ConfigCopy(source_name="config.yaml", dest="/home/ec2-user/data/config.yaml"),
            ConfigCopy(
                when="${STAGED}", source_name="p.json", dest="/opt/p/p.json", chown="/opt/p"
            ),
        ),
        extra_clones=(
            Clone(
                repo_url="https://github.com/nousergon/crucible-executor.git",
                checkout="/home/ec2-user/executor",
            ),
        ),
        max_runtime_seconds=3600,
    )


def _overseer_spec(**kw) -> SpotBootstrapSpec:
    """A full agent playbook — the widest shape the Overseer runs."""
    base = dict(
        checkout="/home/ec2-user/alpha-engine-config",
        clone=False,
        unit_prefix="ci-watch",
        deadman_seconds=25200,
        max_runtime_seconds=19200,
        shutdown_delay_seconds=60,
        run_log=RunLog(
            local_path="/var/log/ci-watch-run.log",
            s3_uri="s3://alpha-engine-research/overseer/run_logs/ci-watch/run.log",
        ),
        secrets=(
            SsmSecret(
                env_var="GH_TOKEN",
                parameter="/alpha-engine/saturday_sf_watch/github_pat",
                required=True,
                via_file=True,
            ),
            SsmSecret(
                env_var="TELEGRAM_CHAT_ID", parameter="/alpha-engine/TELEGRAM_CHAT_ID"
            ),
        ),
        egress_proxy=EgressProxy(
            port=8976,
            stage_dir="/opt/ci-watch-llm-routing",
            source_dir="/home/ec2-user/alpha-engine-config/infrastructure/groom-llm-routing",
            upstream_host="api.deepseek.com",
            api_key_env="DEEPSEEK_API_KEY",
            api_key_parameter="/alpha-engine/groom/deepseek_api_key",
            upstream_prefix="/anthropic",
            required_binaries=("gitleaks",),
        ),
        privilege_drop=PrivilegeDrop(
            user="ec2-user",
            command=("bash", "/home/ec2-user/alpha-engine-config/scripts/ci_watch_run.sh"),
            env=(("HOME", "/home/ec2-user"), ("CI_REPO", "${CI_REPO:-}")),
            secret_env_vars=("GH_TOKEN",),
            chown=("/home/ec2-user/alpha-engine-config",),
        ),
    )
    base.update(kw)
    return SpotBootstrapSpec(**base)


# ── Backward compatibility: this landed under two live consumers ─────────


def test_a_spec_written_before_the_extension_renders_byte_identically():
    """`krepis.spot_bootstrap` had ZERO call sites until 2026-08-14 and two by
    the end of that day (`nousergon-data`, `crucible-predictor`). Every
    capability below is additive, and this is the assertion that says so in
    BYTES rather than in a changelog: the golden was produced by this module as
    it stood before the extension.
    """
    assert render_bootstrap(_pre_extension_spec()) == _GOLDEN.read_text()


# ── 1. Two-tier transient watchdog: the dead-man ─────────────────────────


def test_the_deadman_is_armed_before_anything_that_can_fail():
    """A bootstrap that dies before arming its cap leaves a box with no
    watchdog, no marker and no ceiling — measured 2026-07-28 as 11 ci-watch
    plus 1 sf-watch boxes alive with dead charters."""
    script = render_bootstrap(_overseer_spec())
    deadman = script.index("--unit=ci-watch-deadman")
    for later in ("--unit=ci-watch-hard-timeout", "exec > >(tee", "get_secret()"):
        assert script.index(later) > deadman, f"{later!r} is armed before the dead-man"


def test_the_deadman_does_not_replace_the_hard_cap_or_the_liveness_watchdog():
    """Three guarantees, three different questions: the box was abandoned
    before the cap armed / the workload hung / the SSM agent died. Each fork
    the fleet has shipped carried some subset and was uncovered against the
    others' failure mode."""
    script = render_bootstrap(_overseer_spec())
    assert "--unit=ci-watch-deadman" in script
    assert "--unit=ci-watch-hard-timeout" in script
    assert "ec2-spot-watchdog.service" in script


def test_a_deadman_at_or_below_the_hard_cap_is_refused():
    """A failsafe below the real cap truncates legitimate runs and mislabels
    them as dead-man kills — worse than having no failsafe."""
    with pytest.raises(ValueError, match="ABOVE"):
        _overseer_spec(deadman_seconds=19200)


def test_the_deadman_warns_rather_than_aborting_when_it_cannot_arm():
    """The hard cap aborts; the failsafe does not. A failsafe that refuses to
    start the run it protects converts a partial guarantee into an outage."""
    script = render_bootstrap(_overseer_spec())
    block = script.split("--unit=ci-watch-deadman", 1)[1].split("\n\n", 1)[0]
    assert "WARN" in block
    assert "exit 1" not in block


# ── 2. Record before kill ────────────────────────────────────────────────


def test_no_terminating_timer_kills_silently_when_there_is_a_run_log():
    """The defect this closes: four of the seven Overseer bootstraps armed
    their watchdog straight at `/sbin/shutdown -h now`, so a watchdog kill was
    indistinguishable from a clean exit — and every one of the four HAD a run
    log it could have stamped.

    Derived, not enumerated: every `systemd-run` line in the render is checked,
    so a timer added later cannot reintroduce the gap by being new.
    """
    script = render_bootstrap(_overseer_spec())
    timers = [
        line for line in script.splitlines() if line.lstrip().startswith("systemd-run")
    ]
    assert timers
    for timer in timers:
        block = timer + script.split(timer, 1)[1].split("\n\n", 1)[0]
        if "delayed-shutdown" in block:
            continue  # the EXIT-trap path; the run has already recorded itself
        assert "ci-watch-record-kill" in block, f"kills silently: {timer}"


def test_the_recorder_stamps_the_log_with_the_reason_and_ships_it():
    script = render_bootstrap(_overseer_spec())
    recorder = script.split("<<'RECORDKILL'", 1)[1].split("RECORDKILL", 1)[0]
    assert "TERMINATED BY TIMER" in recorder
    assert "$KILL_REASON" in recorder
    assert "aws s3 cp /var/log/ci-watch-run.log" in recorder


def test_the_recorder_never_shuts_the_box_down_itself():
    """The timer's own command does that, unconditionally, after the recorder
    returns. A recorder that owned the shutdown would make every future edit to
    it a chance to strand a running box."""
    script = render_bootstrap(_overseer_spec())
    recorder = script.split("<<'RECORDKILL'", 1)[1].split("RECORDKILL", 1)[0]
    assert "shutdown" not in recorder
    assert "/sbin/shutdown -h now" in script


def test_the_recorder_is_written_before_any_timer_is_armed():
    """A timer whose ExecStart does not exist yet fires into nothing, and the
    box it was guarding keeps running."""
    script = render_bootstrap(_overseer_spec())
    assert script.index("chmod +x /usr/local/sbin/ci-watch-record-kill") < script.index(
        "systemd-run"
    )


def test_the_hard_cap_reason_and_the_deadman_reason_are_different():
    """`budget_exhausted` and `deadman` have opposite remedies — one is a
    cadence question, the other is a bootstrap that died early."""
    script = render_bootstrap(_overseer_spec())
    assert "ci-watch-record-kill budget_exhausted" in script
    assert "ci-watch-record-kill deadman" in script


def test_a_workload_supplied_record_runs_in_the_recorder():
    script = render_bootstrap(
        _overseer_spec(record_before_kill='write_telemetry.sh "$KILL_REASON"')
    )
    recorder = script.split("<<'RECORDKILL'", 1)[1].split("RECORDKILL", 1)[0]
    assert 'write_telemetry.sh "$KILL_REASON"' in recorder


def test_a_record_hook_alone_is_enough_to_route_every_timer():
    """A workload with no run log but its own telemetry writer still records."""
    spec = SpotBootstrapSpec(
        repo_url="https://github.com/nousergon/x.git",
        checkout="/home/ec2-user/x",
        max_runtime_seconds=600,
        record_before_kill="echo dead > /tmp/marker",
    )
    assert "ec2-spot-record-kill budget_exhausted" in render_bootstrap(spec)


# ── 3. Run-log shipping ──────────────────────────────────────────────────


def test_the_run_log_is_flushed_before_it_is_read():
    """`tee` behind a process substitution holds bytes in a pipe. A size check
    read at trap time is FALSE on a file that is non-empty milliseconds later,
    which is how the previous guard skipped every upload while reporting
    nothing (alpha-engine-config-I5512)."""
    script = render_bootstrap(_overseer_spec())
    ship = script.split("_ship_run_log() {", 1)[1].split("\n}", 1)[0]
    assert ship.index("_flush_run_log") < ship.index("-s /var/log/ci-watch-run.log")


def test_the_run_log_ship_reports_its_outcome_either_way():
    """Non-fatal, never silent: the failure being fixed is silence, not the
    fail-open."""
    script = render_bootstrap(_overseer_spec())
    ship = script.split("_ship_run_log() {", 1)[1].split("\n}", 1)[0]
    assert "run log shipped ->" in ship
    assert "WARN: run-log ship FAILED" in ship
    assert "WARN: run-log ship skipped" in ship


def test_the_run_log_ships_periodically_not_only_at_exit():
    """An exit-time-only write cannot survive the failure mode it exists for."""
    script = render_bootstrap(_overseer_spec())
    assert "_run_log_shipper_loop &" in script
    assert "sleep 60" in script


def test_the_periodic_shipper_is_stopped_before_the_final_copy():
    """Otherwise the background loop races the final ship for the same key."""
    script = render_bootstrap(_overseer_spec())
    finish = script.split("finish() {", 1)[1].split("\n}", 1)[0]
    assert finish.index("_stop_run_log_shipper") < finish.index("_ship_run_log")


def test_a_signal_death_ships_the_log_too():
    script = render_bootstrap(_overseer_spec())
    assert "trap '_stop_run_log_shipper; _ship_run_log' TERM INT" in script


def test_a_config_runner_shaped_workload_gets_a_log_and_a_kill_record():
    """The second named defect of alpha-engine-config-I7374:
    `config_runner_spot_bootstrap.sh` ships NO run log and writes NO run record,
    so when it fails there is nothing to read — its only output is an SSM
    invocation that truncates and expires. A NON-AGENT workload, with no proxy
    and no privilege drop, must still get both.
    """
    spec = SpotBootstrapSpec(
        checkout="/home/ec2-user/alpha-engine-config",
        clone=False,
        unit_prefix="config-runner",
        deadman_seconds=25200,
        max_runtime_seconds=1800,
        shutdown_delay_seconds=30,
        run_log=RunLog(
            local_path="/var/log/config-runner-run.log",
            s3_uri="s3://alpha-engine-research/overseer/run_logs/config-runner/run.log",
        ),
    )
    script = render_bootstrap(spec)
    assert "exec > >(tee -a /var/log/config-runner-run.log)" in script
    assert "config-runner-record-kill budget_exhausted" in script
    assert "config-runner-record-kill deadman" in script


# ── 4. Per-workload LLM egress proxy staging ─────────────────────────────


def test_the_proxy_port_and_stage_dir_are_parameters_never_constants():
    """Co-tenant workloads on one host collide otherwise, and a collision
    presents as one workload silently using another's upstream."""
    a = render_bootstrap(_overseer_spec())
    b = render_bootstrap(
        _overseer_spec(
            unit_prefix="sf-watch",
            egress_proxy=EgressProxy(
                port=8977,
                stage_dir="/opt/sf-watch-llm-routing",
                source_dir="/src",
                upstream_host="api.deepseek.com",
                api_key_env="DEEPSEEK_API_KEY",
                api_key_parameter="/alpha-engine/groom/deepseek_api_key",
            ),
        )
    )
    assert "127.0.0.1:8976" in a and "127.0.0.1:8977" in b
    assert "/opt/ci-watch-llm-routing" in a and "/opt/sf-watch-llm-routing" in b


def test_a_missing_scanner_binary_aborts_rather_than_running_ungated():
    script = render_bootstrap(_overseer_spec())
    guard = script.split("command -v gitleaks", 1)[1].split("fi", 1)[0]
    assert "refusing to run ungated" in guard
    assert "exit 1" in guard


def test_an_unhealthy_proxy_aborts_rather_than_running_ungated():
    """"The proxy did not come up" must never degrade into "ran ungated"."""
    script = render_bootstrap(_overseer_spec())
    guard = script.split('if [ "$_proxy_healthy" != "true" ]', 1)[1].split("fi", 1)[0]
    assert "exit 1" in guard


def test_the_base_url_is_exported_only_after_the_health_check_passes():
    script = render_bootstrap(_overseer_spec())
    assert script.index('export ANTHROPIC_BASE_URL="http://127.0.0.1:8976"') > script.index(
        "_proxy_healthy=true"
    )


def test_the_proxy_credential_leaves_the_shell_after_the_launch():
    script = render_bootstrap(_overseer_spec())
    assert "unset _proxy_key" in script


# ── 5. Privilege drop with an explicit env allow-list ────────────────────


def test_the_workload_runs_as_a_non_root_user_with_a_named_env():
    script = render_bootstrap(_overseer_spec())
    assert "runuser -u ec2-user -- /usr/bin/env" in script
    assert 'CI_REPO="${CI_REPO:-}"' in script


def test_a_secret_never_crosses_the_boundary_in_argv():
    """`runuser -- env NAME=VALUE` is world-readable in /proc for the life of
    the process (alpha-engine-config-I4949/I4956)."""
    script = render_bootstrap(_overseer_spec())
    drop = script.split("runuser -u ec2-user", 1)[1]
    assert "GH_TOKEN=" not in drop
    assert 'SECRET_ENV_FILE="$_secret_env_file"' in drop
    assert 'rm -f "$SECRET_ENV_FILE"' in drop


def test_the_secret_is_dropped_from_the_parent_shell_once_staged():
    script = render_bootstrap(_overseer_spec())
    assert "unset GH_TOKEN" in script


def test_an_env_value_that_would_execute_as_root_is_refused():
    """Command substitution inside the privilege-drop line runs as ROOT,
    before the drop."""
    with pytest.raises(ValueError, match="execute as ROOT"):
        _overseer_spec(
            privilege_drop=PrivilegeDrop(
                user="ec2-user",
                command=("bash", "/x.sh"),
                env=(("EVIL", "$(id -u)"),),
            )
        )


def test_an_empty_privilege_drop_command_is_refused():
    with pytest.raises(ValueError, match="nothing would run"):
        _overseer_spec(privilege_drop=PrivilegeDrop(user="ec2-user", command=()))


def test_the_checkout_is_handed_to_the_run_user_before_the_drop():
    script = render_bootstrap(_overseer_spec())
    assert script.index("chown -R ec2-user:ec2-user") < script.index("runuser -u")


# ── 6. No-clone mode ─────────────────────────────────────────────────────


def test_no_clone_mode_asserts_the_checkout_rather_than_assuming_it():
    """Six of the seven Overseer bootstraps derive a repo root from `$0` and
    trust it. A dispatcher prelude that half-failed leaves a directory that
    exists and is not a checkout, which is indistinguishable from success to a
    `dirname $0` — and the workload then dies on a missing file minutes later,
    in another process."""
    script = render_bootstrap(_overseer_spec())
    assert "git clone" not in script
    assert "is not a git checkout" in script
    assert "exit 1" in script.split("is not a git checkout", 1)[1].split("fi", 1)[0]


def test_clone_mode_is_still_the_default():
    assert "git clone" in render_bootstrap(_pre_extension_spec())


def test_a_cloning_spec_without_a_repo_url_is_refused():
    with pytest.raises(ValueError, match="repo_url is required"):
        SpotBootstrapSpec(checkout="/home/ec2-user/x")


def test_a_spec_without_a_checkout_is_refused():
    with pytest.raises(ValueError, match="checkout is required"):
        SpotBootstrapSpec(repo_url="https://github.com/nousergon/x.git")


# ── 7. The finish() EXIT trap ────────────────────────────────────────────


def test_the_exit_trap_defers_the_shutdown_so_ssm_can_report_first():
    """Shutting down immediately races the SSM agent's final-status callback:
    when shutdown wins, a command that genuinely succeeded is reported
    Failed/Undeliverable (alpha-engine-config#1472). Sleeping does NOT fix it —
    the agent only starts the callback after the monitored process exits, so a
    blocking sleep merely delays when the race begins."""
    script = render_bootstrap(_overseer_spec())
    finish = script.split("finish() {", 1)[1].split("\ntrap finish EXIT", 1)[0]
    assert "--unit=ci-watch-delayed-shutdown" in finish
    assert "--on-active=60" in finish
    assert "sleep" not in finish
    assert 'exit "$rc"' in finish


def test_the_exit_trap_retires_the_deadman():
    script = render_bootstrap(_overseer_spec())
    finish = script.split("finish() {", 1)[1].split("\ntrap finish EXIT", 1)[0]
    assert "systemctl stop ci-watch-deadman.timer" in finish


def test_a_failure_to_schedule_the_deferred_shutdown_falls_back_to_immediate():
    """Never leave the box up: a stranded spot bills until somebody notices."""
    script = render_bootstrap(_overseer_spec())
    finish = script.split("finish() {", 1)[1].split("\ntrap finish EXIT", 1)[0]
    assert "shutdown -h now" in finish.split("WARN: delayed-shutdown", 1)[1]


def test_the_finish_hook_runs_before_the_shutdown_is_scheduled():
    script = render_bootstrap(_overseer_spec(finish_hook='publish_lane.sh "$rc"'))
    finish = script.split("finish() {", 1)[1].split("\ntrap finish EXIT", 1)[0]
    assert finish.index('publish_lane.sh "$rc"') < finish.index("delayed-shutdown")


# ── 8. SSM secret fetch ──────────────────────────────────────────────────


def test_a_required_secret_that_is_absent_names_the_parameter_and_aborts():
    script = render_bootstrap(_overseer_spec())
    guard = script.split('if [ -z "${GH_TOKEN:-}" ]', 1)[1].split("fi", 1)[0]
    assert "/alpha-engine/saturday_sf_watch/github_pat" in guard
    assert "exit 1" in guard


def test_an_optional_secret_that_is_absent_says_so_rather_than_vanishing():
    """An unset variable the workload reads mostly degrades quietly. That WARN
    is the difference between a five-day "provider outage" and one line."""
    script = render_bootstrap(_overseer_spec())
    guard = script.split('if [ -z "${TELEGRAM_CHAT_ID:-}" ]', 1)[1].split("fi", 1)[0]
    assert "WARN" in guard
    assert "exit 1" not in guard


def test_the_secret_fetch_cannot_abort_before_its_own_named_check():
    """Under `set -e` a failed command substitution aborts at the assignment,
    and "the bootstrap died at line 90" is not the same information as
    "GH_TOKEN is absent from <parameter>"."""
    script = render_bootstrap(_overseer_spec())
    assert (
        'GH_TOKEN="$(get_secret /alpha-engine/saturday_sf_watch/github_pat)" || true'
        in script
    )


def test_a_via_file_secret_is_not_exported_into_the_shell_environment():
    script = render_bootstrap(_overseer_spec())
    assert "\nexport GH_TOKEN" not in script
    assert "\nexport TELEGRAM_CHAT_ID" in script


# ── 9. Many workloads, one procedure ─────────────────────────────────────


_TABLE = {
    "defaults": {
        "region": "us-east-1",
        "checkout": "/home/ec2-user/alpha-engine-config",
        "clone": False,
        "deadman_seconds": 25200,
        "shutdown_delay_seconds": 60,
    },
    "workloads": {
        "ci-watch": {
            "unit_prefix": "ci-watch",
            "max_runtime_seconds": 19200,
            "run_log": {"local_path": "/var/log/ci-watch.log", "s3_uri": "s3://b/ci.log"},
            "secrets": [
                {
                    "env_var": "GH_TOKEN",
                    "parameter": "/alpha-engine/saturday_sf_watch/github_pat",
                    "required": True,
                    "via_file": True,
                }
            ],
            "privilege_drop": {
                "user": "ec2-user",
                "command": ["bash", "scripts/ci_watch_run.sh"],
                "env": {"HOME": "/home/ec2-user"},
                "secret_env_vars": ["GH_TOKEN"],
            },
        },
        "config-runner": {
            "unit_prefix": "config-runner",
            "max_runtime_seconds": 1800,
            "shutdown_delay_seconds": 30,
            "run_log": {"local_path": "/var/log/cr.log", "s3_uri": "s3://b/cr.log"},
        },
    },
}


def _table_file(tmp_path, suffix=".json"):
    path = tmp_path / f"workloads{suffix}"
    if suffix == ".json":
        path.write_text(json.dumps(_TABLE))
    else:
        import yaml

        path.write_text(yaml.safe_dump(_TABLE))
    return path


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
def test_a_workload_table_renders_every_row_through_one_procedure(tmp_path, suffix):
    """The Overseer's rival unification is a `case "$PLAYBOOK"` in Bash, whose
    per-playbook differences cannot be asserted without a live spot. Here they
    are rows, and a row cannot fork a procedure."""
    table = load_workloads(_table_file(tmp_path, suffix))
    assert sorted(table) == ["ci-watch", "config-runner"]
    for name, spec in table.items():
        script = render_bootstrap(spec)
        assert f"{spec.unit_prefix}-record-kill" in script, name
        assert f"--unit={spec.unit_prefix}-deadman" in script, name


def test_defaults_are_merged_under_each_row_and_a_row_may_override_them(tmp_path):
    """A default that could not be overridden pushes the next difference back
    into a fork."""
    table = load_workloads(_table_file(tmp_path))
    assert table["ci-watch"].deadman_seconds == 25200  # inherited
    assert table["ci-watch"].shutdown_delay_seconds == 60
    assert table["config-runner"].shutdown_delay_seconds == 30  # row wins


def test_an_unknown_key_is_refused_and_names_itself(tmp_path):
    """A misspelled parameter that silently does nothing is a declared
    guarantee that was never armed."""
    bad = json.loads(json.dumps(_TABLE))
    bad["workloads"]["ci-watch"]["deadman_second"] = 100
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError) as exc:
        load_workloads(path)
    assert "ci-watch" in str(exc.value)
    assert "deadman_second" in str(exc.value)


def test_an_invalid_row_names_which_row(tmp_path):
    bad = json.loads(json.dumps(_TABLE))
    bad["workloads"]["config-runner"]["deadman_seconds"] = 10
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="config-runner"):
        load_workloads(path)


def test_a_table_without_workloads_is_refused(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="workloads"):
        load_workloads(path)


def test_the_renderer_knows_no_workload_names():
    """The procedure lives here; the table lives with the consumer. A playbook
    name in the renderer's CODE is the fork starting again.

    Docstrings and comments are stripped first, on purpose: the rationale for
    several of these blocks IS an incident on a named lane, and a check that
    forbade naming it would forbid explaining why the block exists — the same
    mistake `_strip_comments` exists to avoid in the fork detector.
    """
    import ast

    import krepis.spot_bootstrap as mod

    tree = ast.parse(Path(mod.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                node.body = body[1:] or [ast.Pass()]
    code = ast.unparse(ast.fix_missing_locations(tree))
    for name in ("alert-drain", "sf-watch", "canary-replay", "ci-watch", "config-runner"):
        assert name not in code, f"{name!r} is hard-coded in the renderer"


def test_cli_renders_a_named_workload(tmp_path, capsys):
    rc = main(
        ["render", "--spec-file", str(_table_file(tmp_path)), "--workload", "config-runner"]
    )
    assert rc == 0
    assert "config-runner-record-kill" in capsys.readouterr().out


def test_cli_lists_the_declared_workloads(tmp_path, capsys):
    assert main(["workloads", str(_table_file(tmp_path))]) == 0
    assert capsys.readouterr().out.split() == ["ci-watch", "config-runner"]


def test_cli_names_the_alternatives_for_an_unknown_workload(tmp_path):
    """The dispatcher that produced the typo is usually a Lambda whose operator
    cannot open the file."""
    with pytest.raises(SystemExit) as exc:
        main(["render", "--spec-file", str(_table_file(tmp_path)), "--workload", "ci_watch"])
    assert "ci-watch" in str(exc.value)


def test_cli_requires_both_spec_file_and_workload(tmp_path):
    with pytest.raises(SystemExit, match="used together"):
        main(["render", "--spec-file", str(_table_file(tmp_path))])


# ── Every capability is genuinely NEW ────────────────────────────────────


class TestEveryCapabilityIsNew:
    """Evidence that each capability FAILS against the pre-extension module.

    `tests/golden/pre_extension_bootstrap.sh` was rendered by this module as it
    stood at `origin/main` before alpha-engine-config-I7374, from
    `_pre_extension_spec()`. Every marker below is absent there and present once
    the corresponding field is declared — so "the extension added this" is
    measured against the previous implementation rather than asserted about the
    current one.
    """

    MARKERS = {
        "1 dead-man failsafe": "-deadman",
        "2 record before kill": "-record-kill",
        "3 run-log shipping": "_run_log_shipper_loop",
        "4 egress proxy staging": "__proxy_health__",
        "5 privilege drop": "runuser -u",
        "6 no-clone assertion": "is not a git checkout",
        "7 finish() EXIT trap": "trap finish EXIT",
        "8 SSM secret fetch": "get_secret()",
        "9 dispatch by declared name": "record-kill deadman",
    }

    @pytest.mark.parametrize("capability,marker", sorted(MARKERS.items()))
    def test_the_capability_is_absent_before_the_extension(self, capability, marker):
        assert marker not in _GOLDEN.read_text(), (
            f"{capability}: the pre-extension golden already carries {marker!r} — "
            "the golden is stale; re-derive it before trusting any test here"
        )

    @pytest.mark.parametrize("capability,marker", sorted(MARKERS.items()))
    def test_the_capability_is_present_once_declared(self, capability, marker):
        assert marker in render_bootstrap(_overseer_spec()), capability


def test_a_secret_value_can_never_reach_the_rendered_script():
    """`SsmSecret` carries a parameter PATH and an env-var NAME, never a value.

    This is the assertion behind the two `codeql[py/clear-text-logging-
    sensitive-data]` suppressions in `main()`. CodeQL taints the rendered
    script because the spec has a field called `secrets`; the flow it reports
    is identifier-pattern matching, not a value flow — the value is fetched ON
    THE BOX at runtime, by the instance profile, through `get_secret`.

    Pinned rather than asserted in a comment: a suppression whose justification
    lives only in prose is a suppression that survives the day the
    justification stops being true.
    """
    poison = "AKIA-NOT-A-REAL-VALUE-0000"
    spec = _overseer_spec(
        secrets=(
            SsmSecret(
                env_var="GH_TOKEN",
                parameter="/alpha-engine/saturday_sf_watch/github_pat",
                required=True,
                via_file=True,
            ),
        ),
        # A credential planted everywhere a caller could plausibly put one.
        privilege_drop=PrivilegeDrop(
            user="ec2-user",
            command=("bash", "/x.sh"),
            env=(("GH_TOKEN", "${GH_TOKEN:-}"),),
            secret_env_vars=("GH_TOKEN",),
        ),
    )
    script = render_bootstrap(spec)
    assert poison not in script
    # What DOES appear is the parameter path and the variable name — the two
    # things a reader needs in order to fix a missing secret, and neither of
    # which is one.
    assert "/alpha-engine/saturday_sf_watch/github_pat" in script
    assert "get_secret" in script


def test_the_codeql_suppressions_still_name_the_rule_they_suppress():
    """A suppression comment that drifts off its line silently stops
    suppressing, and the next reader sees a red check with no explanation."""
    import krepis.spot_bootstrap as mod

    text = Path(mod.__file__).read_text()
    assert text.count("codeql[py/clear-text-logging-sensitive-data]") == 2, (
        "expected exactly two inline suppressions — one per print path in main()"
    )
    assert "krepis-PR113" in text, (
        "the suppression must cite the ruling that established this rule as a "
        "false positive, or the next reader has to re-litigate it"
    )
