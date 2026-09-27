"""``render_self_starting_user_data`` (alpha-engine-config-I11597).

The rendered user-data is the whole dispatch for a self-starting box — there is
no SSM command after it — so its properties are asserted here on the string:
the job is installed verbatim, it runs as a capped oneshot that never blocks
cloud-init, and the box powers off however the job ends.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from krepis.ec2_spot import USER_DATA_MAX_BYTES
from krepis.spot_bootstrap import SELF_START_DIR, render_self_starting_user_data

_JOB = """set -uo pipefail
export THINKTANK_SPOT_RUN_TOKEN=abc123
echo "$HOME $(date -u +%Y-%m-%d)"
exec bash infrastructure/thinktank_spot_bootstrap.sh
"""


def _render(**kw) -> str:
    job = kw.pop("job_script", _JOB)
    args = dict(
        unit="alpha-engine-thinktank-run",
        description="daily Think Tank run",
        timeout_seconds=7200,
    )
    args.update(kw)
    return render_self_starting_user_data(job, **args)


def _unit_file(rendered: str) -> str:
    body = rendered.split("<<'KREPIS_SELF_START_UNIT_EOF'\n", 1)[1]
    return body.split("\nKREPIS_SELF_START_UNIT_EOF\n", 1)[0] + "\n"


def test_is_a_shell_script_bash_accepts():
    rendered = _render()
    assert rendered.startswith("#!/bin/bash\n")
    subprocess.run(["bash", "-n"], input=rendered, text=True, check=True)


def test_job_is_installed_verbatim_behind_a_quoted_heredoc():
    """Nothing in the job may be expanded by cloud-init's shell: $HOME and
    $(date) must reach the box as written, to be evaluated when the job runs."""
    rendered = _render()
    assert f"cat > {SELF_START_DIR}/alpha-engine-thinktank-run.sh <<'KREPIS_SELF_START_JOB_EOF'\n" in rendered
    assert _JOB + "KREPIS_SELF_START_JOB_EOF\n" in rendered


def test_installed_job_round_trips_through_bash(tmp_path):
    """Execute the user-data's install half against a scratch root and diff
    the file it writes against the input."""
    rendered = _render().replace(SELF_START_DIR, str(tmp_path / "lib")).replace(
        "/etc/systemd/system", str(tmp_path)
    )
    install_half = rendered.split("systemctl daemon-reload", 1)[0]
    # Neuter only the ERR trap's power-off: this runs on a developer's machine.
    install_half = install_half.replace("/sbin/shutdown -h now' ERR", "true' ERR")
    assert "/sbin/shutdown -h now' ERR" not in install_half
    subprocess.run(["bash", "-c", install_half], check=True)
    assert (tmp_path / "lib" / "alpha-engine-thinktank-run.sh").read_text() == _JOB
    assert (tmp_path / "alpha-engine-thinktank-run.service").read_text() == _unit_file(rendered)


def test_unit_is_a_capped_oneshot_that_powers_the_box_off():
    unit = _unit_file(_render(timeout_seconds=5400, stop_grace_seconds=90))
    assert "Type=oneshot\n" in unit
    assert f"ExecStart=/bin/bash {SELF_START_DIR}/alpha-engine-thinktank-run.sh\n" in unit
    # RuntimeMaxSec has no effect on a oneshot; the cap must be TimeoutStartSec.
    assert "TimeoutStartSec=5400\n" in unit
    assert "RuntimeMaxSec" not in unit
    assert "TimeoutStopSec=90\n" in unit
    assert "KillMode=control-group\n" in unit
    assert "ExecStopPost=/sbin/shutdown -h now\n" in unit
    assert "[Install]" not in unit, "never enabled: a reboot must not re-run the job"


def test_start_never_blocks_cloud_init():
    rendered = _render()
    assert "systemctl start --no-block alpha-engine-thinktank-run.service\n" in rendered
    assert "systemctl enable" not in rendered
    assert rendered.index("systemctl daemon-reload") < rendered.index("systemctl start")


def test_a_failing_user_data_powers_the_box_off():
    rendered = _render()
    assert "set -euo pipefail\n" in rendered
    trap = next(line for line in rendered.splitlines() if line.startswith("trap "))
    assert trap.endswith(" ERR") and "/sbin/shutdown -h now" in trap
    assert rendered.index(trap) < rendered.index("install -d")


@pytest.mark.skipif(shutil.which("systemd-analyze") is None, reason="needs systemd-analyze")
def test_unit_file_passes_systemd_analyze_verify(tmp_path):
    path = tmp_path / "krepis-self-start-test.service"
    path.write_text(_unit_file(_render(description="100% of the run")))
    proc = subprocess.run(
        ["systemd-analyze", "verify", str(path)], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stderr.strip() == "", proc.stderr


def test_percent_in_description_is_escaped_for_systemd():
    assert "Description=100%% done\n" in _unit_file(_render(description="100% done"))


@pytest.mark.parametrize(
    "kw, match",
    [
        ({"unit": "Bad Unit"}, "unit"),
        ({"unit": "x;rm -rf /"}, "unit"),
        ({"description": "two\nlines"}, "single"),
        ({"description": "  "}, "single"),
        ({"job_script": "  \n"}, "empty"),
        ({"job_script": "echo\nKREPIS_SELF_START_JOB_EOF\necho pwned\n"}, "terminator"),
        ({"job_script": "KREPIS_SELF_START_UNIT_EOF\n"}, "terminator"),
        ({"timeout_seconds": 0}, "timeout"),
        ({"stop_grace_seconds": 0}, "stop_grace"),
    ],
)
def test_unsafe_inputs_are_refused(kw, match):
    with pytest.raises(ValueError, match=match):
        _render(**kw)


def test_over_the_user_data_limit_says_to_fetch_instead():
    with pytest.raises(ValueError, match="fetch its body"):
        _render(job_script="echo x\n" * (USER_DATA_MAX_BYTES // 7 + 1))


def test_typical_job_leaves_most_of_the_budget():
    assert len(_render().encode()) < USER_DATA_MAX_BYTES // 4


# ── Launch record (alpha-engine-config-I5752 / I11597) ─────────────────────

_URI = "s3://alpha-engine-research/thinktank/_control/launched/2026-09-25-abc.json"
_RECORD = {
    "run_token": "abc",
    "trading_day": "2026-09-25",
    "budget_seconds": 5400,
    "note": 'quote \' dq " pct 100% bs \\ uni é',
}


def _record_block(rendered: str) -> str:
    after_unit = rendered.split("\nKREPIS_SELF_START_UNIT_EOF\n", 1)[1]
    return after_unit.split("systemctl daemon-reload", 1)[0]


def _run_record_block(tmp_path, rendered: str, *, aws_fails: bool = False):
    """Execute the record block with curl/aws shimmed on PATH; return
    (parsed record or None, stderr)."""
    import json
    import os

    shims = tmp_path / "bin"
    shims.mkdir()
    (shims / "curl").write_text(
        '#!/bin/sh\ncase "$*" in *api/token*) echo tok;; *instance-id*) echo i-0abc123;; esac\n'
    )
    out = tmp_path / "record.json"
    (shims / "aws").write_text(
        "#!/bin/sh\nexit 1\n" if aws_fails else f'#!/bin/sh\necho "$@" > {tmp_path}/argv\ncat > {out}\n'
    )
    for f in shims.iterdir():
        f.chmod(0o755)
    env = dict(os.environ, PATH=f"{shims}:{os.environ['PATH']}")
    script = "set -euo pipefail\n" + _record_block(rendered)
    proc = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    record = json.loads(out.read_text()) if out.exists() else None
    return record, proc.stderr


def test_no_launch_record_unless_asked():
    assert "aws s3 cp" not in _render()


def test_box_writes_the_launch_record_before_starting_the_job(tmp_path):
    import datetime

    rendered = _render(launch_record_uri=_URI, launch_record=_RECORD, timeout_seconds=7200)
    assert rendered.index("aws s3 cp") < rendered.index("systemctl start --no-block")
    record, _ = _run_record_block(tmp_path, rendered)
    assert record["schema"] == "krepis_self_start_launch.v1"
    assert record["unit"] == "alpha-engine-thinktank-run"
    assert record["timeout_seconds"] == 7200
    assert record["instance_id"] == "i-0abc123"
    for key, value in _RECORD.items():
        assert record[key] == value, key  # static fields survive printf verbatim
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    booted = datetime.datetime.strptime(record["booted_at"], fmt)
    deadline = datetime.datetime.strptime(record["deadline_at"], fmt)
    assert (deadline - booted).total_seconds() == 7200
    argv = (tmp_path / "argv").read_text()
    assert f"s3 cp - {_URI} --region us-east-1" in argv


def test_a_failed_record_write_is_reported_and_the_job_still_starts(tmp_path):
    rendered = _render(launch_record_uri=_URI, launch_record=_RECORD)
    record, stderr = _run_record_block(tmp_path, rendered, aws_fails=True)
    assert record is None
    assert f"launch record NOT written to {_URI}" in stderr


def test_the_rendering_is_clock_free():
    """A replayed launch must present byte-identical user-data, or EC2 refuses
    its ClientToken. Nothing time-dependent may be rendered on this side."""
    a = _render(launch_record_uri=_URI, launch_record=_RECORD)
    b = _render(launch_record_uri=_URI, launch_record=dict(_RECORD))
    assert a == b


@pytest.mark.parametrize(
    "kw, match",
    [
        ({"launch_record_uri": "https://example.com/x"}, "s3://"),
        ({"launch_record_uri": "s3://bucket/key with space"}, "s3://"),
        ({"launch_record_uri": _URI, "launch_record": {"instance_id": "x"}}, "box-filled"),
        ({"launch_record": {"a": 1}}, "without launch_record_uri"),
    ],
)
def test_launch_record_inputs_are_checked(kw, match):
    with pytest.raises(ValueError, match=match):
        _render(**kw)
