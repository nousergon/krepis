"""Render the remote bootstrap script for an EC2 spot workload.

The last stage of the spot lifecycle that was not already a krepis module,
and the only one that duplicated. `ec2_spot` launches, `ssm_dispatcher`
dispatches, `ssm_log_capture` ships logs, `heartbeat` reports liveness — and
the bootstrap, the script those dispatchers actually send, lived as a Bash
heredoc copied into two repos that then diverged (alpha-engine-config-I6922).

## What that cost, measured

`crucible-predictor/infrastructure/_spot_common.sh` and
`nousergon-data/infrastructure/_spot_common.sh` were created eight days apart
by independently applying the same written standard (ARCHITECTURE.md §111),
which specified the shape and named no implementation. Neither was forked from
the other; 9 of their ~12 functions share a name.

On 2026-08-11 the weekly pipeline failed three consecutive times, each on a
different defect inside this one script, each revealed only once the previous
was fixed:

| Run | Failure | Fixed by |
|---|---|---|
| `watch-rerun-2026-08-10-5` | `rc=137` at exactly PT5M0.0s — `systemctl start` blocked forever on a `Type=oneshot` unit whose `ExecStart` never exits | `crucible-predictor#461` |
| `-6` | `ERROR: python3.12 not found` — the bootstrap ASSERTED an interpreter the AL2023 AMI does not ship | `#462` |
| `-7` | `fatal: repository '' does not exist` — `REPO_URL` documented as exported into the heredoc, never actually exported | `#463` |

The first two had already been fixed in `nousergon-data`'s copy (`#1294`,
`#1296`) 16 and 23 hours earlier. Each fix was hand-carried, and each port cost
a failed weekly run to discover. Three ports in two days is why this module
exists rather than a fourth.

## What is shared and what is not

Everything up to the clone is byte-identical across both repos once their
fixes converge, and it is where all three defects lived:

1. preamble — `set -eo pipefail` plus the environment every workload needs
2. **watchdog** — the systemd unit, its supervision loop, and a guarded enable
3. **interpreter** — install, then assert as a POST-condition

Only the tail differs, and only in data:

4. clone — which repo, at which checkout path
5. config staging — which S3 keys land at which paths

So the parameters are a repo, a checkout, and a list of config copies. Nothing
about the hard part is repo-specific, which is exactly why it should never
have been copied.

## Why render rather than execute

This module emits the script; `ssm_dispatcher` sends it. Keeping those
separate means the output is a pure function of its inputs — testable without
AWS, diffable in review, and assertable against the defects above. A module
that both rendered and dispatched would need a live spot to test the thing
most worth testing.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path

__all__ = [
    "BOOTSTRAP_SIGNATURES",
    "MIN_CATEGORIES",
    "Clone",
    "ConfigCopy",
    "EgressProxy",
    "InlineBootstrap",
    "PrivilegeDrop",
    "RunLog",
    "SpotBootstrapSpec",
    "SsmSecret",
    "load_workloads",
    "render_bootstrap",
    "render_install_deps",
    "scan_for_inline_bootstraps",
]

#: Interpreter the fleet's `requirements.txt` files are resolved against. A
#: silent fall back to the AMI's system python3 is drift, not resilience —
#: the wheels differ.
PYTHON = "python3.12"

#: Seconds `systemctl enable --now` is allowed before the bootstrap gives up.
#: The guard exists because a misdeclared unit blocks FOREVER: run 5 above
#: consumed its entire 300s SSM budget and died under SIGKILL with no output,
#: which read as "bootstrap is slow" rather than "systemctl is blocked". A
#: bounded failure with a message is worth more than the 60s it costs.
SYSTEMCTL_ENABLE_TIMEOUT_SEC = 60


@dataclass(frozen=True)
class ConfigCopy:
    """One ``aws s3 cp`` from the staging prefix onto the spot.

    ``dest`` is an absolute path on the instance. It must be a path the
    workload's own config resolver actually searches — staging a file
    somewhere plausible but unsearched is a `FileNotFoundError` at workload
    start, not at bootstrap (alpha-engine-config-I6846: the per-stage split
    moved `config.yaml` to `/home/ec2-user/data/config/`, which
    `resolve_experiment_config` does not look in, and MorningEnrich died on
    it for two days). Assert the destination against the resolver in the
    consuming repo's tests; this module cannot know the resolver.
    """

    source_name: str
    dest: str
    #: Directory to create first. Defaults to ``dest``'s parent.
    mkdir: str | None = None
    #: ``chown -R ec2-user:ec2-user`` this path after the copy. Needed when
    #: the destination tree is created by root during bootstrap but read by
    #: ec2-user during the workload.
    chown: str | None = None
    #: Shell condition gating this copy, e.g. ``"${STAGED_PREDICTOR_CONFIG}"``.
    #: Rendered as ``if [ <when> = "1" ]``, with an ``else`` that SAYS the copy
    #: was skipped. An optional artifact that vanishes silently is
    #: indistinguishable from one that failed to copy, and the workload then
    #: dies on a `FileNotFoundError` several minutes and one process later.
    #: The condition is a launcher-side value that must reach the spot through
    #: :attr:`SpotBootstrapSpec.exports`, like every other interpolation.
    when: str | None = None

    def parent(self) -> str:
        if self.mkdir:
            return self.mkdir
        return self.dest.rsplit("/", 1)[0] or "/"


@dataclass(frozen=True)
class Clone:
    """One additional repository checkout on the spot.

    A workload that needs a sibling repo on ``sys.path`` clones more than one
    — `crucible-backtester` clones three (itself, `crucible-executor`,
    `crucible-predictor`) because its predictor replay runs the predictor's
    modules in-process. Modelling that as data is what stops the third repo
    from being a reason to keep a private fork of the whole bootstrap.
    """

    repo_url: str
    checkout: str
    branch: str | None = None


@dataclass(frozen=True)
class RunLog:
    """Capture this run's whole output and keep it after the box is gone.

    An SSM command invocation is not a run log: its output is truncated and it
    expires, so a workload whose only record is the invocation cannot be
    post-mortemed at all — which is the state
    `alpha-engine-config/infrastructure/config_runner_spot_bootstrap.sh` has
    shipped in since it was written (alpha-engine-config-I7374).

    Three properties, each of which the fleet has already paid to learn
    (alpha-engine-config-I5512):

    1. **Flush before reading.** ``tee`` behind a process substitution is a
       background writer holding bytes in a pipe. A ``[ -s "$FILE" ]`` guard
       read at trap time is FALSE while the file is non-empty milliseconds
       later, so the guard skipped the upload entirely and the S3 prefix stayed
       empty for weeks while every run "shipped".
    2. **Never silent.** The outcome is logged either way. Non-fatal, because
       the failure being fixed is silence, not the fail-open.
    3. **Ship periodically, not only at exit.** An exit-time-only write cannot
       survive the failure mode it exists for. A box killed mid-run leaves a
       log at most ``ship_interval_seconds`` stale instead of nothing.
    """

    local_path: str
    s3_uri: str
    #: Seconds between background pushes to the SAME key.
    ship_interval_seconds: int = 60
    #: Tenths of a second the flush waits for ``tee`` to drain before giving up.
    flush_wait_deciseconds: int = 20


@dataclass(frozen=True)
class SsmSecret:
    """One SSM parameter fetched on the box, by the instance profile.

    ``via_file`` routes the value into the 0600 env file the privilege-drop
    child sources and unlinks, instead of across ``runuser -- env`` where it
    would be visible in the child's argv (alpha-engine-config-I4949/I4956).
    Any value that is a credential belongs there.
    """

    env_var: str
    parameter: str
    #: An absent value is FATAL. Default False: several fleet secrets are
    #: genuinely optional and their consumers degrade with a stated WARN.
    required: bool = False
    via_file: bool = False


@dataclass(frozen=True)
class EgressProxy:
    """A local content-scanning LLM proxy, staged and health-checked per run.

    Per-workload port and staging directory, never constants: co-tenant
    workloads on one host collide otherwise, and a collision presents as one
    workload silently reading another's upstream.

    Fail-closed on both edges. A missing scanner binary and an unhealthy proxy
    both abort the bootstrap — an agent that reaches its upstream without the
    scan in front of it is the exact outcome the proxy exists to prevent, and
    "the proxy did not come up" must never degrade into "ran ungated".
    """

    port: int
    stage_dir: str
    #: Directory ON THE BOX holding the proxy's own files, staged by the clone
    #: or a config copy. Named rather than embedded: krepis renders the
    #: procedure, the consumer owns the proxy implementation.
    source_dir: str
    upstream_host: str
    #: Environment variable the proxy reads its upstream credential from.
    api_key_env: str
    #: SSM parameter holding that credential.
    api_key_parameter: str
    upstream_prefix: str = ""
    files: "tuple[str, ...]" = ("llm_egress_proxy.py",)
    health_path: str = "/__proxy_health__"
    log_path: "str | None" = None
    #: Variables set to ``http://127.0.0.1:<port>`` once the proxy is healthy.
    base_url_envs: "tuple[str, ...]" = ("ANTHROPIC_BASE_URL",)
    #: Binaries that must be present before the proxy may start. Absent ⇒ abort.
    required_binaries: "tuple[str, ...]" = ()
    health_attempts: int = 10
    extra_args: "tuple[str, ...]" = ()


@dataclass(frozen=True)
class PrivilegeDrop:
    """Run the workload as a non-root user with an EXPLICIT env allow-list.

    ``runuser -- /usr/bin/env NAME=VALUE …`` passes ONLY what is named. That is
    the point and also the hazard: anything omitted reaches the workload UNSET,
    and a script reading an unset variable mostly degrades quietly rather than
    failing. On 2026-07-27 an auth token never crossed this boundary and the
    alert-drain lane failed for five days looking like a provider outage.

    So the allow-list is data on the spec, diffable in review, and asserted in
    tests against what the workload actually reads.

    ``env`` values are rendered as double-quoted shell words, so
    ``"${GH_TOKEN:-}"``-style expansion works. A value carrying ``"``, a
    backtick or ``$(`` is REJECTED at render time rather than emitted: command
    substitution inside the privilege-drop line runs as ROOT, before the drop.
    """

    user: str
    #: argv of the workload, run as ``user``.
    command: "tuple[str, ...]"
    #: ``NAME -> shell word``. Ordered as given; the render preserves order so
    #: a diff of two renders is readable.
    env: "tuple[tuple[str, str], ...]" = ()
    #: Names staged into the 0600 env file the child sources then unlinks.
    #: Their values must already be in this shell (usually from an
    #: :class:`SsmSecret` with ``via_file=True``).
    secret_env_vars: "tuple[str, ...]" = ()
    #: Paths handed to ``user`` (``chown -R``) before the drop. Root cloned
    #: them; the workload writes them.
    chown: "tuple[str, ...]" = ()
    #: Prefix for the mktemp template of the secret env file.
    secret_file_prefix: str = "spot-secret-env"


_ENV_VALUE_FORBIDDEN = re.compile(r'["`]|\$\(')


@dataclass(frozen=True)
class SpotBootstrapSpec:
    """Everything that differs between one spot workload and another."""

    repo_url: str = ""
    checkout: str = ""
    #: Shell variable references are expanded ON THE SPOT, so anything the
    #: script interpolates must reach it through ``exports``. The predictor's
    #: `REPO_URL` was interpolated but never exported, and the clone ran
    #: against an empty string for a day (crucible-predictor#463). Passing the
    #: URL as a literal here removes that class entirely.
    config_copies: tuple[ConfigCopy, ...] = ()
    #: Extra ``KEY=value`` pairs exported before the script body.
    exports: dict[str, str] = field(default_factory=dict)
    branch: str = "main"
    region: str = "us-east-1"
    #: Repos cloned in ADDITION to the primary one, in order.
    extra_clones: tuple[Clone, ...] = ()
    #: Hard runtime cap. Arms a transient ``systemd-run --on-active`` timer
    #: that powers the instance off after this many seconds, whatever the
    #: workload is doing.
    #:
    #: This is a SEPARATE guarantee from the ``ec2-spot-watchdog`` unit, not an
    #: alternative to it, and the two are always rendered together. The unit
    #: answers "the SSM agent died, so nothing can ever reach this box again";
    #: the timer answers "the workload itself hung". `crucible-backtester`
    #: carried only the timer and `nousergon-data`/`crucible-predictor` only
    #: the unit, so each fork was uncovered against the other's failure mode —
    #: which is the per-copy divergence this module exists to end, showing up
    #: as a missing guarantee rather than as a bug.
    max_runtime_seconds: int | None = None

    # ── Overseer-substrate capabilities (alpha-engine-config-I7374) ──────────
    #
    # Every field below defaults to OFF and renders nothing, so a spec written
    # before they existed renders byte-identically — asserted against a frozen
    # golden in tests/golden/pre_extension_bootstrap.sh. That is not politeness:
    # `krepis.spot_bootstrap` acquired live consumers in the data plane on
    # 2026-08-14, and this extension landed under them.

    #: Prefix for every systemd unit this spec renders. Per-workload, because
    #: two workloads on one host would otherwise fight over a unit name.
    unit_prefix: str = "ec2-spot"

    #: **Tier one of two: the dead-man.** Armed before the script can fail,
    #: above ``max_runtime_seconds``, and cancelled by the exit trap. A
    #: bootstrap that dies before arming its real cap otherwise leaves a box
    #: running with no watchdog, no marker and no cost ceiling — measured
    #: 2026-07-28 as 11 ci-watch plus 1 sf-watch boxes alive with dead
    #: charters. ``max_runtime_seconds`` is tier two and answers a different
    #: question (the workload hung); the SSM-liveness unit is a third and
    #: answers a third (nothing can reach this box again). None replaces
    #: another.
    deadman_seconds: "int | None" = None

    #: Extra command a terminating timer runs BEFORE powering the box off,
    #: with ``KILL_REASON`` exported. See :func:`render_bootstrap` for why
    #: every terminating timer routes through one recorder.
    record_before_kill: "str | None" = None

    run_log: "RunLog | None" = None

    #: SSM parameters fetched on the box. Order is preserved.
    secrets: "tuple[SsmSecret, ...]" = ()

    egress_proxy: "EgressProxy | None" = None

    privilege_drop: "PrivilegeDrop | None" = None

    #: Deferred self-terminate from the EXIT trap. Shutting the box down
    #: IMMEDIATELY on exit races the SSM agent's own final-status callback:
    #: when shutdown wins, the control plane reports Failed/Undeliverable for
    #: a command that genuinely succeeded, and anything trusting SSM's terminal
    #: status as ground truth reports a false failure
    #: (alpha-engine-config#1472). Sleeping does NOT fix it — the agent only
    #: starts its callback after the monitored process exits, so a blocking
    #: sleep merely delays when the race begins. Scheduling the shutdown out of
    #: band and exiting immediately is what decouples the two.
    shutdown_delay_seconds: "int | None" = None

    #: Extra command the EXIT trap runs after the log ship and before the
    #: shutdown is scheduled — the workload's own run record, lane telemetry,
    #: completion marker. ``rc`` is in scope.
    finish_hook: "str | None" = None

    #: **No-clone mode.** Six of the seven Overseer bootstraps never clone:
    #: their dispatcher's SSM prelude already placed the checkout, and they
    #: derive a repo root from ``$0``. Deriving is not asserting — a prelude
    #: that half-failed leaves a path that exists and is wrong. With
    #: ``clone=False`` the checkout is ASSERTED as a precondition instead.
    clone: bool = True

    def __post_init__(self) -> None:
        if not self.checkout:
            raise ValueError("checkout is required (the workload's path on the spot)")
        if self.clone and not self.repo_url:
            raise ValueError(
                "repo_url is required when clone is True; pass clone=False for a "
                "workload whose checkout is placed by its dispatcher"
            )
        if (
            self.deadman_seconds is not None
            and self.max_runtime_seconds is not None
            and self.deadman_seconds <= self.max_runtime_seconds
        ):
            raise ValueError(
                f"deadman_seconds ({self.deadman_seconds}) must sit ABOVE "
                f"max_runtime_seconds ({self.max_runtime_seconds}) — a failsafe "
                "at or below the real cap truncates legitimate runs and reports "
                "them as dead-man kills"
            )
        if self.privilege_drop is not None:
            for name, value in self.privilege_drop.env:
                if _ENV_VALUE_FORBIDDEN.search(value):
                    raise ValueError(
                        f"privilege-drop env {name}={value!r} contains a quote or "
                        "command substitution — that would execute as ROOT, before "
                        "the drop"
                    )
            if not self.privilege_drop.command:
                raise ValueError("privilege_drop.command is empty — nothing would run")


def _quote(value: str) -> str:
    return shlex.quote(value)


def _watchdog_block() -> str:
    """systemd watchdog — self-terminate if the SSM agent stops.

    ``Type=simple``, never ``oneshot``: ``ExecStart`` is an endless
    supervision loop, and ``systemctl start`` on a ``Type=oneshot`` unit
    blocks until ``ExecStart`` exits, with ``TimeoutStartSec`` defaulting to
    infinity. The unit shipped as ``oneshot`` + ``RemainAfterExit=yes`` and
    therefore hung every bootstrap that used it.
    """
    return f"""
if ! systemctl is-enabled ec2-spot-watchdog 2>/dev/null; then
  cat > /etc/systemd/system/ec2-spot-watchdog.service <<'UNIT'
[Unit]
Description=EC2 Spot Watchdog — self-terminate on SSM agent stoppage
After=amazon-ssm-agent.service
Requires=amazon-ssm-agent.service

[Service]
Type=simple
ExecStart=/usr/local/bin/ec2-spot-watchdog.sh
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
UNIT
  cat > /usr/local/bin/ec2-spot-watchdog.sh <<'WDSH'
#!/usr/bin/env bash
set -euo pipefail
while true; do
  if ! systemctl is-active amazon-ssm-agent >/dev/null 2>&1; then
    sleep 60
    if ! systemctl is-active amazon-ssm-agent >/dev/null 2>&1; then
      shutdown -h now
    fi
  fi
  sleep 60
done
WDSH
  chmod +x /usr/local/bin/ec2-spot-watchdog.sh
  timeout {SYSTEMCTL_ENABLE_TIMEOUT_SEC} systemctl enable --now ec2-spot-watchdog || {{
    echo "ERROR: enabling ec2-spot-watchdog did not return within {SYSTEMCTL_ENABLE_TIMEOUT_SEC}s — the unit is misdeclared (an endless ExecStart under Type=oneshot blocks systemctl start forever)" >&2
    exit 1
  }}
fi
""".strip()


def _interpreter_block() -> str:
    """Install the interpreter, then assert it as a POST-condition.

    The assertion came first historically and the install was absent — an AMI
    contract nothing provides. The order matters: asserting before installing
    is a precondition on an image we do not build, and the fleet has now paid
    for that twice.
    """
    return f"""
dnf install -y -q {PYTHON} {PYTHON}-pip {PYTHON}-devel git gcc 2>/dev/null || \\
    dnf install -y -q python3 python3-pip python3-devel git gcc
command -v {PYTHON} >/dev/null || {{ echo "ERROR: {PYTHON} not found after dnf install" >&2; exit 1; }}
echo "Using: $({PYTHON} --version)"
""".strip()


def _hard_timeout_block(spec: SpotBootstrapSpec) -> str:
    """Power the box off after ``max_runtime_seconds``, whatever is running.

    A transient timer rather than a unit file: it needs no ``[Install]``
    section, no ``daemon-reload`` and no idempotence guard, and it dies with
    the instance. ``systemd-run`` failing is fatal — an uncapped spot whose
    workload hangs runs until somebody notices the bill, and "the cap could
    not be armed" is exactly the condition under which the run must not
    start.

    Since alpha-engine-config-I7374 this routes through the kill recorder when
    the spec has anything to record, so the cap firing is distinguishable from
    a clean exit rather than being the same silence.
    """
    seconds = int(spec.max_runtime_seconds or 0)
    return _terminating_timer(
        spec,
        unit=f"{spec.unit_prefix}-hard-timeout",
        seconds=seconds,
        description=f"spot hard runtime cap ({seconds}s)",
        reason="budget_exhausted",
        on_failure=(
            f'  echo "ERROR: could not arm the {seconds}s hard-timeout timer — '
            'refusing to start an uncapped spot workload" >&2\n'
            "  exit 1"
        ),
    )


def _one_clone(repo_url: str, checkout: str, branch: str) -> str:
    q = _quote(checkout)
    return (
        f"if [ ! -d {q}/.git ]; then\n"
        f"  rm -rf {q}\n"
        f"  git clone --depth 1 --branch {_quote(branch)} {_quote(repo_url)} {q}\n"
        f"fi"
    )


def _clone_block(spec: SpotBootstrapSpec) -> str:
    """Clone the workload repo and any siblings, idempotently.

    The URL and branch are baked in as literals rather than interpolated from
    the spot's environment — see :class:`SpotBootstrapSpec`.
    """
    if not spec.clone:
        # No-clone mode. The dispatcher's SSM prelude placed this checkout; the
        # bootstrap ASSERTS it rather than deriving a path from $0 and trusting
        # it. A prelude that half-failed leaves a directory that exists and is
        # not a checkout, which is indistinguishable from success to a
        # `dirname $0` — and the workload then fails several minutes later on a
        # missing file, in another process.
        q = _quote(spec.checkout)
        return (
            f"if [ ! -d {q}/.git ]; then\n"
            f'  echo "ERROR: {spec.checkout} is not a git checkout — this workload '
            'runs in no-clone mode, so its dispatcher was supposed to place one" >&2\n'
            "  exit 1\n"
            "fi"
        )
    blocks = [_one_clone(spec.repo_url, spec.checkout, spec.branch)]
    for clone in spec.extra_clones:
        blocks.append(
            _one_clone(clone.repo_url, clone.checkout, clone.branch or spec.branch)
        )
    return "\n".join(blocks)


def _one_config_copy(copy: ConfigCopy, region: str) -> "list[str]":
    lines = [f"mkdir -p {_quote(copy.parent())}"]
    lines.append(
        f'aws s3 cp "${{S3_STAGING}}/{copy.source_name}" {_quote(copy.dest)} '
        f"--region {_quote(region)} --quiet"
    )
    if copy.chown:
        lines.append(f"chown -R ec2-user:ec2-user {_quote(copy.chown)}")
    return lines


def _config_block(spec: SpotBootstrapSpec) -> str:
    if not spec.config_copies:
        return ""
    lines: list[str] = []
    for copy in spec.config_copies:
        body = _one_config_copy(copy, spec.region)
        if copy.when is None:
            lines.extend(body)
            continue
        lines.append(f'if [ "{copy.when}" = "1" ]; then')
        lines.extend(f"  {line}" for line in body)
        lines.append("else")
        lines.append(
            f'  echo "SKIPPED: {copy.source_name} not staged '
            f'(condition {copy.when} was not 1) — {copy.dest} will not exist"'
        )
        lines.append("fi")
    return "\n".join(lines)


# ── Overseer-substrate blocks (alpha-engine-config-I7374) ────────────────────


def _record_kill_path(spec: SpotBootstrapSpec) -> str:
    return f"/usr/local/sbin/{spec.unit_prefix}-record-kill"


def _wants_kill_record(spec: SpotBootstrapSpec) -> bool:
    """Does this spec have anything to record before a timer kills the box?

    A run log alone is enough. The whole point of
    alpha-engine-config-I7374's second defect is that four of seven bootstraps
    armed their watchdog straight at ``/sbin/shutdown -h now``, so a watchdog
    kill was indistinguishable from a clean exit — and every one of those four
    HAD a run log it could have stamped. Making the recorder follow the run log
    rather than an opt-in flag is what stops the next spec from reproducing it
    by omission.
    """
    return spec.run_log is not None or spec.record_before_kill is not None


def _record_kill_block(spec: SpotBootstrapSpec) -> str:
    """Write the script every terminating timer runs before powering off.

    ``shutdown -h now`` races the bootstrap's own EXIT trap, so without this a
    watchdog death and a spot reclaim are the same event in the telemetry —
    both look like "heartbeat stopped". Stamping the reason first is what
    separates "the run needed more than its budget" from "the box was taken
    away", and those two have opposite remedies.

    The script NEVER shuts the box down itself; the timer's own command does
    that unconditionally afterwards. A recorder that owned the shutdown would
    make every future edit to it a chance to strand a box.
    """
    log = spec.run_log
    lines = [
        "#!/bin/sh",
        "# Written by krepis.spot_bootstrap. Runs BEFORE a terminating timer",
        "# powers the box off. Records; never shuts down (its caller does).",
        "set +e",
        'KILL_REASON="${1:-unknown}"',
        "export KILL_REASON",
        f'logger -t {spec.unit_prefix}-kill "terminating timer fired: '
        'reason=${KILL_REASON}"',
    ]
    if log is not None:
        stamp = (
            f'printf "[{spec.unit_prefix}] TERMINATED BY TIMER reason=%s at %s\\n" '
            '"$KILL_REASON" "$(date -u +%FT%TZ)" >> '
            f"{_quote(log.local_path)}"
        )
        lines += [
            "# Stamp the run log, then push it. A box about to power off has no",
            "# later chance, and the stamp is the only thing that tells a reader",
            "# the log ends because the box was killed rather than because the",
            "# workload finished.",
            stamp,
            "sync 2>/dev/null",
            f"aws s3 cp {_quote(log.local_path)} {_quote(log.s3_uri)} "
            f"--region {_quote(spec.region)} >/dev/null 2>&1",
        ]
    if spec.record_before_kill:
        lines += ["# Workload-supplied record (spec.record_before_kill).", spec.record_before_kill]
    lines.append("exit 0")
    body = "\n".join(lines)
    path = _record_kill_path(spec)
    return (
        f"cat > {_quote(path)} <<'RECORDKILL'\n{body}\nRECORDKILL\n"
        f"chmod +x {_quote(path)}"
    )


def _terminating_timer(
    spec: SpotBootstrapSpec,
    *,
    unit: str,
    seconds: int,
    description: str,
    reason: str,
    on_failure: str,
) -> str:
    """One ``systemd-run`` timer that powers the box off after ``seconds``.

    Transient rather than a unit file: no ``[Install]``, no ``daemon-reload``,
    no idempotence guard, and it dies with the instance.

    The shutdown is emitted unconditionally after the recorder, so a recorder
    that fails cannot strand a running box — the failure mode a
    record-before-kill hook must not introduce.
    """
    if _wants_kill_record(spec):
        rec = _record_kill_path(spec)
        command = (
            f"/bin/sh -c '{rec} {shlex.quote(reason)} >/dev/null 2>&1; "
            "/sbin/shutdown -h now'"
        )
    else:
        command = "/sbin/shutdown -h now"
    return (
        f"systemd-run --on-active={int(seconds)} --unit={unit} \\\n"
        f"    --description={_quote(description)} {command} || {{\n"
        f"{on_failure}\n"
        "}"
    )


def _deadman_block(spec: SpotBootstrapSpec) -> str:
    seconds = int(spec.deadman_seconds or 0)
    unit = f"{spec.unit_prefix}-deadman"
    timer = _terminating_timer(
        spec,
        unit=unit,
        seconds=seconds,
        description=f"{spec.unit_prefix} dead-man self-terminate "
        "(fires only if the real cap never arms)",
        reason="deadman",
        on_failure=(
            f'  echo "WARN: [{spec.unit_prefix}] dead-man could not be armed — a '
            'crash before the runtime cap arms would leave this box running" >&2'
        ),
    )
    return (
        "# Tier one of two. Armed before anything below can fail, and above the\n"
        "# runtime cap so it never truncates a legitimate run. WARN rather than\n"
        "# fatal: a failsafe that refuses to start the run it protects has\n"
        "# turned a partial guarantee into an outage.\n" + timer
    )


def _run_log_block(spec: SpotBootstrapSpec) -> str:
    log = spec.run_log
    assert log is not None
    return f"""
# Every line of this bootstrap AND its child is captured here and shipped to
# S3, so the record survives a spot reclaim, a watchdog kill or a crash.
mkdir -p {_quote(log.local_path.rsplit("/", 1)[0] or "/")}
exec > >(tee -a {_quote(log.local_path)}) 2>&1

_flush_run_log() {{
  # `tee` behind a process substitution is a BACKGROUND writer holding bytes in
  # a pipe. At trap time it has not flushed, so a size check reads FALSE on a
  # file that is non-empty milliseconds later — which is how the previous guard
  # skipped every upload while reporting nothing.
  sync 2>/dev/null || true
  _waited=0
  while [ "$_waited" -lt {int(log.flush_wait_deciseconds)} ] && [ ! -s {_quote(log.local_path)} ]; do
    sleep 0.1
    _waited=$((_waited + 1))
  done
  sync 2>/dev/null || true
}}

_ship_run_log() {{
  _flush_run_log
  if [ ! -s {_quote(log.local_path)} ]; then
    # Genuinely empty AFTER flushing is worth saying out loud: it means the run
    # produced no output at all, which is never normal.
    echo "WARN: run-log ship skipped — {log.local_path} still empty after flush"
    return 0
  fi
  if _err="$(aws s3 cp {_quote(log.local_path)} {_quote(log.s3_uri)} --region {_quote(spec.region)} 2>&1 >/dev/null)"; then
    echo "run log shipped -> {log.s3_uri} ($(wc -c < {_quote(log.local_path)} 2>/dev/null || echo '?') bytes)"
  else
    # Loud, never fatal: the failure being fixed here is silence, not the
    # fail-open. The box must still wind down and report.
    echo "WARN: run-log ship FAILED -> {log.s3_uri}: $_err"
  fi
}}

# Periodic, not exit-only. An exit-time-only write cannot survive the failure
# mode it exists for; a box killed at any point leaves a log at most
# {int(log.ship_interval_seconds)}s stale instead of nothing at all.
_run_log_shipper_loop() {{
  while true; do
    sleep {int(log.ship_interval_seconds)}
    [ -s {_quote(log.local_path)} ] || continue
    aws s3 cp {_quote(log.local_path)} {_quote(log.s3_uri)} --region {_quote(spec.region)} >/dev/null 2>&1 || true
  done
}}
_run_log_shipper_loop &
_RUN_LOG_SHIPPER_PID=$!

_stop_run_log_shipper() {{
  [ -n "${{_RUN_LOG_SHIPPER_PID:-}}" ] && kill "${{_RUN_LOG_SHIPPER_PID}}" 2>/dev/null || true
}}

trap '_stop_run_log_shipper; _ship_run_log' TERM INT
""".strip()


def _finish_block(spec: SpotBootstrapSpec) -> str:
    """The EXIT trap: stop the shipper, ship, record, schedule shutdown, exit."""
    lines = ["finish() {", "  rc=$?"]
    if spec.deadman_seconds is not None:
        lines.append(
            f"  systemctl stop {spec.unit_prefix}-deadman.timer >/dev/null 2>&1 || true"
        )
    if spec.run_log is not None:
        lines += [
            "  # Stop the periodic shipper FIRST so it cannot race this final copy.",
            "  _stop_run_log_shipper",
            "  _ship_run_log",
        ]
    if spec.finish_hook:
        lines += ["  # spec.finish_hook — the workload's own run record.", f"  {spec.finish_hook}"]
    delay = int(spec.shutdown_delay_seconds or 0)
    lines += [
        f'  echo "exit rc=$rc — shutdown scheduled {delay}s from now; this script '
        'exits immediately so the SSM agent can report the real status first"',
        f"  systemd-run --on-active={delay} --unit={spec.unit_prefix}-delayed-shutdown \\",
        f"      --description={_quote(spec.unit_prefix + ' delayed self-terminate (post-SSM-report)')} "
        "/sbin/shutdown -h now >/dev/null 2>&1 || {",
        '    echo "WARN: delayed-shutdown scheduling failed — shutting down now" >&2',
        "    shutdown -h now >/dev/null 2>&1 || true",
        "  }",
        '  exit "$rc"',
        "}",
        "trap finish EXIT",
    ]
    return "\n".join(lines)


def _secrets_block(spec: SpotBootstrapSpec) -> str:
    lines = [
        "# Fetched by the instance profile — never passed in, never in argv.",
        "get_secret() { aws ssm get-parameter --name \"$1\" --with-decryption \\",
        f"  --query 'Parameter.Value' --output text --region {_quote(spec.region)} 2>/dev/null; }}",
    ]
    for secret in spec.secrets:
        # `|| true` on the FETCH only. Under `set -e` a failed command
        # substitution aborts the script right here, before the named check
        # below can say WHICH parameter was missing — and "the bootstrap died
        # at line 90" is not the same information as "GH_TOKEN is absent from
        # /alpha-engine/.../github_pat". The guard below is what fails.
        lines.append(
            f'{secret.env_var}="$(get_secret {_quote(secret.parameter)})" || true'
        )
        if secret.required:
            lines += [
                f'if [ -z "${{{secret.env_var}:-}}" ]; then',
                f'  echo "ERROR: {secret.env_var} missing from SSM ({secret.parameter}) — '
                'refusing to run without it" >&2',
                "  exit 1",
                "fi",
            ]
        else:
            lines += [
                f'if [ -z "${{{secret.env_var}:-}}" ]; then',
                f'  echo "WARN: {secret.env_var} absent from SSM ({secret.parameter}) — '
                'the workload will see it unset"',
                "fi",
            ]
        if not secret.via_file:
            lines.append(f"export {secret.env_var}")
    return "\n".join(lines)


def _egress_proxy_block(spec: SpotBootstrapSpec) -> str:
    proxy = spec.egress_proxy
    assert proxy is not None
    url = f"http://127.0.0.1:{int(proxy.port)}"
    log_path = proxy.log_path or f"/var/log/{spec.unit_prefix}-egress-proxy.log"
    lines: list[str] = [
        "# Per-workload port and staging dir — co-tenants on one host collide",
        "# otherwise, and a collision reads as one workload using another's",
        "# upstream. Fail-closed on both edges: no scanner and no healthy proxy",
        "# each abort the run, because reaching the upstream ungated is the",
        "# outcome this proxy exists to prevent.",
    ]
    for binary in proxy.required_binaries:
        lines += [
            f"if ! command -v {binary} >/dev/null 2>&1; then",
            f'  echo "ERROR: {binary} unavailable — refusing to run ungated '
            '(fail-closed)" >&2',
            "  exit 1",
            "fi",
        ]
    lines.append(f"mkdir -p {_quote(proxy.stage_dir)}")
    for name in proxy.files:
        lines.append(
            f"cp {_quote(proxy.source_dir.rstrip('/') + '/' + name)} "
            f"{_quote(proxy.stage_dir.rstrip('/') + '/' + name)}"
        )
    entry = proxy.stage_dir.rstrip("/") + "/" + proxy.files[0]
    lines += [
        f'_proxy_key="$(get_secret {_quote(proxy.api_key_parameter)})" || true',
        'if [ -z "$_proxy_key" ]; then',
        f'  echo "ERROR: {proxy.api_key_parameter} missing from SSM — the egress '
        'proxy cannot start" >&2',
        "  exit 1",
        "fi",
        f'echo "launching egress proxy on 127.0.0.1:{int(proxy.port)} -> {proxy.upstream_host}"',
        f'{proxy.api_key_env}="$_proxy_key" nohup {PYTHON} {_quote(entry)} \\',
        f"    --port {int(proxy.port)} --upstream-host {_quote(proxy.upstream_host)} \\",
        f"    --api-key-env {_quote(proxy.api_key_env)}"
        + (f" --upstream-prefix {_quote(proxy.upstream_prefix)}" if proxy.upstream_prefix else "")
        + ("".join(f" {_quote(a)}" for a in proxy.extra_args))
        + " \\",
        f"    > {_quote(log_path)} 2>&1 &",
        "_proxy_pid=$!",
        "unset _proxy_key",
        "_proxy_healthy=false",
        f"for _ in $(seq 1 {int(proxy.health_attempts)}); do",
        f'  if curl -fsS "{url}{proxy.health_path}" >/dev/null 2>&1; then',
        "    _proxy_healthy=true",
        "    break",
        "  fi",
        "  sleep 1",
        "done",
        'if [ "$_proxy_healthy" != "true" ]; then',
        f'  echo "ERROR: egress proxy failed its health check on {url}{proxy.health_path} '
        '— refusing to run ungated" >&2',
        '  kill "$_proxy_pid" 2>/dev/null || true',
        "  exit 1",
        "fi",
        f'echo "egress proxy healthy (pid=$_proxy_pid) — tail {log_path} on failure"',
    ]
    for name in proxy.base_url_envs:
        lines.append(f'export {name}="{url}"')
    return "\n".join(lines)


def _privilege_drop_block(spec: SpotBootstrapSpec) -> str:
    drop = spec.privilege_drop
    assert drop is not None
    lines: list[str] = []
    for path in drop.chown:
        lines.append(
            f"chown -R {_quote(drop.user + ':' + drop.user)} {_quote(path)} 2>/dev/null || \\"
        )
        lines.append(f'  echo "WARN: chown {path} -> {drop.user} failed"')
    if drop.secret_env_vars:
        lines += [
            "# Secrets reach the child through a 0600 file it sources and unlinks,",
            "# NOT through argv — `runuser -- env NAME=VALUE` is world-readable in",
            "# /proc for the life of the process (alpha-engine-config-I4949/I4956).",
            "umask 077",
            f'_secret_env_file="$(mktemp /tmp/{drop.secret_file_prefix}-XXXXXXXX)"',
            'chmod 600 "$_secret_env_file"',
        ]
        for name in drop.secret_env_vars:
            lines += [
                f'if [ -n "${{{name}:-}}" ]; then',
                f"  printf '%s=%s\\n' {_quote(name)} \"$(printf %q \"${{{name}}}\")\" "
                '>> "$_secret_env_file"',
                "fi",
            ]
        lines += [
            f'chown {_quote(drop.user)} "$_secret_env_file" 2>/dev/null || true',
            "# Drop them from THIS shell now that they are staged.",
            "unset " + " ".join(drop.secret_env_vars),
        ]
    lines += [
        "# `runuser -- env` passes ONLY what is named here. Anything omitted",
        "# reaches the workload UNSET, and a script reading an unset variable",
        "# mostly degrades quietly rather than failing — which is why this",
        "# allow-list is spec data asserted in tests, not an ad-hoc line.",
        f"runuser -u {_quote(drop.user)} -- /usr/bin/env \\",
    ]
    for name, value in drop.env:
        lines.append(f'    {name}="{value}" \\')
    if drop.secret_env_vars:
        lines += [
            '    SECRET_ENV_FILE="$_secret_env_file" \\',
            "    bash -c 'set -a; . \"$SECRET_ENV_FILE\"; set +a; rm -f \"$SECRET_ENV_FILE\"; "
            "unset SECRET_ENV_FILE; exec \"$0\" \"$@\"' \\",
        ]
    lines.append("    " + " ".join(_quote(part) for part in drop.command))
    # Reached only on success: under `set -e` a non-zero workload aborts here
    # and the EXIT trap tears the box down with that code, which is the same
    # outcome by a shorter path. Stated because the absence of an explicit
    # failure branch reads like an omission.
    lines.append("_workload_rc=$?")
    lines.append('exit "$_workload_rc"')
    return "\n".join(lines)


def render_bootstrap(spec: SpotBootstrapSpec) -> str:
    """The complete remote bootstrap script.

    A pure function of ``spec`` — no AWS calls, no clock, no environment
    reads — so the defects this module exists to prevent are assertable in a
    unit test rather than on a live spot.
    """
    exports = " ".join(
        f"{key}={_quote(value)}" for key, value in sorted(spec.exports.items())
    )
    preamble = (
        "set -eo pipefail\n"
        f"export HOME=/home/ec2-user XDG_CACHE_HOME=/tmp "
        f"AWS_REGION={spec.region} AWS_DEFAULT_REGION={spec.region}"
    )
    if exports:
        preamble += f"\nexport {exports}"

    blocks = [preamble]
    # The recorder is written BEFORE any timer is armed. A timer whose
    # ExecStart does not exist yet is a timer that fires into nothing, and the
    # box it was guarding keeps running.
    if _wants_kill_record(spec):
        blocks.append(_record_kill_block(spec))
    if spec.deadman_seconds is not None:
        blocks.append(_deadman_block(spec))
    if spec.run_log is not None:
        blocks.append(_run_log_block(spec))
    blocks.append(_watchdog_block())
    if spec.max_runtime_seconds is not None:
        blocks.append(_hard_timeout_block(spec))
    if spec.shutdown_delay_seconds is not None:
        blocks.append(_finish_block(spec))
    blocks += [_interpreter_block(), _clone_block(spec)]
    config = _config_block(spec)
    if config:
        blocks.append(config)
    if spec.secrets or spec.egress_proxy is not None:
        blocks.append(_secrets_block(spec))
    if spec.egress_proxy is not None:
        blocks.append(_egress_proxy_block(spec))
    if spec.privilege_drop is not None:
        blocks.append(_privilege_drop_block(spec))
    return "\n\n".join(blocks) + "\n"


def render_install_deps(spec: SpotBootstrapSpec) -> str:
    """``pip install -r requirements.txt`` in the checkout.

    Separate from the bootstrap because it has its own SSM budget — deps take
    minutes where the bootstrap takes ~50s, and one timeout covering both
    tells you nothing about which was slow.

    No system-python fallback: :func:`render_bootstrap` guarantees the
    interpreter, so falling back here would silently resolve wheels against a
    different version than ``requirements.txt`` was compiled for.

    The install log is kept, not piped through ``tail -1``. A pip run that
    exits 0 having quietly skipped an extra looks identical to a clean one
    through a one-line window, and that window is all a failure downstream
    leaves behind: on 2026-08-11 the predictor's spot smoke died on a missing
    ``flow_doctor`` — a strict-mode dependency named in ``requirements.txt`` —
    and the deps step's only surviving output was pip's run-as-root warning
    (config-I6949). ``pip check`` after the install names an environment whose
    dependencies do not actually resolve, which is the shape that failure
    takes when it is a resolution problem rather than a missing line.

    **A dropped extra is fatal here, not merely reported.** Surfacing it in the
    log is not enough: pip emits it as a WARNING on a zero exit, so the step
    passes and the failure lands later as an ``ImportError`` in a different
    process with the install log gone. This renderer is the fleet copy of a
    guard the predictor's ``_spot_common.sh`` already carried; it was the
    weaker of the two until config-I6922, so migrating a consumer onto it
    would have silently regressed that consumer. Raised to a superset instead.
    """
    return f"""set -eo pipefail
export HOME=/home/ec2-user XDG_CACHE_HOME=/tmp AWS_REGION={spec.region} AWS_DEFAULT_REGION={spec.region}
cd {_quote(spec.checkout)}
_pip_log=/tmp/pip-install-deps.log
if ! {PYTHON} -m pip install --no-warn-script-location -r requirements.txt > "$_pip_log" 2>&1; then
  echo "ERROR: pip install -r requirements.txt failed" >&2
  tail -80 "$_pip_log" >&2
  exit 1
fi
grep -E "^Successfully installed" "$_pip_log" || true
# A dropped extra is a BROKEN ENVIRONMENT, not a note. pip reports it as a
# WARNING on a SUCCESSFUL exit, so nothing downstream fails until an import
# does — in another process, minutes later, with the install log long gone.
# That is how config#6963 reached production: the training smoke died on
# `ModuleNotFoundError: No module named 'flow_doctor'` out of
# krepis.logging.setup_logging, from an install pip had called a success.
# AL2023 ships pip 23.2.1, which predates PEP 685 extras normalisation
# (measured boundary: 23.2.1 drops, 23.3.2 honours), so a stock-AMI bootstrap
# sits on the broken side by default and cannot rely on the resolver to be
# strict. Fail here, where the log is still in hand and the cause is one line.
if grep -E "^WARNING: .*does not provide the extra" "$_pip_log" >&2; then
  echo "ERROR: pip dropped a requested extra (above) — the environment is incomplete." >&2
  echo "       Extras must be HYPHENATED; pip <23.3 does not normalise '_' to '-'." >&2
  exit 1
fi
# Non-fatal on purpose: an inconsistent environment is reported, not raised.
# (a) The failure mode left unraised is a pre-existing AMI-baked conflict
# unrelated to this checkout, which would otherwise fail every lane on every
# run. (b) It is recorded on stdout of this SSM step, captured with the rest
# of the deps output. A resolution defect in requirements.txt still surfaces
# here in full, one step before the import that would have failed on it.
{PYTHON} -m pip check || echo "WARNING: pip check reports an inconsistent environment (above)"
"""


# ── Fork detection ───────────────────────────────────────────────────────────
#
# The seventh copy is the one nobody looks for. `alpha-engine-config-I6922`
# deliverable 4 said "search the fleet for further copies" and named the search:
# `grep -rl ec2-spot-watchdog --include=*.sh`. It found exactly two, and it was
# wrong — `crucible-backtester/infrastructure/_spot_common.sh` was a third copy,
# 908 lines, invisible to that grep because it had never adopted the watchdog
# unit whose name the grep anchored on. A search keyed to ONE marker cannot see
# a fork whose divergence is the absence of that marker, which is the divergence
# most worth seeing.
#
# So the detector is keyed on a SET of behaviours a spot bootstrap performs.
#
# A file trips it by matching two different CATEGORIES, not two patterns.
# Categories, because several patterns describe one act: a transient timer
# whose command is `shutdown -h now` matches both `systemd-transient-timer`
# and `self-terminate` while doing a single thing, and counting that as two
# made `nousergon-data/infrastructure/preflight_sweep.sh` — which arms a
# watchdog on a box somebody else already provisioned — read as a bootstrap.
# A detector with a false positive gets an exemption list, and an exemption
# list is where the next real fork goes to hide.
#
# Only supervision collapses into one category, and only because that is where
# the double-count was: installing an interpreter and cloning a checkout are
# genuinely two acts, so they stay separate and a file doing both is caught
# even when it neither supervises nor dispatches.

#: ``name -> (category, pattern)``. Adding a signature here tightens every
#: repo's detector at once, which is the point: a fleet-wide invariant with one
#: definition cannot drift between the repos that enforce it.
BOOTSTRAP_SIGNATURES = {
    "systemd-unit": ("supervision", re.compile(r"^\s*ExecStart\s*=", re.M)),
    "systemd-transient-timer": (
        "supervision", re.compile(r"\bsystemd-run\b[^\n]*--on-active")
    ),
    "self-terminate": ("supervision", re.compile(r"\bshutdown\s+-h\s+now\b")),
    "interpreter-install": (
        "interpreter",
        re.compile(r"\b(?:dnf|yum|apt-get)\s+install\b[^\n]*\bpython3"),
    ),
    "repo-clone": (
        "checkout", re.compile(r"\bgit\s+clone\b[^\n]*/home/ec2-user/")
    ),
    "ssm-remote-dispatch": (
        "dispatch",
        re.compile(r"\bkrepis\.ssm_dispatcher\b|\baws\s+ssm\s+send-command\b"),
    ),
}

#: A file that reaches the renderer is not a fork — it IS the shared copy.
#: Matched against the INVOCATION, never a mention. Both surviving forks carry
#: a comment naming `krepis.spot_bootstrap` and telling the reader to keep the
#: copies in step; a detector matching the name would read those comments as
#: proof of adoption and clear the two files the whole exercise is about.
#: Measured — that is exactly what the first cut of this scan did.
_DELEGATES = re.compile(r"-m\s+krepis\.spot_bootstrap\b")

#: Minimum distinct signature CATEGORIES before a file is called a bootstrap.
MIN_CATEGORIES = 2


def _strip_comments(text: str) -> str:
    """Drop whole-line shell comments before classifying.

    Both directions matter. A file explaining a defect it has already fixed
    must not read as carrying it — the fleet's existing guards were bitten by
    a naive substring check that reported the `Type=oneshot` explanation as
    the `Type=oneshot` bug. And a comment must not be able to CLEAR a file
    either, which is the failure above.
    """
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )


def _mask_delegate_invocations(code: str) -> str:
    """Blank the renderer INVOCATION, and only it — never the rest of the file.

    ``scan_for_inline_bootstraps`` used to ``continue`` on the first
    `_DELEGATES` match anywhere in a file (alpha-engine-config-I7378): a file
    that renders through this module and still carries a restated heredoc
    fork elsewhere read as clean, because the whole file stopped being
    evaluated the moment the sanctioned call appeared in it. The cutover this
    scanner exists to enforce made every file it converted permanently
    exempt from it.

    A shell statement invoking the renderer is usually spread across several
    physical lines via backslash continuation (``--repo-url u \\``, one flag
    per line) — every corpus example does this. So the unit masked is the
    LOGICAL line: a run of physical lines chained by a trailing ``\\``. Only
    the logical lines that themselves contain a real `_DELEGATES` match are
    blanked; a heredoc body, a second statement, or anything else in the file
    is untouched and still evaluated below. This keeps the two invariants
    that matter: a comment can never clear anything (comments are already
    gone by the time this runs), and a real invocation clears only the
    statement it IS, never the file it lives in.
    """
    lines = code.split("\n")
    n = len(lines)
    masked = list(lines)
    i = 0
    while i < n:
        group_end = i
        while lines[group_end].rstrip().endswith("\\") and group_end + 1 < n:
            group_end += 1
        group = range(i, group_end + 1)
        joined = "\n".join(lines[idx] for idx in group)
        if _DELEGATES.search(joined):
            for idx in group:
                masked[idx] = ""
        i = group_end + 1
    return "\n".join(masked)


@dataclass(frozen=True)
class InlineBootstrap:
    """One file that bootstraps a spot without going through this module."""

    path: Path
    signatures: "tuple[str, ...]"

    def __str__(self) -> str:
        return f"{self.path}: {', '.join(self.signatures)}"


def scan_for_inline_bootstraps(
    root: "Path | str",
    *,
    subdirs: "tuple[str, ...]" = ("infrastructure", "scripts", "bin"),
    suffixes: "tuple[str, ...]" = (".sh", ".bash"),
) -> "list[InlineBootstrap]":
    """Every shell file under ``root`` that bootstraps a spot inline.

    Returns the findings; raising is the caller's job, because the useful
    assertion differs by repo — most want "none", and a repo mid-cutover wants
    "none outside this declared set" for exactly as long as its cutover PR is
    open.

    Derived, not enumerated: nothing here names a file, and the classifier is
    behavioural. A copy under a new name, in a new directory, with a new
    function prefix — `crucible-backtester`'s fork renamed every function it
    shared with its twin — still matches, because what it DOES is unchanged.
    That is the property a filename list cannot have.
    """
    root = Path(root)
    findings: list[InlineBootstrap] = []
    for subdir in subdirs:
        base = root / subdir
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix not in suffixes:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                # Unreadable is not clean. A file this scan cannot open is a
                # file it cannot clear, and reporting it as absent is how a
                # detector reports green while measuring nothing.
                findings.append(InlineBootstrap(path=path, signatures=("unreadable",)))
                continue
            code = _mask_delegate_invocations(_strip_comments(text))
            hits = tuple(
                name for name, (_, pat) in sorted(BOOTSTRAP_SIGNATURES.items())
                if pat.search(code)
            )
            categories = {BOOTSTRAP_SIGNATURES[name][0] for name in hits}
            if len(categories) >= MIN_CATEGORIES:
                findings.append(InlineBootstrap(path=path, signatures=hits))
    return findings


# ── Many workloads, one procedure (alpha-engine-config-I7374 capability 9) ───
#
# The Overseer grew its own `case "$PLAYBOOK"` dispatcher — a second
# unification competing with this module, inside the repo whose forks it was
# meant to end. Two rival unifications are worse than either, and the losing
# one is the one that is Bash in a private repo, because the differences it
# carries are unassertable without a live spot.
#
# So dispatch is DATA read by this renderer, and this module knows no workload
# names: it holds the PROCEDURE, the consumer holds the TABLE. Adding a
# workload is a row, and a row cannot fork a procedure.

_NESTED = {
    "config_copies": ConfigCopy,
    "extra_clones": Clone,
    "secrets": SsmSecret,
    "run_log": RunLog,
    "egress_proxy": EgressProxy,
    "privilege_drop": PrivilegeDrop,
}
_TUPLE_FIELDS = {"config_copies", "extra_clones", "secrets"}


def _build(cls, value):
    if not isinstance(value, dict):
        raise ValueError(f"{cls.__name__} expects a mapping, got {type(value).__name__}")
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(value) - known)
    if unknown:
        # Fail loud. A misspelled parameter that silently does nothing is the
        # exact shape of a declared guarantee that was never armed — which is
        # the class this whole module exists to end.
        raise ValueError(
            f"{cls.__name__}: unknown key(s) {unknown} — known keys are "
            f"{sorted(known)}"
        )
    return cls(**{key: _coerce(cls, key, raw) for key, raw in value.items()})


def _coerce(cls, key: str, raw):
    nested = _NESTED.get(key) if cls is SpotBootstrapSpec else None
    if nested is None:
        if key in ("files", "base_url_envs", "required_binaries", "extra_args", "command", "chown", "secret_env_vars"):
            return tuple(raw)
        if key == "env":
            # A mapping preserves insertion order in the file; a list of pairs
            # is accepted too, for a table that wants a duplicate-free diff.
            if isinstance(raw, dict):
                return tuple(raw.items())
            return tuple((str(k), str(v)) for k, v in raw)
        return raw
    if key in _TUPLE_FIELDS:
        return tuple(_build(nested, item) for item in raw)
    return _build(nested, raw)


def spec_from_mapping(mapping: dict) -> SpotBootstrapSpec:
    """One :class:`SpotBootstrapSpec` from a plain mapping (JSON/YAML row)."""
    return _build(SpotBootstrapSpec, mapping)


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_workloads(source: "Path | str") -> "dict[str, SpotBootstrapSpec]":
    """Every workload declared in a JSON or YAML table, as rendered specs.

    Shape::

        defaults:
          region: us-east-1
          deadman_seconds: 28800
        workloads:
          ci-watch:
            checkout: /home/ec2-user/alpha-engine-config
            clone: false
            max_runtime_seconds: 19200

    ``defaults`` is merged UNDER each row, one level deep into nested mappings,
    so a shared dead-man or shutdown delay is declared once. A row still
    overrides any of it — the table is where differences live, and a default
    that could not be overridden would push the next difference back into a
    fork.
    """
    path = Path(source)
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        import yaml  # noqa: PLC0415 — optional at import time, required here

        doc = yaml.safe_load(text)
    else:
        doc = json.loads(text)
    if not isinstance(doc, dict) or "workloads" not in doc:
        raise ValueError(f"{path}: expected a mapping with a 'workloads' key")
    defaults = doc.get("defaults") or {}
    out: dict[str, SpotBootstrapSpec] = {}
    for name, row in doc["workloads"].items():
        try:
            out[name] = spec_from_mapping(_merge(defaults, row or {}))
        except (ValueError, TypeError) as exc:
            # Name the workload. A table-wide error message that does not say
            # WHICH row is invalid sends the reader through every row by hand.
            raise ValueError(f"workload {name!r}: {exc}") from exc
    return out


def _spec_from_args(args: argparse.Namespace) -> SpotBootstrapSpec:
    copies: list[ConfigCopy] = []
    for raw in args.config_copy or []:
        # source:dest[:chown]
        parts = raw.split(":")
        if len(parts) not in (2, 3):
            raise SystemExit(
                f"--config-copy expects source:dest[:chown], got {raw!r}"
            )
        copies.append(
            ConfigCopy(
                source_name=parts[0],
                dest=parts[1],
                chown=parts[2] if len(parts) == 3 else None,
            )
        )
    for raw in getattr(args, "config_copy_if", None) or []:
        # condition:source:dest[:chown]
        parts = raw.split(":")
        if len(parts) not in (3, 4):
            raise SystemExit(
                f"--config-copy-if expects condition:source:dest[:chown], got {raw!r}"
            )
        copies.append(
            ConfigCopy(
                when=parts[0],
                source_name=parts[1],
                dest=parts[2],
                chown=parts[3] if len(parts) == 4 else None,
            )
        )
    clones: list[Clone] = []
    for raw in getattr(args, "extra_clone", None) or []:
        # checkout=url[@branch] — keyed on the checkout because a URL contains
        # colons and a path does not contain '='.
        if "=" not in raw:
            raise SystemExit(
                f"--extra-clone expects checkout=url[@branch], got {raw!r}"
            )
        checkout, url = raw.split("=", 1)
        branch = None
        if "@" in url and not url.rstrip("/").endswith("@"):
            head, _, tail = url.rpartition("@")
            # Only a trailing @ref, never the '@' of a scp-style git remote,
            # which appears before the first '/'.
            if head and "/" in head:
                url, branch = head, tail
        clones.append(Clone(repo_url=url, checkout=checkout, branch=branch))
    exports: dict[str, str] = {}
    for raw in args.export or []:
        if "=" not in raw:
            raise SystemExit(f"--export expects KEY=value, got {raw!r}")
        key, value = raw.split("=", 1)
        exports[key] = value
    return SpotBootstrapSpec(
        repo_url=args.repo_url,
        checkout=args.checkout,
        config_copies=tuple(copies),
        exports=exports,
        branch=args.branch,
        region=args.region,
        extra_clones=tuple(clones),
        max_runtime_seconds=getattr(args, "max_runtime_seconds", None),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m krepis.spot_bootstrap",
        description=(
            "Render the remote bootstrap script for an EC2 spot workload — "
            "watchdog, interpreter, clone, staged config. The institutional "
            "replacement for the Bash heredoc copied into every spot "
            "launcher repo, which diverged and cost three hand-carried "
            "fixes in two days (alpha-engine-config-I6922)."
        ),
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    for name, help_text in (
        ("render", "Emit the bootstrap script on stdout."),
        ("render-deps", "Emit the dependency-install script on stdout."),
    ):
        p = subparsers.add_parser(name, help=help_text)
        p.add_argument(
            "--spec-file",
            help=(
                "JSON/YAML table of workloads. With --workload, every other "
                "flag is ignored: the row IS the spec."
            ),
        )
        p.add_argument(
            "--workload",
            help="Name of the row in --spec-file to render.",
        )
        p.add_argument("--repo-url", default="")
        p.add_argument("--checkout", default="", help="Absolute clone path on the spot.")
        p.add_argument("--branch", default="main")
        p.add_argument("--region", default="us-east-1")
        p.add_argument(
            "--config-copy",
            action="append",
            metavar="SOURCE:DEST[:CHOWN]",
            help=(
                "Copy SOURCE from $S3_STAGING to absolute DEST. Repeatable. "
                "DEST must be a path the workload's config resolver searches."
            ),
        )
        p.add_argument(
            "--config-copy-if",
            action="append",
            metavar="CONDITION:SOURCE:DEST[:CHOWN]",
            help=(
                "As --config-copy, but only when CONDITION evaluates to 1 on "
                "the spot. The skip is announced, never silent. Repeatable."
            ),
        )
        p.add_argument(
            "--extra-clone",
            action="append",
            metavar="CHECKOUT=URL[@BRANCH]",
            help=(
                "Clone a sibling repo in addition to the primary one, for a "
                "workload that imports another repo's modules. Repeatable."
            ),
        )
        p.add_argument(
            "--max-runtime-seconds",
            type=int,
            default=None,
            help=(
                "Arm a hard runtime cap that powers the instance off after N "
                "seconds. Rendered alongside the SSM-liveness watchdog, never "
                "instead of it."
            ),
        )
        p.add_argument(
            "--export",
            action="append",
            metavar="KEY=VALUE",
            help="Extra environment export. Repeatable.",
        )
        p.add_argument(
            "--json",
            action="store_true",
            help="Emit {\"script\": ...} instead of the bare script.",
        )

    scan = subparsers.add_parser(
        "scan",
        help="Report shell files under a repo that bootstrap a spot inline.",
    )
    scan.add_argument("root", help="Repository root to scan.")

    listing = subparsers.add_parser(
        "workloads",
        help="List the workloads declared in a spec file.",
    )
    listing.add_argument("spec_file", help="JSON/YAML workload table.")

    args = parser.parse_args(argv)

    if args.cmd == "workloads":
        for name in sorted(load_workloads(args.spec_file)):
            print(name)
        return 0

    if args.cmd == "scan":
        findings = scan_for_inline_bootstraps(args.root)
        for finding in findings:
            print(finding)
        # Exit code IS the verdict: this is meant to run in CI and in the
        # fleet sweep, and a scanner whose only output is prose gets read by
        # nobody twice.
        return 1 if findings else 0

    if args.spec_file or args.workload:
        if not (args.spec_file and args.workload):
            raise SystemExit("--spec-file and --workload are used together")
        table = load_workloads(args.spec_file)
        if args.workload not in table:
            # Name the alternatives. "unknown workload" alone sends the reader
            # to open the file, and the dispatcher that produced the typo is
            # usually a Lambda whose operator cannot.
            raise SystemExit(
                f"unknown workload {args.workload!r} — declared: "
                + ", ".join(sorted(table))
            )
        spec = table[args.workload]
    else:
        spec = _spec_from_args(args)
    script = render_bootstrap(spec) if args.cmd == "render" else render_install_deps(spec)
    if args.json:
        print(json.dumps({"script": script}))
    else:
        sys.stdout.write(script)
    return 0


if __name__ == "__main__":
    sys.exit(main())
