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
import shlex
import sys
from dataclasses import dataclass, field

__all__ = [
    "ConfigCopy",
    "SpotBootstrapSpec",
    "render_bootstrap",
    "render_install_deps",
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

    def parent(self) -> str:
        if self.mkdir:
            return self.mkdir
        return self.dest.rsplit("/", 1)[0] or "/"


@dataclass(frozen=True)
class SpotBootstrapSpec:
    """Everything that differs between one spot workload and another."""

    repo_url: str
    checkout: str
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


def _clone_block(spec: SpotBootstrapSpec) -> str:
    """Clone the workload repo, idempotently.

    The URL and branch are baked in as literals rather than interpolated from
    the spot's environment — see :class:`SpotBootstrapSpec`.
    """
    checkout = _quote(spec.checkout)
    return f"""
if [ ! -d {checkout}/.git ]; then
  rm -rf {checkout}
  git clone --depth 1 --branch {_quote(spec.branch)} {_quote(spec.repo_url)} {checkout}
fi
""".strip()


def _config_block(spec: SpotBootstrapSpec) -> str:
    if not spec.config_copies:
        return ""
    lines: list[str] = []
    for copy in spec.config_copies:
        lines.append(f"mkdir -p {_quote(copy.parent())}")
        lines.append(
            f'aws s3 cp "${{S3_STAGING}}/{copy.source_name}" {_quote(copy.dest)} '
            f"--region {_quote(spec.region)} --quiet"
        )
        if copy.chown:
            lines.append(f"chown -R ec2-user:ec2-user {_quote(copy.chown)}")
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

    blocks = [
        preamble,
        _watchdog_block(),
        _interpreter_block(),
        _clone_block(spec),
    ]
    config = _config_block(spec)
    if config:
        blocks.append(config)
    return "\n\n".join(blocks) + "\n"


def render_install_deps(spec: SpotBootstrapSpec) -> str:
    """``pip install -r requirements.txt`` in the checkout.

    Separate from the bootstrap because it has its own SSM budget — deps take
    minutes where the bootstrap takes ~50s, and one timeout covering both
    tells you nothing about which was slow.

    No system-python fallback: :func:`render_bootstrap` guarantees the
    interpreter, so falling back here would silently resolve wheels against a
    different version than ``requirements.txt`` was compiled for.
    """
    return f"""set -eo pipefail
export HOME=/home/ec2-user XDG_CACHE_HOME=/tmp AWS_REGION={spec.region} AWS_DEFAULT_REGION={spec.region}
cd {_quote(spec.checkout)}
{PYTHON} -m pip install --quiet --no-warn-script-location -r requirements.txt 2>&1 | tail -1
"""


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
        p.add_argument("--repo-url", required=True)
        p.add_argument("--checkout", required=True, help="Absolute clone path on the spot.")
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

    args = parser.parse_args(argv)
    spec = _spec_from_args(args)
    script = render_bootstrap(spec) if args.cmd == "render" else render_install_deps(spec)
    if args.json:
        print(json.dumps({"script": script}))
    else:
        sys.stdout.write(script)
    return 0


if __name__ == "__main__":
    sys.exit(main())
