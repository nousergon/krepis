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
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "BOOTSTRAP_SIGNATURES",
    "MIN_CATEGORIES",
    "Clone",
    "ConfigCopy",
    "InlineBootstrap",
    "SpotBootstrapSpec",
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


def _hard_timeout_block(seconds: int) -> str:
    """Power the box off after ``seconds``, whatever the workload is doing.

    A transient timer rather than a unit file: it needs no ``[Install]``
    section, no ``daemon-reload`` and no idempotence guard, and it dies with
    the instance. ``systemd-run`` failing is fatal — an uncapped spot whose
    workload hangs runs until somebody notices the bill, and "the cap could
    not be armed" is exactly the condition under which the run must not
    start.
    """
    return f"""
systemd-run --on-active={int(seconds)} --unit=ec2-spot-hard-timeout \\
    --description='spot hard runtime cap ({int(seconds)}s)' /sbin/shutdown -h now || {{
  echo "ERROR: could not arm the {int(seconds)}s hard-timeout timer — refusing to start an uncapped spot workload" >&2
  exit 1
}}
""".strip()


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

    blocks = [preamble, _watchdog_block()]
    if spec.max_runtime_seconds is not None:
        blocks.append(_hard_timeout_block(spec.max_runtime_seconds))
    blocks += [_interpreter_block(), _clone_block(spec)]
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

    args = parser.parse_args(argv)

    if args.cmd == "scan":
        findings = scan_for_inline_bootstraps(args.root)
        for finding in findings:
            print(finding)
        # Exit code IS the verdict: this is meant to run in CI and in the
        # fleet sweep, and a scanner whose only output is prose gets read by
        # nobody twice.
        return 1 if findings else 0

    spec = _spec_from_args(args)
    script = render_bootstrap(spec) if args.cmd == "render" else render_install_deps(spec)
    if args.json:
        print(json.dumps({"script": script}))
    else:
        sys.stdout.write(script)
    return 0


if __name__ == "__main__":
    sys.exit(main())
