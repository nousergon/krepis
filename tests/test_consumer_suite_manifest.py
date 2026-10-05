"""``.github/consumers.yaml`` is held to krepis's own source.

alpha-engine-config-I7487 / nous-ergon-ops-I710 / nous-ergon-ops-I738.

The consumer-suite workflow runs each declared consumer's suite against the
krepis candidate, with only the runtime prerequisites that consumer declares it
``provides``. That only protects anyone if the manifest is TRUE about krepis:

* every program krepis executes by name is declared — a new one is a change
  to krepis's runtime contract (krepis-PR95 added gitleaks as a test-job
  install and never declared it anywhere a consumer would see, I738);
* every consumer has answered for every declared prerequisite, so adding one
  fails here until somebody has checked each consumer's test job and image;
* the gitleaks pin is the one krepis's own CI installs, not a second copy.
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src" / "krepis"
MANIFEST = REPO / ".github" / "consumers.yaml"
TEST_WORKFLOW = REPO / ".github" / "workflows" / "test.yml"
SUITE_WORKFLOW = REPO / ".github" / "workflows" / "consumer-suite.yml"

_SUBPROCESS_CALLS = {"run", "call", "check_call", "check_output", "Popen"}


def _manifest() -> dict:
    return yaml.safe_load(MANIFEST.read_text())


def _programs_executed_by_name(tree: ast.AST) -> set:
    """Literal program names krepis runs: ``shutil.which("x")`` and a
    ``subprocess.<call>([...])`` whose argv starts with a string literal.
    An argv built at runtime (a caller-supplied command) is the caller's
    prerequisite, not krepis's, and is deliberately not counted."""
    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        owner = node.func.value
        owner_name = owner.id if isinstance(owner, ast.Name) else None
        if owner_name == "shutil" and node.func.attr == "which" and node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                found.add(arg.value)
        elif owner_name == "subprocess" and node.func.attr in _SUBPROCESS_CALLS and node.args:
            argv = node.args[0]
            if isinstance(argv, (ast.List, ast.Tuple)) and argv.elts:
                head = argv.elts[0]
                if isinstance(head, ast.Constant) and isinstance(head.value, str):
                    found.add(os.path.basename(head.value))
    return found


def _krepis_programs() -> dict:
    """``{program: sorted([module, ...])}`` over every module under src/krepis."""
    out = {}
    for path in sorted(SRC.rglob("*.py")):
        module = "krepis." + ".".join(path.relative_to(SRC).with_suffix("").parts)
        for program in _programs_executed_by_name(ast.parse(path.read_text(), filename=str(path))):
            out.setdefault(program, []).append(module)
    return out


def test_the_scanner_sees_both_forms():
    tree = ast.parse(
        "import shutil, subprocess\n"
        "shutil.which('alpha')\n"
        "subprocess.run(['/usr/bin/beta', '-x'])\n"
        "subprocess.Popen(cmd)\n"  # caller-supplied: not krepis's prerequisite
    )
    assert _programs_executed_by_name(tree) == {"alpha", "beta"}


def test_every_program_krepis_executes_is_a_declared_runtime_prerequisite():
    declared = _manifest()["runtime_prerequisites"]
    undeclared = {p: mods for p, mods in _krepis_programs().items() if p not in declared}
    assert not undeclared, (
        f"krepis executes {sorted(undeclared)} (from {undeclared}) but "
        f"{MANIFEST.relative_to(REPO)} does not declare them under "
        "runtime_prerequisites. A program krepis runs is part of its runtime "
        "contract with every consumer: declare it, then answer for it in each "
        "consumer's `provides` (nous-ergon-ops-I738)."
    )


def test_every_declared_prerequisite_is_still_executed_and_names_its_users():
    programs = _krepis_programs()
    for name, spec in _manifest()["runtime_prerequisites"].items():
        assert name in programs, f"runtime_prerequisites.{name} is declared but nothing in src/krepis runs it"
        assert sorted(spec["used_by"]) == sorted(programs[name]), (
            f"runtime_prerequisites.{name}.used_by={spec['used_by']} but it is run from {programs[name]}"
        )


def test_every_consumer_answers_for_every_prerequisite():
    doc = _manifest()
    required = set(doc["runtime_prerequisites"])
    assert doc["consumers"], "consumers.yaml declares no consumers — the consumer suite would run nothing"
    for consumer in doc["consumers"]:
        missing = required - set(consumer["provides"])
        assert not missing, (
            f"{consumer['repo']} does not declare it provides {sorted(missing)}. Check that "
            "consumer's test job and deploy image carry it, fix them if not, and only then "
            "add it to `provides` (nous-ergon-ops-I738)."
        )
        for key in ("repo", "python", "requirements", "test"):
            assert consumer.get(key), f"{consumer.get('repo')} is missing `{key}`"


def test_a_prerequisite_is_pinned_or_explicitly_preinstalled():
    for name, spec in _manifest()["runtime_prerequisites"].items():
        if spec.get("preinstalled"):
            continue
        assert re.fullmatch(r"\d+\.\d+\.\d+", str(spec.get("version"))), name
        assert re.fullmatch(r"[0-9a-f]{64}", str(spec.get("sha256"))), name


def test_gitleaks_pin_is_the_one_krepis_ci_installs():
    spec = _manifest()["runtime_prerequisites"]["gitleaks"]
    workflow = TEST_WORKFLOW.read_text()
    assert f'GITLEAKS_VERSION="{spec["version"]}"' in workflow
    assert f'GITLEAKS_SHA256="{spec["sha256"]}"' in workflow


def test_the_workflow_builds_its_matrix_from_the_manifest():
    workflow = SUITE_WORKFLOW.read_text()
    assert 'open(".github/consumers.yaml")' in workflow
    assert "fromJSON(needs.plan.outputs.consumers)" in workflow
    # The candidate must land AFTER the consumer's requirements, or the
    # consumer's own pin silently wins and the job tests nothing.
    assert workflow.index('pip install -r "consumer/${REQUIREMENTS}"') < workflow.index(
        'pip install "./krepis[${EXTRAS}]"'
    )


def test_declared_siblings_are_well_formed_and_cloned_by_the_workflow():
    """``siblings`` lets a consumer whose suite reads other public repositories
    (crucible-backtester: crucible-executor and nousergon-data) run here the
    way its own CI runs it, instead of being left uncovered (nous-ergon-ops-I710).
    """
    for consumer in _manifest()["consumers"]:
        for sib in consumer.get("siblings") or []:
            assert set(sib) == {"repo", "sparse", "env"}, sib
            assert re.fullmatch(r"nousergon/[A-Za-z0-9._-]+", sib["repo"]), sib
            assert re.fullmatch(r"[A-Za-z0-9._/-]+", sib["sparse"]), sib
            assert re.fullmatch(r"[A-Z][A-Z0-9_]*", sib["env"]), sib
    workflow = SUITE_WORKFLOW.read_text()
    assert '"siblings": json.dumps(c.get("siblings") or [])' in workflow
    assert 'echo "${var}=${dest}" >> "${GITHUB_ENV}"' in workflow
    # Siblings must be on disk before the suite runs.
    assert workflow.index("Check out declared sibling repositories") < workflow.index(
        "suite against the candidate"
    )
