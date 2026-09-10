"""The coverage gate's *scope* is asserted here, not only its number.

repository-baseline-policy.md §4.2 C5: the way a coverage gate stops being
honest is by narrowing what it measures rather than by lowering the number —
which reads as an improvement in every report. Measured on symposion, removing
one flag moved the reported figure from 34.76% to 92.36% with no new test
code.

krepis carries no `[tool.coverage]` section in `pyproject.toml` at all — the
whole gate is the single `--cov=krepis --cov-fail-under=90` invocation in
`.github/workflows/test.yml`, run once per matrix leg (Python 3.9–3.13). So
these tests read that workflow file (plain text/regex — `tomllib` is 3.11+
and this repo's matrix runs 3.9, so a TOML parse would take the 3.9/3.10 legs
down at collection) and assert what a passing suite cannot otherwise notice:

* the measured source is the WHOLE ``krepis`` package (C1), never a path or
  submodule narrower than that;
* the floor is enforced by a non-zero exit (C2), appears exactly once (not
  once per matrix leg with different values), and is a ratchet that may be
  raised and never lowered (C3);
* no coverage config anywhere in the repo sets an ``omit`` — krepis has none
  today, and this test forces a future one into review rather than letting it
  shrink the denominator silently;
* every source module under ``src/krepis`` is inside the measured package —
  nothing is invisible to the gate by living outside ``src/krepis``.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "test.yml"
PACKAGE_ROOT = REPO_ROOT / "src" / "krepis"

#: The floor may be RAISED here as coverage improves. Lowering it is a policy
#: amendment (repository-baseline-policy.md §4.2 C3), not a code change.
MINIMUM_FLOOR = 90

#: krepis pins no justified omissions today. Widening this set is a scope
#: decision, not a drive-by coverage bump — a real entry here should also be
#: reflected in a pyproject.toml [tool.coverage.run] comment.
EXPECTED_OMIT: set[str] = set()

#: Files that are not importable Python source (data/schema assets living
#: inside the package directory) but sit under src/krepis — --cov=krepis
#: only ever measures .py files, so these are inert with respect to scope,
#: not a narrowing.
_NON_SOURCE_SUFFIXES = {".yaml", ".json", ".typed"}


def test_coverage_source_is_the_whole_package() -> None:
    """C1 — ``--cov`` names the installed package, so unimported modules still count."""
    text = TEST_WORKFLOW.read_text(encoding="utf-8")
    sources = re.findall(r"--cov=(\S+?)(?:\s|$)", text)
    assert sources == ["krepis"], (
        f"coverage source must be exactly the krepis package, got {sources!r}. "
        "Narrowing it to a submodule or a path measures the tested subset and "
        "reports it as the repository."
    )


def test_coverage_floor_is_enforced_once_and_never_lowered() -> None:
    """C2 + C3 — the gate exits non-zero below a floor that only ratchets up."""
    text = TEST_WORKFLOW.read_text(encoding="utf-8")
    found = re.findall(r"--cov-fail-under=(\d+)", text)
    assert len(found) == 1, (
        f"expected exactly one --cov-fail-under in {TEST_WORKFLOW.name} (one "
        f"invocation shared by every matrix leg), got {found!r} — a second "
        "occurrence could diverge per leg and mask a regression on some "
        "interpreters."
    )
    assert int(found[0]) >= MINIMUM_FLOOR, (
        f"coverage floor {found[0]} is below the ratchet {MINIMUM_FLOOR}. "
        "A floor is raised as coverage improves and never lowered to make a "
        "change pass (repository-baseline-policy.md §4.2 C3)."
    )


def test_no_coverage_config_omits_source_files() -> None:
    """A shrunk denominator is a narrowing that neither flag would reveal."""
    for name in (".coveragerc", "setup.cfg", "tox.ini", "pyproject.toml"):
        path = REPO_ROOT / name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"^\s*omit\s*=", text, re.MULTILINE), (
            f"{name} sets a coverage `omit`. krepis's pinned justified set is "
            f"{sorted(EXPECTED_OMIT)!r} today (empty) — update EXPECTED_OMIT "
            "here alongside the justification if a real omission is added; "
            "an unreviewed omit removes files from the denominator, raising "
            "the reported figure without adding a test."
        )


def test_every_source_module_is_inside_the_measured_package() -> None:
    """No .py file claiming to be krepis source lives outside src/krepis."""
    src = REPO_ROOT / "src"
    stray = sorted(
        p.relative_to(REPO_ROOT).as_posix()
        for p in src.rglob("*.py")
        if PACKAGE_ROOT not in p.parents and p != PACKAGE_ROOT
    )
    assert not stray, (
        f"source modules outside the measured package are invisible to the "
        f"coverage gate: {stray}"
    )


def test_measured_package_is_a_real_importable_package() -> None:
    """--cov=krepis only measures something if src/krepis is a package."""
    assert (PACKAGE_ROOT / "__init__.py").is_file(), (
        "src/krepis/__init__.py is missing — --cov=krepis would resolve to "
        "nothing, and the gate would pass vacuously with zero files measured."
    )
    modules = [
        p for p in PACKAGE_ROOT.rglob("*")
        if p.is_file() and p.suffix not in _NON_SOURCE_SUFFIXES and p.suffix != ""
    ]
    assert modules, "no files found under src/krepis — scope check would pass vacuously"
