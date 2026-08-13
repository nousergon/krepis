#!/usr/bin/env python3
"""Merge-time patch-version bump.

Adopted for krepis 2026-08-13 (alpha-engine-config-I7197), mirroring
nousergon-lib where it has run since config-I2716 (2026-07-16).

PRs no longer touch the version line — version-bump-check.yml forbids it
(except under the release:manual label). This script is the single writer:
auto-version-bump.yml runs it on push to main and pushes the bump commit
directly via the AUTOBUMP_DEPLOY_KEY ruleset bypass.

Reads pyproject.toml [project].version and src/krepis/__init__.py
__version__, asserts they are in lockstep (same invariant as
tests/test_version_pin.py), bumps the patch component in BOTH files, and
prints the new version to stdout. Fails loud on any mismatch or parse
failure — a wrong version string published to PyPI is unrecoverable
(PyPI forbids re-upload of a yanked version number).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PYPROJECT = Path("pyproject.toml")
INIT = Path("src/krepis/__init__.py")

_PYPROJECT_RE = re.compile(r'^version = "(\d+)\.(\d+)\.(\d+)"$', re.MULTILINE)
_INIT_RE = re.compile(r'^__version__ = "(\d+)\.(\d+)\.(\d+)"$', re.MULTILINE)


def read_version(text: str, pattern: re.Pattern[str], path: Path) -> tuple[int, int, int]:
    matches = pattern.findall(text)
    if len(matches) != 1:
        raise SystemExit(
            f"autobump: expected exactly one version line in {path}, found {len(matches)}"
        )
    return tuple(int(p) for p in matches[0])  # type: ignore[return-value]


def bump(pyproject_path: Path = PYPROJECT, init_path: Path = INIT) -> str:
    py_text = pyproject_path.read_text()
    init_text = init_path.read_text()
    py_ver = read_version(py_text, _PYPROJECT_RE, pyproject_path)
    init_ver = read_version(init_text, _INIT_RE, init_path)
    if py_ver != init_ver:
        raise SystemExit(
            f"autobump: version lockstep violated — {pyproject_path} has "
            f"{'.'.join(map(str, py_ver))}, {init_path} has {'.'.join(map(str, init_ver))}"
        )
    major, minor, patch = py_ver
    new = f"{major}.{minor}.{patch + 1}"
    pyproject_path.write_text(_PYPROJECT_RE.sub(f'version = "{new}"', py_text))
    init_path.write_text(_INIT_RE.sub(f'__version__ = "{new}"', init_text))
    return new


if __name__ == "__main__":
    sys.stdout.write(bump() + "\n")
