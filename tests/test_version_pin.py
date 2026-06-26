"""Pin ``krepis.__version__`` to ``pyproject.toml::version`` (lockstep).

A pyproject-only bump (or an __init__-only bump) ships a wheel whose
package metadata and runtime ``__version__`` disagree — confusing for any
consumer that reads the runtime attribute. This test fails CI when the two
drift, so a bump that misses one side is caught before release.

Also guards the PyPI core-metadata 512-char cap on ``summary`` (sourced from
``pyproject.toml::project.description``): twine accepts a longer value
locally and at build time, but PyPI rejects the upload with HTTP 400 — and
by then ``auto-tag.yml`` has already cut the git tag, leaving PyPI out of
sync with the tag until a fresh patch release.
"""

from __future__ import annotations

import re
from pathlib import Path

import krepis

_PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _pyproject_field(field: str) -> str:
    """Read a top-level ``field = "..."`` from pyproject.toml's [project].

    Stdlib-free + Python-3.9-safe single regex (no tomllib import); the
    file ships exactly one top-level ``version``/``description`` line.
    """
    text = _PYPROJECT.read_text()
    match = re.search(rf'^{field}\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert match is not None, f"{field} not found in pyproject.toml"
    return match.group(1)


def test_init_version_matches_pyproject():
    """Bumping one of pyproject.toml / __init__.py without the other fails here."""
    assert krepis.__version__ == _pyproject_field("version"), (
        f"krepis.__version__={krepis.__version__!r} "
        f"!= pyproject.toml::version={_pyproject_field('version')!r} — bump "
        f"BOTH in lockstep (pyproject.toml + src/krepis/__init__.py)."
    )


def test_description_within_pypi_summary_cap():
    """PyPI rejects a >512-char project.description (the core-metadata
    ``summary`` field) at upload time, after auto-tag has already tagged."""
    description = _pyproject_field("description")
    assert len(description) <= 512, (
        f"pyproject.toml::description is {len(description)} chars; PyPI caps "
        f"the summary at 512 and rejects the upload with HTTP 400 after the "
        f"tag is already cut. Shorten it."
    )
