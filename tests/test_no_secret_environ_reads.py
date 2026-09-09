"""Regression: no module in this repo reads a pinned secret via ``os.environ.get``.

alpha-engine-config-I10233: ``router.py``'s ``get_router()`` read
``OPENROUTER_API_KEY`` via a literal ``os.environ.get(...)`` — invisible to
every CONSUMER's own repo-tree secret scan once this module ships inside
site-packages, the identical defect class as
alpha-engine-config-I7924/I7925 (``nousergon_lib.preflight`` reading
``GITHUB_TOKEN`` the same way, which halted preopen trading). This test
re-greps krepis's own tree on every CI run so that class cannot regress
here silently.

krepis is the base of the ``get_secret`` dependency chain — ``nousergon_lib``
depends on ``krepis.secrets``, not the other way around — so this repo
cannot import ``nousergon_lib.testing.secret_scan`` without a circular
dependency. The scanner logic below is intentionally a small, self-contained
mirror of that module's ``scan_tree`` (crucible-predictor/nousergon-data's
``test_no_secret_environ_reads.py``), not a copy-paste of a bug: it is the
lowest layer in the stack, so it carries no ``scan_installed_packages`` half
— krepis has no first-party runtime dependency of its own whose
``site-packages`` copy could hide a read from this test.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT = _REPO_ROOT / "src"

# Mirrors the fleet-standard pinned-secret list carried by
# crucible-predictor/nousergon-data's test_no_secret_environ_reads.py, minus
# names those repos consume but krepis's own src/ has no call site for.
# krepis is the canonical home of `get_secret`, so any of these appearing
# here as a literal `os.environ.get(...)` read is the exact defect class
# this test exists to catch.
_PINNED_SECRETS = frozenset(
    [
        "ANTHROPIC_API_KEY",
        "OPENROUTER_API_KEY",
        "LANGCHAIN_API_KEY",
        "LANGSMITH_API_KEY",
        "VOYAGE_API_KEY",
        "POLYGON_API_KEY",
        "FMP_API_KEY",
        "FINNHUB_API_KEY",
        "FRED_API_KEY",
        "GMAIL_APP_PASSWORD",
        "GITHUB_TOKEN",
        "RAG_DATABASE_URL",
        "EDGAR_IDENTITY",
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_CHAT_ID",
        "LITELLM_MASTER_KEY",
    ]
)

# `secrets.py` itself is the one legitimate place a pinned-secret name is
# read via `os.environ.get` — it IS the SSM-first/env-fallback resolver
# every other module is required to route through. Excluding it here is not
# a scanner blind spot: `secrets.py`'s own reads are of the SOURCE-TOGGLE
# and per-secret NAME variables, never of a hardcoded pinned-secret literal
# (`get_secret` takes the name as an argument), so it never actually matches
# `_ENV_READ_RE` — this allowlist entry documents the intent rather than
# suppressing a real hit.
_ALLOWED_FILES: frozenset[str] = frozenset({"secrets.py"})

_ENV_READ_RE = re.compile(
    r'os\.(?:environ\.get|getenv)\(\s*["\']([A-Z_][A-Z0-9_]*)["\']'
)


def _iter_python_files():
    for path in _SRC_ROOT.rglob("*.py"):
        parts = set(path.parts)
        if parts & {".venv", "build", "tests", "node_modules", "package", "__pycache__"}:
            continue
        if path.name in _ALLOWED_FILES:
            continue
        yield path


def test_no_secret_environ_reads():
    violations: list[tuple[Path, int, str]] = []
    for path in _iter_python_files():
        text = path.read_text()
        for lineno, line in enumerate(text.splitlines(), start=1):
            for match in _ENV_READ_RE.finditer(line):
                name = match.group(1)
                if name in _PINNED_SECRETS:
                    violations.append((path.relative_to(_REPO_ROOT), lineno, name))
    assert not violations, (
        "Found os.environ.get reads of pinned secrets — use "
        "`from krepis.secrets import get_secret` instead:\n"
        + "\n".join(f"  {p}:{ln}  {name}" for p, ln, name in violations)
    )
