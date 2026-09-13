#!/usr/bin/env python3
"""Render ``src/krepis/resolve_schema.json`` from :mod:`krepis.resolve_contract`.

The committed schema is a BUILD ARTIFACT, not a hand-authored file. Run this
after any change to the contract model and commit the result; the byte-identity
test in ``tests/test_resolve_contract_schema.py`` fails the pull request that
forgets to.

Usage::

    python3 scripts/gen_resolve_schema.py          # write the file
    python3 scripts/gen_resolve_schema.py --check   # exit 1 if it is stale
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from krepis.resolve_contract import render_resolve_schema  # noqa: E402

SCHEMA_PATH = _SRC / "krepis" / "resolve_schema.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit 1 if the committed schema is stale",
    )
    args = parser.parse_args()

    rendered = render_resolve_schema()
    current = SCHEMA_PATH.read_text() if SCHEMA_PATH.exists() else None

    if args.check:
        if current == rendered:
            print(f"{SCHEMA_PATH} is up to date")
            return 0
        print(
            f"{SCHEMA_PATH} is STALE — regenerate with "
            "`python3 scripts/gen_resolve_schema.py` and commit the result",
            file=sys.stderr,
        )
        return 1

    if current == rendered:
        print(f"{SCHEMA_PATH} unchanged")
        return 0

    SCHEMA_PATH.write_text(rendered)
    print(f"wrote {SCHEMA_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
