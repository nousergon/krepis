# AGENTS.md — krepis

Operating instructions for AI agents and human contributors working in this repo.
Canonical, harness-neutral file; `CLAUDE.md` is a symlink to it because Claude Code reads
`CLAUDE.md`, not `AGENTS.md`.

## What this is

`krepis` (Greek κρηπίς — the foundation course a structure stands on) is a **library**, not a
service. It is `pip install`ed by other projects and has no deployment of its own: general-purpose
Python primitives for production data and LLM pipelines on AWS — structured logging, SSM secrets,
alert transports, bounded-backoff HTTP retry, S3 writer locks, LLM cost telemetry, the model-group
router, and trading-calendar/date helpers.

MIT licensed. Contributions under the DCO — see `CONTRIBUTING.md` and sign off with `git commit -s`.

## Working here

```bash
python -m pip install -e ".[dev]"
python -m pytest tests/ -q          # full suite
python -m ruff check src/ tests/    # lint
```

CI runs pytest across **Python 3.9–3.13**. The floor is real: this library is imported by consumers
on older runtimes, so `match`, `X | Y` annotations at runtime, and other 3.10+ syntax will fail the
matrix even though they pass locally.

## Conventions that are enforced, not suggested

- **Bump the version for any `src/` change.** `version-bump-check.yml` fails the PR otherwise, and
  it is not busywork: a push to `main` auto-tags and publishes to PyPI, so an unbumped change either
  fails to publish or silently ships under an existing version. Bump both `pyproject.toml` and
  `src/krepis/__version__` — they must agree.
- **Merging to `main` releases.** There is no separate release step to forget, and no way to walk a
  publish back. Treat a merge as a publish.
- **Fail loud.** This library is a dependency of pipelines that run unattended. A swallowed
  exception here becomes a silent wrong answer several repos away. Bare `except: pass`, silent
  `return None`, and graceful-degrade on a writer are all defects; raise instead. Where a swallow is
  genuinely correct, the inline comment must name what is being swallowed and where it is recorded.
- **Additive-only public API.** Consumers pin `krepis`. Renaming or removing an exported name breaks
  them at import time, in a way their tests will not catch until deploy. Emit both names for a
  release, migrate consumers, then remove — never a same-commit rename.
- **No edge-specific logic.** If a change only makes sense for one consumer, it belongs in that
  consumer. This is the shared foundation; a special case here is a dependency inversion.

## Testing

New behaviour needs a test that fails without the change. Prefer tests that pin the *contract*
rather than an incidental value — a test asserting a specific upstream price, for example, breaks on
a legitimate price change and teaches people to ignore it.

Tests must be hermetic: no network, no AWS calls, no wall-clock dependence. Anything reaching
outside the process gets a fixture or a fake.

## Cross-references

- `README.md` — module-by-module overview
- `CONTRIBUTING.md` — DCO and contribution flow
