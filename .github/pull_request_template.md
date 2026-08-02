## What & why

<!-- What does this change and why? Link any related issue. -->

## Checklist

- [ ] Tests added/updated for the behavior change
- [ ] `pytest` passes locally and coverage stays ≥ 90%
- [ ] `ruff check src/ tests/` is clean for files I touched
- [ ] Public API is additive-only (no renames or removals without migration period)
- [ ] Hermetic tests — no network, no AWS calls, no wall-clock dependence
- [ ] No secrets, credentials, or infrastructure detail committed
- [ ] Fail-loud preserved — no new silent `except: pass` swallows

## Test plan

<!-- How you verified this works. -->

---

**Prepared by:** <!-- model name from the session prompt, e.g. claude-sonnet-5, claude-opus-5, claude-haiku-4-5 -->
