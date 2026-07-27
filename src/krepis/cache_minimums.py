"""Minimum cacheable prompt-prefix length per model.

Below a model's minimum, an explicit ``cache_control`` marker **silently
caches nothing** — no error, just ``cache_creation_input_tokens: 0`` forever
— and it consumes one of only four breakpoint slots per request. So the
value has to be right, and it has to come from one place.

**This is a provider fact, not a fleet decision**, which is why it ships as
package data in this public, pip-installable library rather than in a private
config repo. Public consumers cannot read a private registry at runtime;
that constraint is precisely what drove an earlier consumer to hardcode its
own copy, which then drifted (two of four entries wrong, both too high — see
``cache_minimums.yaml`` for the incident note).

**The values are not monotonic across generations.** Newer is not lower and
tier does not predict: Fable 5 and Opus 5 are 512, Opus 4.7 is 2048, and
Haiku 4.5 is 4096 — the highest of the set, on the tier where mechanical
work routes. Never infer a minimum from tier, recency, or a sibling model.

Usage::

    from krepis.cache_minimums import clears_cache_minimum

    verdict = clears_cache_minimum("claude-haiku-4-5", prefix_tokens)
    if verdict is True:
        block["cache_control"] = {"type": "ephemeral"}
    elif verdict is False:
        ...  # below the minimum — a marker here would waste a slot
    else:
        ...  # unknown model: decide explicitly, do not guess

See ``nous-ergon-ops/policies/prompt-caching-policy.md`` §3.7.
"""
from __future__ import annotations

from functools import lru_cache
from importlib import resources

import yaml

# Resolved through importlib.resources, matching how ``krepis.cost`` loads
# ``model_pricing.yaml``. A ``Path(__file__).parent`` join would break in a
# zipped or frozen install, and diverging from the sibling loader for no
# reason is its own defect class.
_MINIMUMS_RESOURCE = "cache_minimums.yaml"


class CacheMinimumLookupError(LookupError):
    """Raised by :func:`require_cache_minimum` for an unknown model."""


@lru_cache(maxsize=1)
def _load() -> dict[str, int]:
    with resources.files("krepis").joinpath(_MINIMUMS_RESOURCE).open(
        "r", encoding="utf-8"
    ) as fh:
        raw = yaml.safe_load(fh)
    minimums = (raw or {}).get("minimums") or {}
    if not isinstance(minimums, dict) or not minimums:
        raise ValueError(
            f"krepis/{_MINIMUMS_RESOURCE} carries no 'minimums' mapping — refusing to "
            "return an empty table, which would silently read as 'no model "
            "has a minimum' and disable every below-threshold guard"
        )
    return {str(k): int(v) for k, v in minimums.items()}


def known_models() -> tuple[str, ...]:
    """Every model ID with a published minimum, longest first.

    Longest-first because :func:`cache_minimum` resolves dated snapshots by
    longest-prefix match, and the order is what makes that unambiguous.
    """
    return tuple(sorted(_load(), key=len, reverse=True))


def cache_minimum(model: str | None) -> int | None:
    """Minimum cacheable prefix for *model*, or ``None`` if not published.

    Resolution is exact-match first, then longest-prefix match so dated
    snapshots (``claude-sonnet-4-6-20250514``) resolve to their family
    (``claude-sonnet-4-6``). Longest-prefix rather than shortest matters:
    ``claude-opus-4-5`` must not swallow ``claude-opus-4-5x`` style IDs, and
    a shorter family key must never win over a more specific one.

    ``None`` means **not published** — for automatic-prefix providers there
    is no client-side minimum to respect, and for an unrecognised model we
    genuinely do not know. It never means zero. Callers that must have a
    value should use :func:`require_cache_minimum` and handle the raise.
    """
    if not model:
        return None
    table = _load()
    exact = table.get(model)
    if exact is not None:
        return exact
    for known in known_models():
        if model.startswith(known):
            return table[known]
    return None


def require_cache_minimum(model: str | None) -> int:
    """Like :func:`cache_minimum` but raises instead of returning ``None``.

    For call sites where proceeding without the real value would place
    markers blind — failing loud beats a plausible default, because a
    below-minimum marker produces no error to notice later.
    """
    value = cache_minimum(model)
    if value is None:
        raise CacheMinimumLookupError(
            f"no published cache minimum for model {model!r}. If this is an "
            "automatic-prefix provider it needs no client marker at all; if "
            "it is a new explicit-breakpoint model, add it to "
            "krepis/cache_minimums.yaml with a doc link."
        )
    return value


def clears_cache_minimum(model: str | None, prefix_tokens: int) -> bool | None:
    """Whether a *prefix_tokens*-long prefix is worth a breakpoint on *model*.

    Returns ``None`` when the model has no published minimum — deliberately
    tri-state rather than defaulting to ``True``. A ``True`` default would
    place markers on unknown models that may silently never cache; a
    ``False`` default would decline caching that would have worked. Neither
    is safe to pick on the caller's behalf, and both fail invisibly.
    """
    minimum = cache_minimum(model)
    if minimum is None:
        return None
    return prefix_tokens >= minimum
