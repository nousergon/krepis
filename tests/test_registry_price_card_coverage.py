"""Class fix for alpha-engine-config-I11100: a served model with no price
card silently drops its whole cost record (``cost.py``'s ``record_llm_call``
now degrades that to ``cost_source: "unpriced"`` instead — see
``TestRecordLlmCall.test_unknown_model_no_price_card_degrades_to_unpriced``
in ``test_cost_llm.py`` for that half). This file is the OTHER half: nothing
previously caught the promotion itself, i.e. a model becoming a live
registry group's primary with no matching :class:`~krepis.cost.PriceCard`.
Fan-in coverage (keyed on the cost object existing) is now satisfied even
when unpriced, so it will never again flag this — a model-promotion PR must
fail HERE instead.

``LLM_MODEL_REGISTRY.yaml`` is the private, hand-maintained source of truth
and lives in the sibling ``alpha-engine-config`` checkout, read via
``krepis.model_registry.find_registry`` (the SAME discovery order every
production consumer uses — ``$LLM_MODEL_REGISTRY_PATH``, then a bounded walk
up from cwd probing ``alpha-engine-config/private-docs/...``). Read-only:
this test never writes to that file.

CI gap (tracked, not silently swallowed): krepis' ``test.yml`` sparse-checks
out two PUBLIC sibling repos for the existing spot_bootstrap parity guard,
but ``alpha-engine-config`` is PRIVATE — sparse-checking it into CI needs a
scoped credential decision this session did not have authority to make.
Until that lands, this test SKIPS on CI with a loud, named reason rather
than hard-failing every krepis build; it still enforces the invariant on
every dev laptop with the fleet checked out, which is where every
model-promotion PR in this session's history has actually been authored and
reviewed. Follow-up: alpha-engine-config-I11109.
"""

from __future__ import annotations

from datetime import timezone
from datetime import datetime as _datetime

import pytest

from krepis import model_registry as mr
from krepis.cost import PriceCardLookupError, load_default_pricing


def _load_live_registry() -> mr.Registry:
    try:
        path = mr.find_registry()
    except mr.RegistryNotFoundError as exc:
        message = (
            f"LLM_MODEL_REGISTRY.yaml not found ({exc}). This test enforces "
            f"registry-primary price-card coverage (alpha-engine-config-"
            f"I11100) and needs the private alpha-engine-config checkout as "
            f"a sibling of krepis, or $LLM_MODEL_REGISTRY_PATH set. "
            f"alpha-engine-config is PRIVATE, so CI cannot sparse-checkout "
            f"it without a scoped credential — that decision is tracked as "
            f"alpha-engine-config-I11109 and unresolved as of this write. "
            f"Skipping rather than failing every krepis build in the "
            f"meantime; run this locally with the fleet checked out to "
            f"actually enforce the invariant."
        )
        pytest.skip(message)
    return mr.load_registry(path)


def _live_primaries(registry: mr.Registry) -> list[tuple[str, str, str]]:
    """``(group, model_id, api_model_name)`` for every group with a live primary.

    Excludes any group whose primary declares ``chaos_probe: true`` — per the
    registry's own schema (``LLM_MODEL_REGISTRY.yaml`` header, "Marks an
    entry as PART OF A DELIBERATELY, PERMANENTLY BROKEN group") a chaos-probe
    entry is routed at a permanently-unservable, non-billing model on purpose
    (fault injection, alpha-engine-config-I10126) and never reaches a real
    upstream, so it will never generate a real cost record to price. The
    registry's own validator (invariant 21) already forbids mixing a
    chaos_probe member into an ordinary group, so this is never a loophole
    for a real primary to duck the check.
    """
    out = []
    for group in sorted(registry.groups):
        live = registry.live_group_ids(group)
        if not live:
            continue
        primary_id = live[0]
        entry = registry.models[primary_id]
        if entry.get("chaos_probe") is True:
            continue
        api_model_name = entry.get("model") or primary_id
        out.append((group, primary_id, api_model_name))
    return out


def test_every_live_groups_primary_has_an_active_price_card():
    """The class fix: a model promoted to a group's primary with no
    matching :class:`~krepis.cost.PriceCard` must fail THIS test, not be
    discovered live in production when its first cost record is dropped
    (glm-5.3 / alpha-engine-config-I11100 — promoted 2026-09-12, undetected
    until the 2026-09-19 weekly run).
    """
    registry = _load_live_registry()
    pricing = load_default_pricing()
    today = _datetime.now(timezone.utc)

    missing: list[str] = []
    for group, model_id, api_model_name in _live_primaries(registry):
        try:
            pricing.get(api_model_name, today)
        except PriceCardLookupError:
            missing.append(
                f"group={group!r} primary={model_id!r} "
                f"(api model_name={api_model_name!r})"
            )

    assert not missing, (
        "live registry group primary(s) with NO active PriceCard in "
        f"krepis/model_pricing.yaml: {missing}. Every group's primary must "
        "have a matching card — add one (see model_pricing.yaml's header "
        "for schema + sourcing convention) before promoting."
    )


def test_a_primary_with_no_card_is_caught_by_this_test():
    """Negative control: prove the assertion above is load-bearing, not a
    vacuous pass on an empty ``missing`` list because ``_live_primaries``
    silently returned nothing. Deliberately unpriced synthetic model.
    """
    registry = _load_live_registry()
    live = _live_primaries(registry)
    assert live, "no live registry group has ANY primary — is the fixture wrong?"

    pricing = load_default_pricing()
    today = _datetime.now(timezone.utc)
    with pytest.raises(PriceCardLookupError):
        pricing.get("deliberately-unpriced-model-xyz", today)
