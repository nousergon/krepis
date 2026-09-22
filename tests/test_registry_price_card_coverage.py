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
from krepis.cost import (
    PriceCardLookupError,
    live_group_primaries,
    load_default_pricing,
    unpriced_live_primaries,
)


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


def _live_primaries(registry: mr.Registry):
    """Thin alias onto ``krepis.cost.live_group_primaries`` — the SINGLE
    declared source of the primary-selection + chaos-probe carve-out
    (alpha-engine-config-I11112 "Non-inferable" §1: the merge-time and
    run-time enforcers must read the same declared source, not a second copy
    of the rule). This file used to carry its own copy; the copy in
    ``alpha-engine-config``'s own CI test would then have been a THIRD.
    """
    return live_group_primaries(registry)


def test_every_live_groups_primary_has_an_active_price_card():
    """The class fix: a model promoted to a group's primary with no
    matching :class:`~krepis.cost.PriceCard` must fail THIS test, not be
    discovered live in production when its first cost record is dropped
    (glm-5.3 / alpha-engine-config-I11100 — promoted 2026-09-12, undetected
    until the 2026-09-19 weekly run).
    """
    registry = _load_live_registry()
    missing = unpriced_live_primaries(registry, load_default_pricing())

    assert not missing, (
        "live registry group primary(s) with NO active PriceCard in "
        f"krepis/model_pricing.yaml: {[str(m) for m in missing]}. Every "
        "group's primary must have a matching card — add one (see "
        "model_pricing.yaml's header for schema + sourcing convention) "
        "before promoting."
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


# ── Hermetic tests of the RULE itself (alpha-engine-config-I11112) ────────────
#
# The two tests above need the private registry and therefore SKIP on krepis
# CI (I11109). A rule enforced only where it happens to be runnable is the
# exact defect I11112 was filed over — ten preflight assertions rendering as
# a green `OK` because their environment could not reach them. These run
# everywhere: they pin `unpriced_live_primaries`' behaviour against a fake
# registry, so a regression in the shared rule reddens krepis CI even though
# the live-registry assertion above skipped.


class _FakeRegistry:
    """Structural stand-in for ``krepis.model_registry.Registry`` — only the
    three attributes ``live_group_primaries`` actually reads."""

    def __init__(self, groups: "dict[str, list[str]]", models: dict):
        self.groups = groups
        self.models = models

    def live_group_ids(self, group: str) -> "list[str]":
        return list(self.groups.get(group, []))


def test_unpriced_live_primary_is_reported():
    """NEGATIVE CONTROL for the whole mechanism: a group whose primary has no
    price card must come back in the list. This is I11100's glm-5.3 shape
    reproduced without the private registry.
    """
    registry = _FakeRegistry(
        groups={"ultra": ["deliberately-unpriced-model-xyz"]},
        models={"deliberately-unpriced-model-xyz": {"model": "deliberately-unpriced-model-xyz"}},
    )
    missing = unpriced_live_primaries(registry, load_default_pricing())
    assert [m.model_id for m in missing] == ["deliberately-unpriced-model-xyz"]


def test_priced_live_primary_is_not_reported():
    """The positive half — otherwise the test above passes for a function
    that returns every primary unconditionally."""
    pricing = load_default_pricing()
    priced = pricing.cards[0].model_name
    registry = _FakeRegistry(
        groups={"ultra": ["some-id"]},
        models={"some-id": {"model": priced}},
    )
    assert unpriced_live_primaries(registry, pricing) == []


def test_chaos_probe_primary_is_excluded():
    """A deliberately-unservable fault-injection entry never bills, so it can
    never generate a record to price (alpha-engine-config-I10126)."""
    registry = _FakeRegistry(
        groups={"chaos": ["broken-on-purpose"]},
        models={"broken-on-purpose": {"model": "broken-on-purpose", "chaos_probe": True}},
    )
    assert live_group_primaries(registry) == []
    assert unpriced_live_primaries(registry, load_default_pricing()) == []


def test_empty_group_contributes_no_primary():
    """A group with no live member has no primary to price — and must not
    raise an IndexError reaching for one."""
    registry = _FakeRegistry(groups={"retired": []}, models={})
    assert live_group_primaries(registry) == []


def test_price_card_lookup_error_is_the_only_swallowed_exception():
    """The rule converts a MISSING CARD into a finding; any other failure of
    the price table is a defect and must propagate. A table that raises
    something else would otherwise render as full coverage."""
    class _ExplodingTable:
        def get(self, name, when):
            raise RuntimeError("price table is corrupt")

    registry = _FakeRegistry(
        groups={"ultra": ["some-id"]}, models={"some-id": {"model": "anything"}},
    )
    with pytest.raises(RuntimeError, match="price table is corrupt"):
        unpriced_live_primaries(registry, _ExplodingTable())

    assert PriceCardLookupError is not None  # the caught type, named explicitly
