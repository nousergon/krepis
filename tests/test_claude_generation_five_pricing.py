from datetime import date

import pytest

from krepis.cost import compute_cost, load_default_pricing, PriceCardLookupError


@pytest.mark.parametrize("model,day,read", [
    ("claude-opus-5", "2026-07-24", .5),
    ("claude-opus-5-5", "2026-09-22", .2),
    ("claude-sonnet-5", "2026-06-30", .2),
    ("claude-sonnet-5-5", "2026-09-28", .2),
    ("claude-sonnet-5-5", "2026-10-07", .1),
    ("claude-fable-5", "2026-06-09", 1),
    ("claude-mythos-5", "2026-06-09", 1),
    ("claude-fable-5-1", "2026-09-01", .25),
    ("claude-mythos-5-1", "2026-09-01", .25),
    ("claude-fable-5-2", "2026-11-09", .2),
    ("claude-mythos-5-2", "2026-11-09", .2),
])
def test_generation_five_exact_model_and_effective_date(model, day, read):
    card = load_default_pricing().get(model, date.fromisoformat(day))
    assert card.cache_read_per_1m == read
    assert card.cache_create_1h_per_1m == card.input_per_1m * 2
    assert card.cache_create_per_1m == card.input_per_1m * 1.25


def test_model_cannot_be_priced_before_launch():
    with pytest.raises(PriceCardLookupError):
        load_default_pricing().get("claude-fable-5-2", date(2026, 11, 8))


@pytest.mark.parametrize("prompt,expected", [(100000, .001), (100001, .00500005)])
def test_haiku_prompt_tier_counts_cached_input(prompt, expected):
    card = load_default_pricing().get("claude-haiku-5-5", date(2026, 10, 7))
    cost = compute_cost(input_tokens=0, output_tokens=0, cache_read_tokens=prompt,
                        cache_create_tokens=0, card=card)
    assert cost == pytest.approx(expected)

@pytest.mark.parametrize("provider,attempts", [("anthropic", 2), ("openai", 1)])
def test_ambiguous_tier_boundaries_are_unpriced(provider, attempts):
    from krepis.cost import recompute_cost
    from krepis.model_metadata import ModelMetadata
    usage = ModelMetadata(
        model_name="claude-haiku-5-5", provider=provider, attempts=attempts,
        input_tokens=90000, cache_read_tokens=30000,
    )
    with pytest.raises(PriceCardLookupError, match="request boundaries"):
        recompute_cost(usage, load_default_pricing(), at=date(2026, 11, 19))
