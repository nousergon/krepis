"""Tests for the multi-provider cost surface added in v0.9.0.

Covers ``metadata_from_openai_completion``, ``record_llm_call`` cost-source
selection, the OpenRouter ``:variant`` price-card suffix strip, and
provider-scoped tool-fee naming.
"""

from datetime import date
from types import SimpleNamespace

import pytest

from krepis.cost import (
    PriceCardLookupError,
    load_default_pricing,
    load_default_tool_fees,
    metadata_from_openai_completion,
    record_llm_call,
    recompute_cost,
)
from krepis.llm import LLMResult, LLMUsage
from krepis.model_metadata import ModelMetadata

AT = date(2026, 7, 3)


def _openai_completion(model="moonshotai/kimi-k2.6", cost=None, searches=0,
                       cached=0):
    usage = SimpleNamespace(
        prompt_tokens=1000,
        completion_tokens=500,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
    )
    if cost is not None:
        usage.cost = cost
    if searches:
        usage.web_search_requests = searches
    return SimpleNamespace(
        model=model,
        choices=[SimpleNamespace(message=SimpleNamespace(content="x"))],
        usage=usage,
    )


def _anthropic_message(model="claude-haiku-4-5"):
    return SimpleNamespace(
        model=model,
        content=[SimpleNamespace(type="text", text="x")],
        usage=SimpleNamespace(
            input_tokens=1000, output_tokens=500,
            cache_read_input_tokens=0, cache_creation_input_tokens=0,
            cache_creation=None, server_tool_use=None,
        ),
    )


class TestMetadataFromOpenAI:
    def test_field_mapping(self):
        md = metadata_from_openai_completion(
            _openai_completion(cost=0.0042, searches=2, cached=100),
            provider="openrouter",
        )
        assert md.model_name == "moonshotai/kimi-k2.6"
        assert md.provider == "openrouter"
        assert md.input_tokens == 1000
        assert md.output_tokens == 500
        assert md.cache_read_tokens == 100
        assert md.web_search_requests == 2
        assert md.provider_reported_cost_usd == pytest.approx(0.0042)

    def test_no_openrouter_extensions(self):
        md = metadata_from_openai_completion(_openai_completion())
        assert md.provider == "openai"
        assert md.provider_reported_cost_usd is None
        assert md.web_search_requests == 0


class TestVariantSuffixStrip:
    def test_floor_variant_resolves_to_bare_slug_card(self):
        table = load_default_pricing()
        card = table.get("deepseek/deepseek-v4-flash:floor", AT)
        assert card.model_name == "deepseek/deepseek-v4-flash"
        assert card.input_per_1m == pytest.approx(0.09)

    def test_unknown_model_still_raises(self):
        table = load_default_pricing()
        with pytest.raises(PriceCardLookupError):
            table.get("nobody/nothing:floor", AT)


class TestRecordLlmCall:
    def test_llm_result_with_provider_cost_is_canonical(self):
        result = LLMResult(
            text="x", model="moonshotai/kimi-k2.6", provider="openrouter",
            usage=LLMUsage(
                input_tokens=1000, output_tokens=500,
                provider_cost_usd=0.0042,
            ),
            raw_request={},
        )
        record = record_llm_call(result, at=AT)
        assert record["cost_source"] == "provider_reported"
        assert record["cost_usd"] == pytest.approx(0.0042)
        assert record["provider"] == "openrouter"
        assert record["model"] == "moonshotai/kimi-k2.6"

    def test_llm_result_without_provider_cost_uses_card(self):
        result = LLMResult(
            text="x", model="moonshotai/kimi-k2.6", provider="openrouter",
            usage=LLMUsage(input_tokens=1_000_000, output_tokens=0),
            raw_request={},
        )
        record = record_llm_call(result, at=AT)
        assert record["cost_source"] == "price_card"
        assert record["cost_usd"] == pytest.approx(0.66)

    def test_anthropic_message_matches_record_anthropic_call_shape(self):
        record = record_llm_call(_anthropic_message(), at=AT)
        assert record["provider"] == "anthropic"
        assert record["cost_source"] == "price_card"
        # 1000 in @ $1/M + 500 out @ $5/M
        assert record["cost_usd"] == pytest.approx(0.001 + 0.0025)

    def test_openai_completion_with_explicit_provider(self):
        record = record_llm_call(
            _openai_completion(model="deepseek/deepseek-v4-flash:floor"),
            provider="openrouter",
            at=AT,
        )
        assert record["provider"] == "openrouter"
        assert record["cost_source"] == "price_card"
        # 1000 in @ $0.09/M + 500 out @ $0.18/M (card via variant strip)
        assert record["cost_usd"] == pytest.approx(0.00009 + 0.00009)

    def test_unknown_model_no_provider_cost_raises(self):
        result = LLMResult(
            text="x", model="unknown/model", provider="openrouter",
            usage=LLMUsage(input_tokens=10, output_tokens=10),
            raw_request={},
        )
        with pytest.raises(PriceCardLookupError):
            record_llm_call(result, at=AT)

    def test_extra_fields_merge(self):
        record = record_llm_call(
            _anthropic_message(), at=AT, extra_fields={"edition": "am"}
        )
        assert record["edition"] == "am"


class TestProviderScopedToolFees:
    def test_openrouter_web_search_priced_at_openrouter_rate(self):
        md = ModelMetadata(
            model_name="moonshotai/kimi-k2.6", provider="openrouter",
            web_search_requests=10,
        )
        cost = recompute_cost(
            md, load_default_pricing(),
            tool_fee_table=load_default_tool_fees(), at=AT,
        )
        # 10 requests @ $5/1k = $0.05 (no tokens)
        assert cost == pytest.approx(0.05)

    def test_anthropic_web_search_keeps_legacy_rate(self):
        md = ModelMetadata(
            model_name="claude-haiku-4-5", web_search_requests=10,
        )
        cost = recompute_cost(
            md, load_default_pricing(),
            tool_fee_table=load_default_tool_fees(), at=AT,
        )
        # 10 requests @ $10/1k = $0.10
        assert cost == pytest.approx(0.10)
