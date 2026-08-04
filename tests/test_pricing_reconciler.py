"""Tests for krepis.pricing._reconciler — upstream fetchers, normalisers, comparison."""

import json
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from krepis.pricing._reconciler import (
    per_1m,
    normalise_litellm,
    normalise_openrouter,
    values_agree,
    upstream_for,
    check_card_against_upstream,
    ReconcileError,
    UnknownOpenRouterModel,
    LITELLM_PREFIXES,
)


class TestPer1M:
    def test_converts_per_token_to_per_1m(self):
        assert per_1m(1e-6) == 1.0

    def test_small_price_converts_correctly(self):
        assert per_1m(2.5e-7) == 0.25

    def test_none_returns_none(self):
        assert per_1m(None) is None

    def test_empty_string_returns_none(self):
        assert per_1m("") is None

    def test_zero_returns_zero(self):
        assert per_1m(0) == 0.0


class TestNormaliseLitellm:
    def test_basic_conversion(self):
        card = {
            "input_cost_per_token": 1e-6,
            "output_cost_per_token": 5e-6,
            "cache_read_input_token_cost": 1e-7,
            "max_input_tokens": 128000,
            "max_output_tokens": 4096,
        }
        norm = normalise_litellm(card)
        assert norm["input_per_1m"] == 1.0
        assert norm["output_per_1m"] == 5.0
        assert norm["cache_read_per_1m"] == 0.1
        assert norm["max_context_tokens"] == 128000
        assert norm["max_output_tokens"] == 4096

    def test_falls_back_to_legacy_cache_spelling(self):
        """When cache_read_input_token_cost is None, uses input_cost_per_token_cache_hit."""
        card = {
            "input_cost_per_token": 1e-6,
            "output_cost_per_token": 5e-6,
            "input_cost_per_token_cache_hit": 2.5e-7,
            "max_input_tokens": 128000,
            "max_output_tokens": 4096,
        }
        norm = normalise_litellm(card)
        assert norm["cache_read_per_1m"] == 0.25

    def test_missing_fields_default_to_none(self):
        norm = normalise_litellm({})
        assert norm["input_per_1m"] is None
        assert norm["output_per_1m"] is None
        assert norm["cache_read_per_1m"] is None
        assert norm["max_context_tokens"] is None
        assert norm["max_output_tokens"] is None


class TestNormaliseOpenRouter:
    def test_basic_conversion(self):
        model = {
            "id": "deepseek/deepseek-v4-flash",
            "pricing": {"prompt": 1.5e-7, "completion": 6e-7, "input_cache_read": 2.8e-8},
            "top_provider": {"max_completion_tokens": 4096},
            "context_length": 128000,
        }
        norm = normalise_openrouter(model)
        assert norm["input_per_1m"] == 0.15
        assert norm["output_per_1m"] == 0.60
        assert norm["cache_read_per_1m"] == 0.028
        assert norm["max_context_tokens"] == 128000
        assert norm["max_output_tokens"] == 4096

    def test_null_pricing_is_tolerated(self):
        model = {"id": "test/model", "pricing": None, "top_provider": None, "context_length": None}
        norm = normalise_openrouter(model)
        assert norm["input_per_1m"] is None
        assert norm["max_context_tokens"] is None


class TestValuesAgree:
    def test_exact_match(self):
        assert values_agree(1.0, 1.0, price_field=False)

    def test_price_within_band(self):
        assert values_agree(1.0, 1.019, price_field=True)

    def test_price_exceeds_band(self):
        assert not values_agree(1.0, 1.03, price_field=True)

    def exact_limit_comparison(self):
        assert values_agree(128000, 128000, price_field=False)

    def test_limit_mismatch_detected(self):
        assert not values_agree(128000, 64000, price_field=False)

    def test_both_none_agree(self):
        assert values_agree(None, None, price_field=False)

    def test_one_none_disagrees(self):
        assert not values_agree(1.0, None, price_field=False)
        assert not values_agree(None, 1.0, price_field=False)


class TestUpstreamFor:
    def test_openrouter_route_resolved(self):
        or_db = {"deepseek/deepseek-v4-flash": {"id": "deepseek/deepseek-v4-flash", "pricing": {}}}
        source, key, fields = upstream_for("openrouter", "deepseek/deepseek-v4-flash", "openrouter", {}, or_db)
        assert source == "openrouter"
        assert key == "deepseek/deepseek-v4-flash"
        assert fields is not None

    def test_openrouter_unknown_raises(self):
        with pytest.raises(UnknownOpenRouterModel):
            upstream_for("openrouter", "nonexistent/slug", "openrouter", {}, {})

    def test_litellm_egress_resolved(self):
        litellm_db = {
            "deepseek/deepseek-v4-flash": {
                "input_cost_per_token": 1.5e-7, "output_cost_per_token": 6e-7,
            }
        }
        source, key, fields = upstream_for("deepseek", "deepseek-v4-flash", "egress_proxy", litellm_db, {})
        assert source == "litellm"
        assert key == "deepseek/deepseek-v4-flash"
        assert fields["input_per_1m"] == 0.15

    def test_no_match_returns_none(self):
        source, key, fields = upstream_for("unknown", "nonexistent", "direct", {}, {})
        assert source is None
        assert key is None
        assert fields is None

    def test_litellm_prefixes_are_tried_in_order(self):
        """The prefix list in LITELLM_PREFIXES for deepseek tries qualified first."""
        assert LITELLM_PREFIXES["deepseek"][0] == "deepseek/"
        assert LITELLM_PREFIXES["deepseek"][1] == ""


class TestCheckCardAgainstUpstream:
    def test_card_matches_upstream(self):
        litellm_db = {
            "deepseek/deepseek-v4-flash": {
                "input_cost_per_token": 1.5e-7,
                "output_cost_per_token": 6e-7,
            }
        }
        card_fields = {"input_per_1m": 0.15, "output_per_1m": 0.60}
        drifts = check_card_against_upstream(
            "deepseek-v4-flash",
            provider="deepseek",
            route="egress_proxy",
            upstream_model="deepseek-v4-flash",
            card_fields=card_fields,
            litellm_db=litellm_db,
            openrouter_db={},
        )
        assert drifts == []

    def test_card_drift_detected(self):
        litellm_db = {
            "deepseek/deepseek-v4-flash": {
                "input_cost_per_token": 1.5e-7,
                "output_cost_per_token": 6e-7,
            }
        }
        card_fields = {"input_per_1m": 0.20, "output_per_1m": 0.60}
        drifts = check_card_against_upstream(
            "deepseek-v4-flash",
            provider="deepseek",
            route="egress_proxy",
            upstream_model="deepseek-v4-flash",
            card_fields=card_fields,
            litellm_db=litellm_db,
            openrouter_db={},
        )
        assert len(drifts) == 1
        assert drifts[0]["field"] == "input_per_1m"
        assert drifts[0]["ours"] == 0.20
        assert drifts[0]["theirs"] == 0.15


class TestLitellmPrefixes:
    def test_expected_providers_present(self):
        assert "deepseek" in LITELLM_PREFIXES
        assert "anthropic" in LITELLM_PREFIXES
        assert "openrouter" not in LITELLM_PREFIXES
