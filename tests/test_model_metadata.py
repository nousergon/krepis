"""Unit tests for the ModelMetadata value-object contract."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from krepis.model_metadata import ModelMetadata

NONNEGATIVE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_create_tokens",
    "cache_create_1h_tokens",
    "prompt_cache_miss_tokens",
    "reasoning_tokens",
    "budget_escalations",
    "attempts",
    "reasoning_tokens_max_attempt",
    "web_search_requests",
    "web_fetch_requests",
    "cost_usd",
)


@pytest.fixture
def populated_metadata() -> ModelMetadata:
    return ModelMetadata(
        model_name="test-model",
        model_version="v1",
        provider="openrouter",
        provider_reported_cost_usd=0.125,
        input_tokens=100,
        output_tokens=50,
        cache_read_tokens=200,
        cache_create_tokens=30,
        cache_create_1h_tokens=40,
        prompt_cache_miss_tokens=60,
        reasoning_tokens=20,
        budget_escalations=1,
        attempts=2,
        reasoning_tokens_max_attempt=15,
        addressed_registry_id="test-model-high",
        web_search_requests=3,
        web_fetch_requests=4,
        cost_usd=0.25,
        run_type="weekly_research",
        node_name="research",
        sector_team_id="energy",
        prompt_id="summary",
        prompt_version="v2",
    )


def test_model_name_required() -> None:
    with pytest.raises(ValidationError) as exc:
        ModelMetadata.model_validate({})
    assert exc.value.errors()[0]["loc"] == ("model_name",)


def test_defaults() -> None:
    metadata = ModelMetadata(model_name="test-model")
    assert metadata.model_dump() == {
        "model_name": "test-model",
        "model_version": None,
        "provider": "anthropic",
        "provider_reported_cost_usd": None,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_create_tokens": 0,
        "cache_create_1h_tokens": 0,
        "prompt_cache_miss_tokens": 0,
        "reasoning_tokens": 0,
        "budget_escalations": 0,
        "attempts": 0,
        "reasoning_tokens_max_attempt": 0,
        "addressed_registry_id": None,
        "web_search_requests": 0,
        "web_fetch_requests": 0,
        "cost_usd": 0.0,
        "run_type": None,
        "node_name": None,
        "sector_team_id": None,
        "prompt_id": None,
        "prompt_version": None,
    }


def test_extra_field_rejected() -> None:
    with pytest.raises(ValidationError) as exc:
        ModelMetadata.model_validate({"model_name": "test-model", "unknown": 1})
    assert exc.value.errors()[0]["loc"] == ("unknown",)


@pytest.mark.parametrize("field", NONNEGATIVE_FIELDS)
def test_negative_value_rejected(field: str) -> None:
    with pytest.raises(ValidationError) as exc:
        ModelMetadata.model_validate({"model_name": "test-model", field: -1})
    assert exc.value.errors()[0]["loc"] == (field,)


@pytest.mark.parametrize("field", NONNEGATIVE_FIELDS)
def test_zero_accepted(field: str) -> None:
    metadata = ModelMetadata.model_validate({"model_name": "test-model", field: 0})
    assert metadata.model_dump()[field] == 0


def test_non_numeric_token_count_rejected() -> None:
    with pytest.raises(ValidationError) as exc:
        ModelMetadata.model_validate({
            "model_name": "test-model", "input_tokens": "not-a-number",
        })
    assert exc.value.errors()[0]["loc"] == ("input_tokens",)


@pytest.mark.parametrize("run_type", ["weekly_research", "morning", "EOD"])
def test_run_type_accepts_declared_literals(run_type: str) -> None:
    metadata = ModelMetadata.model_validate({"model_name": "test-model", "run_type": run_type})
    assert metadata.run_type == run_type


@pytest.mark.parametrize("run_type", ["daily", "eod", ""])
def test_run_type_rejects_undeclared_values(run_type: str) -> None:
    with pytest.raises(ValidationError) as exc:
        ModelMetadata.model_validate({"model_name": "test-model", "run_type": run_type})
    assert exc.value.errors()[0]["loc"] == ("run_type",)


def test_dict_round_trip(populated_metadata: ModelMetadata) -> None:
    dumped = populated_metadata.model_dump()
    assert ModelMetadata(**dumped) == populated_metadata


def test_json_round_trip(populated_metadata: ModelMetadata) -> None:
    dumped = populated_metadata.model_dump_json()
    assert ModelMetadata.model_validate_json(dumped) == populated_metadata


def test_copy_update_preserves_untouched_fields(populated_metadata: ModelMetadata) -> None:
    copied = populated_metadata.model_copy(update={"output_tokens": 75})
    assert copied is not populated_metadata
    assert copied.output_tokens == 75
    assert populated_metadata.output_tokens == 50
    assert copied.model_dump(exclude={"output_tokens"}) == populated_metadata.model_dump(
        exclude={"output_tokens"},
    )


def test_fully_populated_values(populated_metadata: ModelMetadata) -> None:
    assert populated_metadata.model_name == "test-model"
    assert populated_metadata.model_version == "v1"
    assert populated_metadata.provider == "openrouter"
    assert populated_metadata.provider_reported_cost_usd == 0.125
    assert populated_metadata.input_tokens == 100
    assert populated_metadata.output_tokens == 50
    assert populated_metadata.cache_read_tokens == 200
    assert populated_metadata.cache_create_tokens == 30
    assert populated_metadata.cache_create_1h_tokens == 40
    assert populated_metadata.prompt_cache_miss_tokens == 60
    assert populated_metadata.reasoning_tokens == 20
    assert populated_metadata.budget_escalations == 1
    assert populated_metadata.attempts == 2
    assert populated_metadata.reasoning_tokens_max_attempt == 15
    assert populated_metadata.addressed_registry_id == "test-model-high"
    assert populated_metadata.web_search_requests == 3
    assert populated_metadata.web_fetch_requests == 4
    assert populated_metadata.cost_usd == 0.25
    assert populated_metadata.run_type == "weekly_research"
    assert populated_metadata.node_name == "research"
    assert populated_metadata.sector_team_id == "energy"
    assert populated_metadata.prompt_id == "summary"
    assert populated_metadata.prompt_version == "v2"
