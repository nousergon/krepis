"""
Unit tests for ``krepis.cost``.

Locks down the price-table contract: per-model effective-date lookup,
ordering invariants, malformed-YAML hard-fail, pure cost math correctness,
and the ``recompute_cost`` overwrite-vs-readonly modes.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from krepis.cost import (
    PriceCard,
    PriceCardLookupError,
    PriceTable,
    PriceTableLoadError,
    ToolFee,
    ToolFeeTable,
    compute_cost,
    load_default_pricing,
    load_default_tool_fees,
    load_pricing,
    load_tool_fees,
    metadata_from_anthropic_message,
    record_anthropic_call,
    recompute_cost,
)
from krepis.model_metadata import ModelMetadata


# ── PriceCard ─────────────────────────────────────────────────────────────


class TestPriceCard:
    def test_minimal(self):
        c = PriceCard(
            model_name="claude-haiku-4-5",
            effective_from=date(2026, 1, 1),
            input_per_1m=1.0,
            output_per_1m=5.0,
            cache_read_per_1m=0.1,
            cache_create_per_1m=1.25,
        )
        assert c.input_per_1m == 1.0

    def test_negative_price_rejected(self):
        with pytest.raises(ValueError):
            PriceCard(
                model_name="x",
                effective_from=date(2026, 1, 1),
                input_per_1m=-1.0,
                output_per_1m=0.0,
                cache_read_per_1m=0.0,
                cache_create_per_1m=0.0,
            )

    def test_extra_field_rejected(self):
        with pytest.raises(ValueError):
            PriceCard(
                model_name="x",
                effective_from=date(2026, 1, 1),
                input_per_1m=1.0,
                output_per_1m=1.0,
                cache_read_per_1m=0.0,
                cache_create_per_1m=0.0,
                undocumented="oops",
            )


# ── PriceTable ordering invariants ────────────────────────────────────────


def _card(model_name: str, year: int, month: int, day: int, *, in_p: float = 1.0) -> PriceCard:
    return PriceCard(
        model_name=model_name,
        effective_from=date(year, month, day),
        input_per_1m=in_p,
        output_per_1m=in_p * 5,
        cache_read_per_1m=in_p * 0.1,
        cache_create_per_1m=in_p * 1.25,
    )


class TestPriceTableValidation:
    def test_single_card_ok(self):
        t = PriceTable(cards=[_card("haiku", 2026, 1, 1)])
        assert len(t.cards) == 1

    def test_multiple_models_independent(self):
        t = PriceTable(cards=[
            _card("haiku", 2026, 1, 1),
            _card("sonnet", 2026, 1, 1),
        ])
        assert len(t.cards) == 2

    def test_two_cards_one_model_ascending_ok(self):
        t = PriceTable(cards=[
            _card("haiku", 2026, 1, 1, in_p=1.0),
            _card("haiku", 2026, 6, 1, in_p=0.5),
        ])
        assert len(t.cards) == 2

    def test_unsorted_rejected(self):
        with pytest.raises(PriceTableLoadError, match="not.*sorted"):
            PriceTable(cards=[
                _card("haiku", 2026, 6, 1),
                _card("haiku", 2026, 1, 1),
            ])

    def test_duplicate_effective_from_rejected(self):
        with pytest.raises(PriceTableLoadError, match="duplicate effective_from"):
            PriceTable(cards=[
                _card("haiku", 2026, 1, 1, in_p=1.0),
                _card("haiku", 2026, 1, 1, in_p=2.0),
            ])


# ── PriceTable.get ────────────────────────────────────────────────────────


class TestPriceTableLookup:
    def setup_method(self):
        self.table = PriceTable(cards=[
            _card("haiku", 2026, 1, 1, in_p=1.0),
            _card("haiku", 2026, 6, 1, in_p=0.5),
            _card("sonnet", 2026, 1, 1, in_p=3.0),
        ])

    def test_returns_active_card_for_query_date(self):
        c = self.table.get("haiku", date(2026, 3, 15))
        assert c.input_per_1m == 1.0  # still on January card

    def test_returns_later_card_after_effective_date(self):
        c = self.table.get("haiku", date(2026, 6, 1))
        assert c.input_per_1m == 0.5  # June card is now active

        c = self.table.get("haiku", date(2026, 8, 1))
        assert c.input_per_1m == 0.5

    def test_accepts_datetime(self):
        c = self.table.get(
            "haiku", datetime(2026, 3, 15, 12, 0, tzinfo=timezone.utc),
        )
        assert c.input_per_1m == 1.0

    def test_unknown_model_hard_fails(self):
        with pytest.raises(PriceCardLookupError, match="opus"):
            self.table.get("opus", date(2026, 3, 15))

    def test_query_before_first_effective_date_hard_fails(self):
        with pytest.raises(PriceCardLookupError):
            self.table.get("haiku", date(2025, 12, 31))


class TestPriceTableLookupDatedSnapshotSuffix:
    """Anthropic SDK returns ``Message.model`` in the dated snapshot form
    (e.g. ``claude-haiku-4-5-20251001``) even when the caller requested
    the alias; the YAML is keyed on the alias. Lookup must accept both.
    """

    def setup_method(self):
        self.table = PriceTable(cards=[
            _card("claude-haiku-4-5", 2026, 1, 1, in_p=1.0),
            _card("claude-sonnet-4-6", 2026, 1, 1, in_p=3.0),
        ])

    def test_dated_suffix_falls_back_to_alias(self):
        c = self.table.get("claude-haiku-4-5-20251001", date(2026, 5, 28))
        assert c.input_per_1m == 1.0

    def test_alias_lookup_unchanged(self):
        c = self.table.get("claude-haiku-4-5", date(2026, 5, 28))
        assert c.input_per_1m == 1.0

    def test_exact_dated_match_wins_over_alias_fallback(self):
        # If someone adds a dated card explicitly, it takes precedence.
        table = PriceTable(cards=[
            _card("claude-haiku-4-5", 2026, 1, 1, in_p=1.0),
            _card("claude-haiku-4-5-20251001", 2026, 1, 1, in_p=9.99),
        ])
        c = table.get("claude-haiku-4-5-20251001", date(2026, 5, 28))
        assert c.input_per_1m == 9.99

    def test_unknown_alias_with_dated_suffix_still_hard_fails(self):
        with pytest.raises(
            PriceCardLookupError, match="claude-foo-9-9-20251001"
        ):
            self.table.get("claude-foo-9-9-20251001", date(2026, 5, 28))

    def test_non_dated_suffix_is_not_stripped(self):
        # Bare 8-digit substring without leading dash → no normalization.
        with pytest.raises(PriceCardLookupError):
            self.table.get("claude-haiku-4-5.20251001", date(2026, 5, 28))


# ── compute_cost ──────────────────────────────────────────────────────────


class TestComputeCost:
    def test_pure_input_only(self):
        card = _card("haiku", 2026, 1, 1, in_p=2.0)
        # 1M input tokens × $2/M = $2.00 exactly.
        cost = compute_cost(
            input_tokens=1_000_000,
            output_tokens=0,
            cache_read_tokens=0,
            cache_create_tokens=0,
            card=card,
        )
        assert cost == pytest.approx(2.0)

    def test_combined_token_classes(self):
        card = PriceCard(
            model_name="haiku",
            effective_from=date(2026, 1, 1),
            input_per_1m=1.0,
            output_per_1m=5.0,
            cache_read_per_1m=0.1,
            cache_create_per_1m=1.25,
        )
        # input 4000 × 1.0 / 1M = 0.004
        # output 1200 × 5.0 / 1M = 0.006
        # cache_read 2000 × 0.1 / 1M = 0.0002
        # cache_create 500 × 1.25 / 1M = 0.000625
        # total = 0.010825
        cost = compute_cost(
            input_tokens=4000,
            output_tokens=1200,
            cache_read_tokens=2000,
            cache_create_tokens=500,
            card=card,
        )
        assert cost == pytest.approx(0.010825)

    def test_zero_tokens_is_zero_cost(self):
        card = _card("haiku", 2026, 1, 1, in_p=10.0)
        cost = compute_cost(
            input_tokens=0,
            output_tokens=0,
            cache_read_tokens=0,
            cache_create_tokens=0,
            card=card,
        )
        assert cost == 0.0


# ── recompute_cost ────────────────────────────────────────────────────────


class TestRecomputeCost:
    def setup_method(self):
        self.table = PriceTable(cards=[
            _card("haiku", 2026, 1, 1, in_p=1.0),
            _card("haiku", 2026, 6, 1, in_p=0.5),
        ])

    def test_overwrites_cost_usd_by_default(self):
        m = ModelMetadata(
            model_name="haiku",
            input_tokens=1_000_000,
            output_tokens=0,
            cost_usd=0.0,
        )
        cost = recompute_cost(m, self.table, at=date(2026, 3, 15))
        assert cost == pytest.approx(1.0)
        assert m.cost_usd == pytest.approx(1.0)

    def test_readonly_does_not_mutate(self):
        m = ModelMetadata(
            model_name="haiku",
            input_tokens=1_000_000,
            cost_usd=99.99,
        )
        cost = recompute_cost(
            m, self.table, at=date(2026, 3, 15), overwrite=False,
        )
        assert cost == pytest.approx(1.0)
        assert m.cost_usd == 99.99  # unchanged

    def test_uses_historical_card_for_historical_at(self):
        # Same call repriced against January vs June rates returns
        # different cost — proves effective-date routing works.
        m = ModelMetadata(model_name="haiku", input_tokens=1_000_000)
        jan_cost = recompute_cost(
            m, self.table, at=date(2026, 3, 15), overwrite=False,
        )
        jun_cost = recompute_cost(
            m, self.table, at=date(2026, 8, 1), overwrite=False,
        )
        assert jan_cost == pytest.approx(1.0)
        assert jun_cost == pytest.approx(0.5)

    def test_unknown_model_hard_fails(self):
        m = ModelMetadata(model_name="opus", input_tokens=1000)
        with pytest.raises(PriceCardLookupError):
            recompute_cost(m, self.table, at=date(2026, 3, 15))

    def test_default_at_uses_now(self):
        # No `at=` → now(UTC). The June card should resolve since today is
        # past 2026-06-01 in this test environment, but this test is
        # date-of-running-dependent. We just verify the call succeeds and
        # picks SOME valid card without raising.
        m = ModelMetadata(model_name="haiku", input_tokens=1000)
        cost = recompute_cost(m, self.table)
        assert cost > 0


# ── load_pricing ──────────────────────────────────────────────────────────


class TestLoadPricing:
    def test_loads_valid_yaml(self, tmp_path):
        yaml_path = tmp_path / "model_pricing.yaml"
        yaml_path.write_text(
            "cards:\n"
            "  - model_name: claude-haiku-4-5\n"
            "    effective_from: 2026-01-01\n"
            "    input_per_1m: 1.0\n"
            "    output_per_1m: 5.0\n"
            "    cache_read_per_1m: 0.1\n"
            "    cache_create_per_1m: 1.25\n"
        )
        table = load_pricing(yaml_path)
        assert len(table.cards) == 1
        assert table.cards[0].model_name == "claude-haiku-4-5"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_pricing(tmp_path / "no_such_file.yaml")

    def test_missing_cards_key_rejected(self, tmp_path):
        yaml_path = tmp_path / "bad.yaml"
        yaml_path.write_text("not_cards: []\n")
        with pytest.raises(PriceTableLoadError, match="cards"):
            load_pricing(yaml_path)

    def test_top_level_list_rejected(self, tmp_path):
        yaml_path = tmp_path / "bad.yaml"
        yaml_path.write_text("- model_name: x\n")
        with pytest.raises(PriceTableLoadError):
            load_pricing(yaml_path)

    def test_extra_card_field_rejected(self, tmp_path):
        yaml_path = tmp_path / "bad.yaml"
        yaml_path.write_text(
            "cards:\n"
            "  - model_name: claude-haiku-4-5\n"
            "    effective_from: 2026-01-01\n"
            "    input_per_1m: 1.0\n"
            "    output_per_1m: 5.0\n"
            "    cache_read_per_1m: 0.1\n"
            "    cache_create_per_1m: 1.25\n"
            "    undocumented: 999\n"
        )
        with pytest.raises(ValueError):
            load_pricing(yaml_path)

    def test_unsorted_cards_rejected(self, tmp_path):
        yaml_path = tmp_path / "bad.yaml"
        yaml_path.write_text(
            "cards:\n"
            "  - model_name: haiku\n"
            "    effective_from: 2026-06-01\n"
            "    input_per_1m: 0.5\n"
            "    output_per_1m: 2.5\n"
            "    cache_read_per_1m: 0.05\n"
            "    cache_create_per_1m: 0.625\n"
            "  - model_name: haiku\n"
            "    effective_from: 2026-01-01\n"
            "    input_per_1m: 1.0\n"
            "    output_per_1m: 5.0\n"
            "    cache_read_per_1m: 0.1\n"
            "    cache_create_per_1m: 1.25\n"
        )
        with pytest.raises(PriceTableLoadError, match="not.*sorted"):
            load_pricing(yaml_path)


# ── load_default_pricing (packaged YAML) ──────────────────────────────────


class TestLoadDefaultPricing:
    def test_returns_pricetable_with_known_models(self):
        table = load_default_pricing()
        # Packaged file ships cards for the three current frontier models.
        # The exact rates may evolve; we only assert the models are present
        # so this test doesn't break on every price update.
        names = {c.model_name for c in table.cards}
        assert "claude-haiku-4-5" in names
        assert "claude-sonnet-4-6" in names
        assert "claude-opus-4-7" in names

    def test_default_card_lookup_works(self):
        table = load_default_pricing()
        card = table.get("claude-sonnet-4-6", date(2026, 5, 25))
        # Sonnet 4.x input rate is $3/M; locked here as a smoke check
        # that the packaged YAML actually parsed.
        assert card.input_per_1m == pytest.approx(3.0)


# ── metadata_from_anthropic_message (SDK adapter) ─────────────────────────


class _FakeServerToolUsage:
    """Duck-typed stand-in for ``anthropic.types.ServerToolUsage``."""

    def __init__(self, *, web_search_requests: int = 0, web_fetch_requests: int = 0):
        self.web_search_requests = web_search_requests
        self.web_fetch_requests = web_fetch_requests


class _FakeUsage:
    """Duck-typed stand-in for ``anthropic.types.Usage``."""

    def __init__(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        cache_read_input_tokens: int | None = None,
        cache_creation_input_tokens: int | None = None,
        server_tool_use: _FakeServerToolUsage | None = None,
    ):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_input_tokens = cache_read_input_tokens
        self.cache_creation_input_tokens = cache_creation_input_tokens
        self.server_tool_use = server_tool_use


class _FakeMessage:
    """Duck-typed stand-in for ``anthropic.types.Message``."""

    def __init__(self, *, model: str, usage: _FakeUsage):
        self.model = model
        self.usage = usage


class TestMetadataFromAnthropicMessage:
    def test_basic_no_cache(self):
        msg = _FakeMessage(
            model="claude-sonnet-4-6",
            usage=_FakeUsage(input_tokens=850, output_tokens=2700),
        )
        m = metadata_from_anthropic_message(msg)
        assert m.model_name == "claude-sonnet-4-6"
        assert m.input_tokens == 850
        assert m.output_tokens == 2700
        assert m.cache_read_tokens == 0
        assert m.cache_create_tokens == 0
        assert m.cost_usd == 0.0  # caller fills via recompute_cost

    def test_with_cache_fields_populated(self):
        msg = _FakeMessage(
            model="claude-haiku-4-5",
            usage=_FakeUsage(
                input_tokens=100,
                output_tokens=200,
                cache_read_input_tokens=1500,
                cache_creation_input_tokens=2000,
            ),
        )
        m = metadata_from_anthropic_message(msg)
        assert m.cache_read_tokens == 1500
        assert m.cache_create_tokens == 2000

    def test_none_cache_fields_zero_default(self):
        # Anthropic SDK returns None on these when caching wasn't used —
        # adapter must zero-default, not propagate None into ModelMetadata
        # (which would fail ge=0 validation).
        msg = _FakeMessage(
            model="claude-haiku-4-5",
            usage=_FakeUsage(
                input_tokens=10,
                output_tokens=20,
                cache_read_input_tokens=None,
                cache_creation_input_tokens=None,
            ),
        )
        m = metadata_from_anthropic_message(msg)
        assert m.cache_read_tokens == 0
        assert m.cache_create_tokens == 0

    def test_model_name_override(self):
        msg = _FakeMessage(
            model="claude-sonnet-4-6-20260101",
            usage=_FakeUsage(input_tokens=1, output_tokens=1),
        )
        m = metadata_from_anthropic_message(msg, model_name="claude-sonnet-4-6")
        assert m.model_name == "claude-sonnet-4-6"

    def test_integrates_with_recompute_cost(self):
        # End-to-end: SDK message → ModelMetadata → recompute against
        # the packaged default rate card. Locks the seam morning-signal
        # (and any other raw-SDK consumer) will actually use.
        msg = _FakeMessage(
            model="claude-sonnet-4-6",
            usage=_FakeUsage(input_tokens=1_000_000, output_tokens=0),
        )
        m = metadata_from_anthropic_message(msg)
        cost = recompute_cost(m, load_default_pricing(), at=date(2026, 5, 25))
        # 1M Sonnet input tokens @ $3/M = $3.00.
        assert cost == pytest.approx(3.0)
        assert m.cost_usd == pytest.approx(3.0)

    def test_captures_server_tool_use_when_present(self):
        msg = _FakeMessage(
            model="claude-sonnet-4-6",
            usage=_FakeUsage(
                input_tokens=100,
                output_tokens=200,
                server_tool_use=_FakeServerToolUsage(
                    web_search_requests=10, web_fetch_requests=3,
                ),
            ),
        )
        m = metadata_from_anthropic_message(msg)
        assert m.web_search_requests == 10
        assert m.web_fetch_requests == 3

    def test_server_tool_use_absent_zero_default(self):
        # Most messages have no server-tool use; SDK leaves the field None.
        msg = _FakeMessage(
            model="claude-sonnet-4-6",
            usage=_FakeUsage(input_tokens=100, output_tokens=200),
        )
        m = metadata_from_anthropic_message(msg)
        assert m.web_search_requests == 0
        assert m.web_fetch_requests == 0


# ── ToolFee ───────────────────────────────────────────────────────────────


def _fee(tool_name: str, year: int, month: int, day: int, *, rate: float = 10.0) -> ToolFee:
    return ToolFee(
        tool_name=tool_name,
        effective_from=date(year, month, day),
        per_1k_requests_usd=rate,
    )


class TestToolFee:
    def test_minimal(self):
        f = _fee("web_search", 2026, 1, 1, rate=10.0)
        assert f.per_1k_requests_usd == 10.0

    def test_negative_rate_rejected(self):
        with pytest.raises(ValueError):
            ToolFee(
                tool_name="web_search",
                effective_from=date(2026, 1, 1),
                per_1k_requests_usd=-1.0,
            )

    def test_extra_field_rejected(self):
        with pytest.raises(ValueError):
            ToolFee(
                tool_name="x",
                effective_from=date(2026, 1, 1),
                per_1k_requests_usd=1.0,
                undocumented="oops",
            )


class TestToolFeeTableValidation:
    def test_unsorted_rejected(self):
        with pytest.raises(PriceTableLoadError, match="not.*sorted"):
            ToolFeeTable(fees=[
                _fee("web_search", 2026, 6, 1),
                _fee("web_search", 2026, 1, 1),
            ])

    def test_duplicate_effective_from_rejected(self):
        with pytest.raises(PriceTableLoadError, match="duplicate effective_from"):
            ToolFeeTable(fees=[
                _fee("web_search", 2026, 1, 1, rate=10.0),
                _fee("web_search", 2026, 1, 1, rate=12.0),
            ])

    def test_multiple_tools_independent(self):
        t = ToolFeeTable(fees=[
            _fee("web_search", 2026, 1, 1, rate=10.0),
            _fee("web_fetch", 2026, 1, 1, rate=0.0),
        ])
        assert len(t.fees) == 2


class TestToolFeeTableLookup:
    def setup_method(self):
        self.table = ToolFeeTable(fees=[
            _fee("web_search", 2026, 1, 1, rate=10.0),
            _fee("web_search", 2026, 6, 1, rate=8.0),
            _fee("web_fetch", 2026, 1, 1, rate=0.0),
        ])

    def test_returns_active_fee_for_query_date(self):
        f = self.table.get("web_search", date(2026, 3, 15))
        assert f.per_1k_requests_usd == 10.0

    def test_returns_later_fee_after_effective(self):
        f = self.table.get("web_search", date(2026, 7, 1))
        assert f.per_1k_requests_usd == 8.0

    def test_unknown_tool_hard_fails(self):
        with pytest.raises(PriceCardLookupError, match="code_execution"):
            self.table.get("code_execution", date(2026, 3, 15))


# ── load_default_tool_fees ────────────────────────────────────────────────


class TestLoadDefaultToolFees:
    def test_returns_table_with_known_tools(self):
        table = load_default_tool_fees()
        names = {f.tool_name for f in table.fees}
        assert "web_search" in names
        assert "web_fetch" in names

    def test_web_search_lookup_returns_published_rate(self):
        table = load_default_tool_fees()
        fee = table.get("web_search", date(2026, 5, 25))
        # Published Anthropic rate is $10/1k web_search requests.
        assert fee.per_1k_requests_usd == pytest.approx(10.0)


# ── load_tool_fees (external path) ────────────────────────────────────────


class TestLoadToolFees:
    def test_loads_valid_yaml(self, tmp_path):
        yaml_path = tmp_path / "pricing.yaml"
        yaml_path.write_text(
            "tool_fees:\n"
            "  - tool_name: web_search\n"
            "    effective_from: 2026-01-01\n"
            "    per_1k_requests_usd: 10.0\n"
        )
        t = load_tool_fees(yaml_path)
        assert len(t.fees) == 1

    def test_missing_tool_fees_key_rejected(self, tmp_path):
        yaml_path = tmp_path / "bad.yaml"
        yaml_path.write_text("cards: []\n")  # has cards but not tool_fees
        with pytest.raises(PriceTableLoadError, match="tool_fees"):
            load_tool_fees(yaml_path)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_tool_fees(tmp_path / "nope.yaml")


# ── compute_cost + tool fees ──────────────────────────────────────────────


class TestComputeCostWithToolFees:
    def test_tokens_plus_tool_requests(self):
        card = _card("sonnet", 2026, 1, 1, in_p=3.0)
        web_search_fee = _fee("web_search", 2026, 1, 1, rate=10.0)
        # 1M input × $3/M = $3.00; 100 web_search × $10/1k = $1.00.
        cost = compute_cost(
            input_tokens=1_000_000,
            output_tokens=0,
            cache_read_tokens=0,
            cache_create_tokens=0,
            card=card,
            tool_requests={"web_search": 100},
            tool_fees={"web_search": web_search_fee},
        )
        assert cost == pytest.approx(4.0)

    def test_zero_tool_requests_no_fee_required(self):
        # Zero count short-circuits; tool_fees not required.
        card = _card("sonnet", 2026, 1, 1, in_p=3.0)
        cost = compute_cost(
            input_tokens=1_000_000,
            output_tokens=0,
            cache_read_tokens=0,
            cache_create_tokens=0,
            card=card,
            tool_requests={"web_search": 0},
            tool_fees=None,
        )
        assert cost == pytest.approx(3.0)

    def test_nonzero_count_without_fee_hard_fails(self):
        card = _card("sonnet", 2026, 1, 1, in_p=3.0)
        with pytest.raises(PriceCardLookupError, match="web_search"):
            compute_cost(
                input_tokens=0,
                output_tokens=0,
                cache_read_tokens=0,
                cache_create_tokens=0,
                card=card,
                tool_requests={"web_search": 5},
                tool_fees=None,
            )


# ── recompute_cost + tool fees ────────────────────────────────────────────


class TestRecomputeCostWithToolFees:
    def setup_method(self):
        self.price_table = PriceTable(cards=[
            _card("sonnet", 2026, 1, 1, in_p=3.0),
        ])
        self.tool_fees = ToolFeeTable(fees=[
            _fee("web_search", 2026, 1, 1, rate=10.0),
            _fee("web_fetch", 2026, 1, 1, rate=0.0),
        ])

    def test_with_web_search_requests(self):
        m = ModelMetadata(
            model_name="sonnet",
            input_tokens=1_000_000,
            web_search_requests=50,
        )
        # 1M input × $3/M = $3.00; 50 × $10/1k = $0.50.
        cost = recompute_cost(
            m, self.price_table,
            tool_fee_table=self.tool_fees,
            at=date(2026, 5, 25),
        )
        assert cost == pytest.approx(3.5)
        assert m.cost_usd == pytest.approx(3.5)

    def test_web_fetch_priced_at_zero_still_works(self):
        # web_fetch is currently free; non-zero count + zero rate = zero fee.
        m = ModelMetadata(
            model_name="sonnet",
            input_tokens=0,
            web_fetch_requests=100,
        )
        cost = recompute_cost(
            m, self.price_table,
            tool_fee_table=self.tool_fees,
            at=date(2026, 5, 25),
        )
        assert cost == pytest.approx(0.0)

    def test_missing_tool_fee_table_hard_fails(self):
        # Non-zero tool requests + no tool_fee_table → loud failure.
        m = ModelMetadata(
            model_name="sonnet",
            input_tokens=0,
            web_search_requests=5,
        )
        with pytest.raises(PriceCardLookupError, match="server-tool"):
            recompute_cost(m, self.price_table, at=date(2026, 5, 25))

    def test_zero_tool_requests_skips_tool_fee_lookup(self):
        # Pure-LLM call (no server-tool use) doesn't need tool_fee_table.
        m = ModelMetadata(
            model_name="sonnet",
            input_tokens=1_000_000,
        )
        cost = recompute_cost(m, self.price_table, at=date(2026, 5, 25))
        assert cost == pytest.approx(3.0)

    def test_full_e2e_anthropic_message_with_web_search(self):
        # End-to-end: Anthropic SDK message → ModelMetadata → recompute
        # against packaged defaults. Locks the seam any raw-SDK consumer
        # that uses web_search will exercise.
        msg = _FakeMessage(
            model="claude-sonnet-4-6",
            usage=_FakeUsage(
                input_tokens=1_000_000,
                output_tokens=0,
                server_tool_use=_FakeServerToolUsage(web_search_requests=10),
            ),
        )
        m = metadata_from_anthropic_message(msg)
        cost = recompute_cost(
            m,
            load_default_pricing(),
            tool_fee_table=load_default_tool_fees(),
            at=date(2026, 5, 25),
        )
        # 1M Sonnet input @ $3/M + 10 web_search @ $10/1k = $3.10.
        assert cost == pytest.approx(3.10)


# ── record_anthropic_call (capture chokepoint, v0.33.0) ───────────────────


class TestRecordAnthropicCall:
    """Lock down the lifted capture primitive that morning-signal,
    alpha-engine-data, and alpha-engine (executor) all consume in their
    raw-SDK call sites."""

    def test_returns_priced_jsonl_ready_record(self):
        msg = _FakeMessage(
            model="claude-haiku-4-5",
            usage=_FakeUsage(input_tokens=1000, output_tokens=200),
        )
        record = record_anthropic_call(msg)
        # Token cost: (1000 * 1.0 + 200 * 5.0) / 1M = 0.002
        assert record["cost_usd"] == pytest.approx(0.002, abs=1e-6)
        assert record["model"] == "claude-haiku-4-5"
        assert record["input_tokens"] == 1000
        assert record["output_tokens"] == 200
        assert record["cache_read_tokens"] == 0
        assert record["cache_create_tokens"] == 0
        assert record["web_search_requests"] == 0
        assert record["web_fetch_requests"] == 0
        # Timestamp is ISO-8601 round-trippable.
        from datetime import datetime
        datetime.fromisoformat(record["ts"])

    def test_includes_tool_fee_pricing(self):
        msg = _FakeMessage(
            model="claude-haiku-4-5",
            usage=_FakeUsage(
                input_tokens=1000, output_tokens=200,
                server_tool_use=_FakeServerToolUsage(web_search_requests=50),
            ),
        )
        record = record_anthropic_call(msg)
        # Tokens 0.002 + 50 × $10/1k = 0.5 → 0.502
        assert record["cost_usd"] == pytest.approx(0.502, abs=1e-6)
        assert record["web_search_requests"] == 50

    def test_extra_fields_merged(self):
        msg = _FakeMessage(
            model="claude-haiku-4-5",
            usage=_FakeUsage(input_tokens=10, output_tokens=5),
        )
        record = record_anthropic_call(msg, extra_fields={
            "run_id": "2026-05-25",
            "agent_id": "data:news_event_extraction",
            "fingerprint": "abc123",
        })
        assert record["run_id"] == "2026-05-25"
        assert record["agent_id"] == "data:news_event_extraction"
        assert record["fingerprint"] == "abc123"
        # Standard fields preserved alongside extras.
        assert record["model"] == "claude-haiku-4-5"

    def test_extra_fields_can_override_standard_fields(self):
        """Caller-owned keys take precedence — the consumer is the
        authority on what a record should look like in its sink."""
        msg = _FakeMessage(
            model="claude-haiku-4-5",
            usage=_FakeUsage(input_tokens=10, output_tokens=5),
        )
        custom_ts = "2026-05-25T17:30:00+00:00"
        record = record_anthropic_call(msg, extra_fields={"ts": custom_ts})
        assert record["ts"] == custom_ts

    def test_model_name_override_propagates(self):
        msg = _FakeMessage(
            model="claude-haiku-4-5-20251001",
            usage=_FakeUsage(input_tokens=10, output_tokens=5),
        )
        record = record_anthropic_call(msg, model_name="claude-haiku-4-5")
        assert record["model"] == "claude-haiku-4-5"

    def test_uses_default_pricing_when_none_passed(self):
        """Caller without operator-managed pricing gets packaged defaults."""
        msg = _FakeMessage(
            model="claude-sonnet-4-6",
            usage=_FakeUsage(input_tokens=1_000_000, output_tokens=0),
        )
        record = record_anthropic_call(msg)
        # 1M Sonnet input @ $3/M = $3.00 against packaged default rate card.
        assert record["cost_usd"] == pytest.approx(3.0)

    def test_explicit_pricing_table_used(self):
        """Operator-managed pricing wins over defaults when passed."""
        custom_table = PriceTable(cards=[PriceCard(
            model_name="claude-sonnet-4-6",
            effective_from=date(2026, 1, 1),
            input_per_1m=99.0,
            output_per_1m=99.0,
            cache_read_per_1m=99.0,
            cache_create_per_1m=99.0,
        )])
        msg = _FakeMessage(
            model="claude-sonnet-4-6",
            usage=_FakeUsage(input_tokens=1_000_000, output_tokens=0),
        )
        record = record_anthropic_call(msg, pricing=custom_table)
        assert record["cost_usd"] == pytest.approx(99.0)

    def test_at_kwarg_threads_to_recompute(self):
        """Historical recompute path: caller passes capture timestamp."""
        msg = _FakeMessage(
            model="claude-haiku-4-5",
            usage=_FakeUsage(input_tokens=1000, output_tokens=0),
        )
        record = record_anthropic_call(msg, at=date(2026, 5, 25))
        # Whatever the at= date evaluates to, no PriceCardLookupError raised
        # is the load-bearing assertion — we have a packaged-default card
        # effective 2026-01-01.
        assert record["cost_usd"] > 0

