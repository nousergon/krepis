"""Per-invocation model identifier + token-cost value object.

``ModelMetadata`` is the metadata structure carried on an LLM call's
cost-telemetry stream. It lives in its own module so that multiple
consumers can share one definition — :mod:`krepis.cost` (which translates
token counts into a USD figure) and any external schema that records
model + token metadata alongside a captured result.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelMetadata(BaseModel):
    """Per-invocation model identifier + token cost + run/agent context.

    Token counts are zero-defaulted because some agent paths don't track
    cache reads/creates. ``cost_usd`` is a derived convenience: the load-
    bearing facts are token counts (immutable) and the active price card
    at the time of the call. Use :func:`krepis.cost.recompute_cost` to
    recompute from token counts whenever the rate card changes — never
    treat ``cost_usd`` as canonical for analytics.

    The remaining fields propagate run + agent context through the cost
    telemetry stream so that cost rows can be drilled down by agent,
    sector team, run type, and prompt version. All optional — populated
    by callers as the matching upstream features ship (prompt versioning
    populates ``prompt_id`` + ``prompt_version``; the LangGraph node
    wrapper populates ``node_name``; the run-orchestrator populates
    ``run_type`` + ``sector_team_id``).
    """

    model_config = ConfigDict(extra="forbid")

    model_name: str
    model_version: str | None = None
    # Which provider served the call ("anthropic" default preserves the
    # pre-multi-provider record shape). Drives provider-scoped tool-fee
    # naming in :func:`krepis.cost.recompute_cost` (e.g. an OpenRouter
    # web search bills at the "openrouter:web_search" fee, not
    # Anthropic's "web_search" rate). Additive within the schema.
    provider: str = "anthropic"
    # USD cost the provider itself reported for the call (OpenRouter
    # returns it in ``usage.cost`` when the request opts in). Preferred
    # over card-derived recompute by :func:`krepis.cost.record_llm_call`
    # — with :floor routing the actually-routed backend's price varies
    # below our card ceilings, so the aggregator's number is canonical.
    # ``None`` = not reported; fall back to the price card.
    provider_reported_cost_usd: float | None = None
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cache_read_tokens: int = Field(default=0, ge=0)
    # Cache-write tokens split by TTL. ``cache_create_tokens`` is the
    # 5-minute (default-TTL) slice; ``cache_create_1h_tokens`` is the
    # 1-hour-TTL slice (Anthropic ``usage.cache_creation.
    # ephemeral_1h_input_tokens``). Zero-defaulted so callers that don't
    # use the 1-hour TTL — the common case — omit it harmlessly. Priced
    # separately by :func:`krepis.cost.recompute_cost` at the 1-hour rate
    # when the active price card carries one. Additive within the schema.
    cache_create_tokens: int = Field(default=0, ge=0)
    cache_create_1h_tokens: int = Field(default=0, ge=0)
    # Input tokens that MISSED the cache, on providers that report it
    # directly (DeepSeek's ``prompt_cache_miss_tokens``). This is the
    # denominator half of the cache-hit rate that
    # nous-ergon-ops/policies/prompt-caching-policy.md §6 makes a
    # first-class metric:
    #
    #     hit_rate = cache_read / (cache_read + cache_miss)
    #
    # It is NOT redundant with ``input_tokens``. On Anthropic (M1,
    # explicit breakpoints) ``input_tokens`` already IS the uncached
    # remainder, so the rate is computable without this field. On
    # automatic-prefix providers (M2 — DeepSeek, Moonshot, Zhipu, which
    # carry the fleet's highest-volume traffic) the provider reports hit
    # and miss as its own pair, and without the miss half the rate can
    # only be approximated. Zero-defaulted and additive within the
    # schema; zero means "not reported by this provider", which is why
    # consumers must treat a zero denominator as unknown rather than as
    # a 100% hit rate.
    prompt_cache_miss_tokens: int = Field(default=0, ge=0)
    # The share of ``output_tokens`` spent on chain-of-thought, where the
    # provider reports it (OpenAI-shape
    # ``completion_tokens_details.reasoning_tokens``). NOT additional to
    # ``output_tokens`` — a subset of it, so pricing must never add the two.
    #
    # It is the quantity a reasoning model's ``max_tokens`` must clear before
    # any content is produced, and until now it was recorded nowhere on a
    # SUCCESSFUL call — visible only in ``krepis.llm._budget_exhausted_error``,
    # i.e. once per outage. Three budget-starvation outages in eight days
    # (alpha-engine-config#6396, I6893, I6858) were each remediated by a guess
    # for exactly that reason. Zero means "not reported by this provider",
    # which on a non-reasoning model is also the true value — consumers must
    # not read zero as evidence a reasoning model spent nothing.
    # Zero-defaulted and additive within the schema.
    reasoning_tokens: int = Field(default=0, ge=0)
    # budget_escalations — how many times this logical call had its
    # ``max_tokens`` ceiling doubled and re-issued because the budget was
    # exhausted before any content was produced. 0 on every healthy call; a
    # call site whose base ceiling is chronically undersized shows up here as
    # a number rather than as the next aborted run
    # (alpha-engine-config-I6917 deliverable 3).
    budget_escalations: int = Field(default=0, ge=0)
    # attempts — transport calls the logical call made. 1 on a clean call, 0
    # only on a record built from a path that never counted one. Without it
    # every summed counter on this row is uninterpretable: a 4-attempt row and
    # a 1-attempt row are indistinguishable (alpha-engine-config-I8334).
    attempts: int = Field(default=0, ge=0)
    # reasoning_tokens_max_attempt — largest reasoning draw of any SINGLE
    # attempt, as distinct from `reasoning_tokens`, which is their sum.
    # `max_tokens` bounds one attempt, so THIS is the figure a ceiling must
    # clear; the sum is what was billed.
    reasoning_tokens_max_attempt: int = Field(default=0, ge=0)
    # The registry entry the call ADDRESSED, when it was resolved from the
    # model registry. Distinct from ``model_name``, which is the upstream name
    # the provider reports.
    #
    # Three registry entries — `deepseek-v4-flash`, `-low`, `-max` — carry the
    # same upstream model string while declaring `{exclude: true}`,
    # `{effort: low}` and `{effort: max}` respectively. Without this field,
    # spend across three distinct configurations collapses into one row and
    # `{exclude: true}` is indistinguishable from `{effort: max}` downstream —
    # the exact distinction the reasoning-budget class turns on. Measured
    # 2026-08-11: two Think Tank tiers on `med` recorded `deepseek-v4-flash`
    # and read as a misroute to `low` for an hour before the registry explained
    # it. alpha-engine-config-I6908.
    #
    # ``None`` = not resolved from the registry (a hand-built spec), which is a
    # different statement from "resolved and unknown". Additive within schema.
    addressed_registry_id: str | None = None
    # Server-tool request counts (Anthropic ``Message.usage.server_tool_use``).
    # Distinct from token classes — these are flat per-request fees billed
    # via :class:`krepis.cost.ToolFee`, not the per-1M-token rate on the
    # price card. Zero-defaulted so consumers that don't use server tools
    # omit the field harmlessly. Additive within schema v2.
    web_search_requests: int = Field(default=0, ge=0)
    web_fetch_requests: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    run_type: Literal["weekly_research", "morning", "EOD"] | None = None
    node_name: str | None = None
    sector_team_id: str | None = None
    prompt_id: str | None = None
    prompt_version: str | None = None
