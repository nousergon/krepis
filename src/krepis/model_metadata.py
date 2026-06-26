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
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cache_read_tokens: int = Field(default=0, ge=0)
    cache_create_tokens: int = Field(default=0, ge=0)
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
