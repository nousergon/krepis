"""The resolve contract, as a model — ``resolve_schema.json`` is generated from it.

``model-router-policy.md`` R17 requires the router's answer to "how do I call
group X" to be a **versioned, schema'd contract, with the schema stored next to
the producer**. That schema existed, and it was hand-written: a JSON file
maintained in parallel with the dicts
:func:`krepis.router.resolve_group_structured` and
:func:`krepis.router.resolve_model_structured` actually build.

That is the same shape R6/R6a forbids one layer down. Two hand-synchronised
descriptions of one fact drift, and the drift is invisible until a consumer
fails validation on a field the producer has been emitting for weeks — or
worse, passes validation against a schema that no longer describes the wire.
It had already happened: the committed ``exec_context`` enum listed
``laptop``/``ec2``/``lambda`` while :data:`krepis.router.EXEC_CONTEXTS` had
carried ``ci`` since ``alpha-engine-config-I7853``, so a resolution performed
from a GitHub Actions runner — a supported, declared context — did not
validate against the contract it satisfies.

So the schema is no longer authored. It is **rendered from this module** by
``scripts/gen_resolve_schema.py`` and committed, and
``tests/test_resolve_contract_schema.py`` fails the pull request whose
committed bytes differ from a fresh render. The vocabulary (execution
contexts, wire formats) is imported from :mod:`krepis.router` rather than
restated here, so adding a context is one edit plus a regeneration, and
forgetting the regeneration is a red check rather than a silent divergence.

Evolution stays additive-then-remove (R19): add the new field as optional,
emit both names for one release, migrate consumers, then remove the old field
and raise :data:`krepis.router.RESOLVE_SCHEMA_VERSION`. Never a same-commit
rename — that has broken every consumer twice already
(``alpha-engine-config-I4453``, and the ``resolve-group`` → ``resolve`` CLI
rename before it).

This module deliberately does NOT validate at resolve time. It describes the
contract; the producer builds plain dicts, because a consumer parsing JSON off
a CLI gets a dict and the thing under test must be the dict that crosses the
boundary, not an object that happens to serialise to one.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .router import EXEC_CONTEXTS, RESOLVE_SCHEMA_VERSION, WIRE_FORMATS

__all__ = [
    "ResolveCapabilities",
    "ResolveCachePricing",
    "ResolveContract",
    "SkippedEntry",
    "SCHEMA_ID",
    "render_resolve_schema",
    "resolve_json_schema",
]

#: ``$id`` of the generated schema. Carries the contract version, so a
#: consumer pinning a schema by URL pins a version rather than "latest".
SCHEMA_ID = (
    f"https://nousergon.dev/schemas/krepis-router-resolve-v{RESOLVE_SCHEMA_VERSION}.json"
)

#: Routes a resolution may name. Not derived from a router constant because
#: there is no single one: ``_LEGACY_CLI_ENDPOINTS`` keys the migration shim,
#: not the vocabulary, and the registry's own ``route`` field is the authority.
RESOLVE_ROUTES = ("litellm_proxy", "egress_proxy", "openrouter", "direct")

#: What KIND of name ``model``/``deployment_id`` carries.
WIRE_ADDRESSING = ("group", "capability_group", "deployment")

#: Which credential the caller must present. Mirrors the
#: ``auth_token_type`` -> secret-name table in :mod:`krepis.router`.
AUTH_TOKEN_TYPES = (
    "placeholder",
    "openrouter_key",
    "litellm_master_key",
    "direct_api_key",
)


class ResolveCapabilities(BaseModel):
    """Per-model feature flags as declared in the registry."""

    model_config = ConfigDict(extra="allow")

    web_search: Optional[bool] = None
    tool_choice: Optional[bool] = None
    prompt_caching: Optional[bool] = None
    automatic_prefix_caching: Optional[bool] = None
    batches: Optional[bool] = None
    streaming: Optional[bool] = None


class ResolveCachePricing(BaseModel):
    """Price card for the resolved entry, in USD per 1M tokens."""

    model_config = ConfigDict(extra="allow")

    cost_per_1m_input: Optional[float] = None
    cost_per_1m_output: Optional[float] = None
    cost_per_1m_cache_read: Optional[float] = None
    cost_per_1m_cache_write: Optional[float] = None


class SkippedEntry(BaseModel):
    """One registry entry the resolver did not use, and why."""

    model_config = ConfigDict(extra="allow")

    registry_id: Optional[str] = None
    provider: Optional[str] = None
    reason: Optional[str] = None


class ResolveContract(BaseModel):
    """The routing decision returned by ``krepis.router.resolve_group_structured()``
    and printed by ``python3 -m krepis.router resolve <group> --json``.

    This is a CROSS-REPO CONTRACT: consumers live in alpha-engine-config (groom
    driver, groom_run.sh, disposition audit, reviewed-merge sweep) and
    claude-code-config (the clauder wrapper). Evolve it ADDITIVELY -- emit a new
    field alongside the old for one release, migrate consumers, then remove. A
    same-commit rename has broken every consumer twice
    (alpha-engine-config-I4453, and the resolve-group->resolve CLI rename before
    it).
    """

    # `extra="allow"` on purpose: the deprecated compatibility aliases and any
    # field a newer producer emits must not fail a consumer validating with an
    # older copy of this contract. Additive-then-remove is unimplementable
    # against a closed schema.
    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": SCHEMA_ID,
            "title": "krepis.router resolve contract",
        },
    )

    schema_version: int = Field(
        ge=RESOLVE_SCHEMA_VERSION,
        description=(
            "Contract version. Consumers MUST branch on this rather than "
            "probing for fields."
        ),
    )
    model: str = Field(
        description=(
            "Model identifier to send on the wire. On the litellm_proxy route "
            "from resolve_group_structured() this is a MODEL GROUP -- the bare "
            "group name, or '{group}-cap-{sorted capabilities}' when the caller "
            "declared `requires` -- because a fallback chain is declared on a "
            "group and addressing a concrete deployment silently opts out of it "
            "(alpha-engine-config-I10399, reversing config-I6727). From "
            "resolve_model_structured() it is the registry entry id, which is a "
            "deliberate pin with no chain. Otherwise the concrete registry model "
            "string."
        ),
    )
    display_name: str = Field(
        description=(
            "Human-readable label, e.g. 'deepseek-v4-pro (high)'. Display only "
            "-- never parse."
        ),
    )
    provider: str = Field(
        description=(
            "Upstream provider, or 'litellm' when the central router is serving."
        ),
    )
    route: Literal[RESOLVE_ROUTES] = Field(  # type: ignore[valid-type]
        description="Which transport serves this group.",
    )
    api_base_url: str = Field(
        description=(
            "Base URL the client must target. Empty string means the provider "
            "SDK default (only valid for route=direct). Consumers MUST fail "
            "closed rather than treating a missing value as empty -- an empty "
            "base URL silently targets api.anthropic.com."
        ),
    )
    anthropic_base_url: Optional[str] = Field(
        default=None,
        deprecated=True,
        description=(
            "DEPRECATED alias for api_base_url, emitted through schema_version 2 "
            "only so unmigrated consumers keep working. Removed at "
            "schema_version 3. Do not add new reads."
        ),
    )
    deployment_id: str = Field(
        description=(
            "The model identifier to send on the wire (bare name for "
            "egress_proxy, full slug for openrouter; on litellm_proxy a MODEL "
            "GROUP for a group resolution -- see `model` -- or the registry "
            "entry id for a single-model resolution). Kept as an alias of "
            "`model` for consumers that read this key."
        ),
    )
    wire_addressing: Optional[Literal[WIRE_ADDRESSING]] = Field(  # type: ignore[valid-type]
        default=None,
        description=(
            "What KIND of name `model`/`deployment_id` is. 'group' and "
            "'capability_group' are names LiteLLM applies a fallback chain to; "
            "'deployment' addresses one concrete entry and has no chain. A "
            "consumer or auditor can tell the two apart without pattern-matching "
            "the string (alpha-engine-config-I10399). Absent from producers "
            "older than that fix."
        ),
    )
    auth_token_type: Literal[AUTH_TOKEN_TYPES] = Field(  # type: ignore[valid-type]
        description=(
            "Which credential the caller must present. 'placeholder' means the "
            "egress proxy injects the real key."
        ),
    )
    group: str = Field(
        description=(
            "The model group that was resolved (low|med|high|ultra). EMPTY "
            "STRING on a single-model resolution whose id is not a member of any "
            "model_groups chain -- several entries carry a vestigial top-level "
            "`group` field the registry itself does not read, and reporting it "
            "would claim a membership the registry denies."
        ),
    )
    registry_id: str = Field(
        description=(
            "Registry entry id that served this resolution, or "
            "'litellm:group:<group>' for a group resolution on the router edge "
            "(which names no entry -- see primary_registry_id). A single-model "
            "resolution always names the real entry."
        ),
    )
    primary_model: Optional[str] = Field(
        default=None,
        description=(
            "The group's declared primary model. Present on the litellm_proxy "
            "route; the model actually served may differ if the proxy fell back. "
            "On a single-model resolution there is no chain, so this is the "
            "pinned entry's own upstream model string and cannot differ from "
            "what served."
        ),
    )
    primary_registry_id: Optional[str] = Field(
        default=None,
        description=(
            "Registry id of the group's declared primary; on a single-model "
            "resolution, the pinned entry's own id."
        ),
    )
    capabilities: ResolveCapabilities = Field(
        description="Per-model feature flags as declared in the registry.",
    )
    params: Dict[str, Any] = Field(
        description=(
            "Model params from the registry (max_tokens, reasoning, "
            "structured_outputs)."
        ),
    )
    cache_pricing: Optional[ResolveCachePricing] = Field(
        default=None,
        description="Price card for the resolved entry, in USD per 1M tokens.",
    )
    supports_prompt_caching: Optional[bool] = Field(
        default=None,
        description=(
            "True only for explicit cache_control breakpoints. Automatic prefix "
            "caching needs no client markers."
        ),
    )
    automatic_prefix_caching: Optional[bool] = Field(
        default=None,
        description="Server-side transparent prefix caching.",
    )
    supports_automatic_prefix_caching: Optional[bool] = Field(
        default=None,
        deprecated=True,
        description=(
            "Per-entry form of automatic_prefix_caching, emitted on the "
            "per-provider route."
        ),
    )
    skipped_entries: Optional[List[SkippedEntry]] = Field(
        default=None,
        description=(
            "Registry entries skipped during resolution, with the reason. "
            "R12/R29: a consumer that discards this cannot tell a fallback from a "
            "primary, and a resolution that dropped the group's declared primary "
            "logs identically to one that did not. Reasons distinguish 'not "
            "reachable from execution context X', 'no <wire>-wire endpoint "
            "declared', 'egress proxy not reachable' and the deprecated "
            "exclude_route constraint."
        ),
    )
    exec_context: Optional[Literal[EXEC_CONTEXTS]] = Field(  # type: ignore[valid-type]
        default=None,
        description=(
            "The execution context this resolution was performed for "
            "(model-router-policy R29). DECLARED by the caller, never inferred by "
            "the resolver. Entries the registry does not mark reachable_from this "
            "context are filtered out and recorded in skipped_entries. Added at "
            "schema_version 2 as an additive field; consumers predating it ignore "
            "it safely. The enum is rendered from krepis.router.EXEC_CONTEXTS, so "
            "a context added there cannot be missing here."
        ),
    )
    wire: Optional[Literal[WIRE_FORMATS]] = Field(  # type: ignore[valid-type]
        default=None,
        description=(
            "The wire format api_base_url speaks. 'anthropic' is the Claude CLI's "
            "Messages format (the default); 'openai' is the OpenAI-compatible "
            "format programmatic callers on the openai transport use. The same "
            "provider is often reachable at both on different ports, so a "
            "consumer MUST NOT assume api_base_url matches its own transport "
            "without checking this field. Added at schema_version 2 as an "
            "additive field."
        ),
    )


def resolve_json_schema() -> dict:
    """The JSON Schema for :class:`ResolveContract`, as a dict."""
    return ResolveContract.model_json_schema()


def render_resolve_schema() -> str:
    """The exact bytes ``resolve_schema.json`` must contain.

    One function so the generator script and the drift test cannot disagree
    about formatting — a byte-identity test comparing two different renderers
    tests the renderers, not the schema.
    """
    return json.dumps(resolve_json_schema(), indent=2) + "\n"
