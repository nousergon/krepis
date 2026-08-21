"""The single derivation of ``LLM_MODEL_REGISTRY.yaml`` (model-router-policy R6/R6a).

``model-router-policy.md`` §3.2 R6 requires that "whatever code turns the
registry into an in-process router and into a proxy config MUST be the same
code. Two independent mappings of the same registry will drift, and the drift
is invisible until it is a production incident."

Two mappings existed anyway, from the policy's adoption until 2026-08-12:
:func:`krepis.router._parse_registry` and ``alpha-engine-config``'s
``scripts/generate_litellm_proxy_config.py``. The generator did not import
krepis; it said in its own docstring that it reproduced krepis's behaviour and
kept step by hand. **All three divergences that accumulated ran the same way —
the proxy was correct and the in-process router was not:**

===  =========================  ==========================  ===============================
 #   Fact                       In-process router           Proxy generator
===  =========================  ==========================  ===============================
 1   Status filter              none                        excludes deprecated+unavailable
 2   Prefix and credential      branched on ``provider``    branches on ``route``
 3   ``tpm`` below the floor    passed through              clamped to the floor
===  =========================  ==========================  ===============================

Measured against the live registry on 2026-08-12, divergence 1 alone made five
``status: unavailable`` deployments reachable in-process — ``low-gpt-oss-120b``,
``med-qwen3-max``, ``med-deepseek-v4-flash-openrouter-max``, ``high-qwen3-max``
and ``high-deepseek-v4-pro-openrouter-max`` — while the proxy correctly excluded
every one. That is R4 ("a deprecated or forbidden model MUST NOT be reachable at
runtime") violated on one of the two paths, which is exactly the failure R6
predicts and exactly the shape nothing could see.

This module is where that stops. It owns every fact derivable from the
registry: discovery, parsing, status filtering, group ordering, and the
per-deployment litellm parameters. Its consumers own **only their own naming and
serialisation**, which legitimately differ:

* :mod:`krepis.router` builds an in-process :class:`litellm.Router` in which
  every group member is named ``{group}-{mid}``.
* The proxy generator emits every live model under its own id *as well as*
  qualified primaries, because the proxy serves callers krepis does not own.

That seam is the point: the topologies differ on purpose, the *facts* do not.
A consumer that reads ``LLM_MODEL_REGISTRY.yaml`` itself has re-forked the
derivation regardless of how thin its parse looks, and R6a's fork-detection
check exists to fail the pull request that does it.

Credential representation is the one derived fact whose shape is per-consumer,
so it is a parameter rather than a branch: an in-process router needs the key's
**value**, while a proxy config needs litellm's ``os.environ/NAME``
**reference** that the proxy resolves at call time. See :class:`ApiKeyStyle`.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: Statuses excluded from every generated surface.
#:
#: ``deprecated`` is the permanent exit. ``unavailable`` is the
#: funded-depth-honesty state (alpha-engine-config-I6561): the entry stays in
#: the registry as the record of what exists, but a member that cannot serve —
#: an unfunded provider account, a 404ing route — must not be emitted as
#: reachable depth. Flip back to ``active`` when the entry's own retest note
#: says so.
#:
#: Model-router-policy R4 makes this the derivation layer's job specifically:
#: deprecated rows MAY remain in the registry as documentation, and it is
#: derivation that must filter them out.
EXCLUDED_STATUSES = frozenset({"deprecated", "unavailable"})

#: Placeholder credential for any route whose real key lives in the egress
#: proxy (model-router-policy R25 key isolation: the router process holds a
#: placeholder only, and an unusable key present in the process is a standing
#: liability with no upside).
EGRESS_PLACEHOLDER_ENV = "EGRESS_PROXY_PLACEHOLDER"

#: Value used when :data:`EGRESS_PLACEHOLDER_ENV` is unset. litellm requires a
#: non-empty ``api_key`` to construct a client at all, so an empty string here
#: fails the deployment at build time rather than at the (correctly ignored)
#: credential check — the proxy holds the real key. Matches the default the
#: proxy shim exports, deliberately: two different placeholders would make a
#: config diff look meaningful when it is not.
EGRESS_PLACEHOLDER_DEFAULT = "unused-placeholder-see-key-isolation-config3007"  # noqa: S105 — not a credential

#: How a consumer wants credentials rendered.
#:
#: ``"value"`` resolves the environment variable now — what an in-process
#: :class:`litellm.Router` needs. ``"reference"`` emits litellm's
#: ``os.environ/NAME`` indirection — what a proxy config needs, so the running
#: proxy resolves it at call time rather than baking a secret into a file on
#: disk.
API_KEY_STYLES = ("value", "reference")

#: Smallest ``tpm`` a deployment may carry and still be routable. See
#: :func:`declared_tpm` for why a sub-floor value is clamped rather than
#: honoured, and why no default is invented for an undeclared one.
MIN_ADMISSIBLE_TPM = 1_000_000

#: Capabilities a CONSUMER may require of a group at resolve time.
#:
#: A group is a capability tier, not a call shape. Two members of one tier can
#: differ on whether they will accept a given request at all — `low`'s primary
#: (`deepseek-v4-flash-low`) refuses a forced `tool_choice` outright while
#: thinking mode is on, and `glm-4.7-flash` accepts it (both measured
#: 2026-08-20, alpha-engine-config-I7897). The registry has recorded that
#: difference in `capabilities.tool_choice` since the schema was written; what
#: did not exist was any way for the difference to reach ROUTING. It did not,
#: and the eval judge addressed `low`, got the member the registry itself says
#: cannot serve a forced tool call, and took a permanent 400 on every attempt
#: (alpha-engine-config-I7904).
#:
#: Listed here rather than derived from the union of `capabilities` keys on
#: purpose: `prompt_caching` / `cache_min_tokens` / `automatic_prefix_caching`
#: are COST facts a consumer reads off the resolved route, and `batches` names
#: a different API surface entirely. Only a flag that decides whether a request
#: is ACCEPTED belongs here — requiring anything else would silently narrow a
#: chain for a preference rather than a contract.
ROUTABLE_CAPABILITIES = ("tool_choice",)

_MAX_WALK_DEPTH = 8


class RegistryNotFoundError(FileNotFoundError):
    """Raised when no registry file can be located.

    Its own type because fail-closed (model-router-policy R20) is the required
    behaviour and callers must be able to distinguish "no registry" from any
    other ``FileNotFoundError`` raised while reading one.
    """


class CapabilityUnavailableError(RuntimeError):
    """No live member of a group declares a capability the caller requires.

    Raised at RESOLVE time, before any request is built. Fail-closed (R20) is
    the required behaviour: the alternative is to hand back a deployment the
    registry itself says cannot accept the call, which is a permanent 400 the
    caller then re-reads as an availability problem.

    Carries the rejected members and the reason each was rejected, because the
    surfaced error is frequently the only artifact a weekly unattended caller
    leaves behind.
    """

    def __init__(self, group: str, capability: str, rejected: List[Tuple[str, str]]) -> None:
        self.group = group
        self.capability = capability
        self.rejected = rejected
        detail = "; ".join(f"{mid}: {why}" for mid, why in rejected) or "the group has no live members at all"
        super().__init__(
            f"model group {group!r} has no live member declaring "
            f"capabilities.{capability} — {detail}. This is a REGISTRY gap, not "
            f"an availability event: no retry and no fallback can satisfy it. "
            f"Add a member that declares capabilities.{capability}: true, or "
            f"move the call site to a group that has one."
        )


def find_registry(explicit: Optional[Path] = None) -> Path:
    """Locate ``LLM_MODEL_REGISTRY.yaml``. Fails closed.

    Order: *explicit* argument, then ``$LLM_MODEL_REGISTRY_PATH``, then a
    bounded walk up from the working directory looking for
    ``private-docs/LLM_MODEL_REGISTRY.yaml`` (also probing an
    ``alpha-engine-config/`` child at each level, which is what makes this work
    from a sibling repo or a worktree).

    There is exactly ONE source of truth (R1), so a miss raises rather than
    falling back to any built-in list. A hardcoded model list reachable when
    discovery fails is a per-consumer copy of the registry, which is the thing
    R1 forbids.
    """
    if explicit is not None:
        p = Path(explicit)
        if p.exists():
            return p
        raise RegistryNotFoundError(f"registry path {p} does not exist")

    env_path = os.environ.get("LLM_MODEL_REGISTRY_PATH")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p
        raise RegistryNotFoundError(
            "LLM_MODEL_REGISTRY_PATH=%s but that file does not exist" % env_path
        )

    cwd = Path.cwd().resolve()
    for _ in range(_MAX_WALK_DEPTH):
        for candidate in (
            cwd / "private-docs" / "LLM_MODEL_REGISTRY.yaml",
            cwd / "alpha-engine-config" / "private-docs" / "LLM_MODEL_REGISTRY.yaml",
        ):
            if candidate.exists():
                return candidate
        if cwd.parent == cwd:
            break
        cwd = cwd.parent

    raise RegistryNotFoundError(
        "LLM_MODEL_REGISTRY.yaml not found — set LLM_MODEL_REGISTRY_PATH or run "
        "from within a tree containing private-docs/LLM_MODEL_REGISTRY.yaml. "
        "Canonical copy: alpha-engine-config/private-docs/LLM_MODEL_REGISTRY.yaml"
    )


def entry_declares_capability(entry: dict, capability: str) -> bool:
    """True only when *entry* declares *capability* as literal ``True``.

    Absence is NOT capability. An undeclared flag means nobody measured it, and
    treating "unknown" as "supported" is how a group ends up routing a forced
    tool call at a model that refuses one. The registry schema requires the
    flag on every entry; a row missing it is a registry defect the validator
    catches, and until it does, this returns False and the member is skipped
    with a stated reason rather than gambled on.
    """
    return (entry.get("capabilities") or {}).get(capability) is True


class Registry:
    """A parsed registry: the derivation's shared intermediate representation.

    Holds the raw document plus the two views every consumer needs — the live
    model index and, per group, its live members in declared order. Consumers
    render from this; none of them re-reads the file.
    """

    def __init__(self, path: Path, doc: Dict[str, Any]) -> None:
        self.path = path
        self.doc = doc
        self.models: Dict[str, dict] = {
            m["id"]: m for m in (doc.get("models") or []) if "id" in m
        }
        self.groups: Dict[str, List[str]] = dict(doc.get("model_groups") or {})

    @property
    def live_models(self) -> Dict[str, dict]:
        """Every model whose ``status`` is not in :data:`EXCLUDED_STATUSES`.

        A row with no ``status`` counts as live: the field is optional in the
        registry schema, and treating its absence as excluded would silently
        empty the config the first time somebody omitted it.
        """
        return {
            mid: entry
            for mid, entry in self.models.items()
            if entry.get("status") not in EXCLUDED_STATUSES
        }

    def live_group_ids(
        self, group: str, *, requires: Tuple[str, ...] = ()
    ) -> List[str]:
        """Live members of *group*, in declared order, with each drop logged.

        Filtering happens BEFORE the caller enumerates, never inside its loop.
        Filtering inside the loop keys "is this the primary?" off ``i == 0`` of
        the RAW list, so a group whose first member is deprecated generates no
        alias at all and ``completion(model="low")`` fails with "model not
        found" — a worse outcome than the dead member the filter exists to
        remove.

        *requires* narrows further to members declaring EVERY named capability
        (:data:`ROUTABLE_CAPABILITIES`). Status and capability are two different
        rejections and are logged as two different reasons: a member excluded
        for status may come back, a member excluded for capability never will
        for this call shape.
        """
        live: List[str] = []
        for mid, _reason in self._group_members(group, requires=requires):
            if _reason is None:
                live.append(mid)
        if not live:
            logger.warning(
                "group %r has no live members%s — no alias will be generated",
                group,
                f" declaring {list(requires)}" if requires else "",
            )
        return live

    def _group_members(
        self, group: str, *, requires: Tuple[str, ...] = ()
    ) -> Iterator[Tuple[str, Optional[str]]]:
        """Yield ``(member_id, rejection_reason_or_None)`` in declared order."""
        for mid in self.groups.get(group, []):
            entry = self.models.get(mid)
            if entry is None:
                logger.warning(
                    "model %r referenced in group %r is not in the models list", mid, group
                )
                yield mid, "not present in the models list"
                continue
            status = entry.get("status")
            if status in EXCLUDED_STATUSES:
                logger.info(
                    "model %r in group %r is status=%s — excluded from derivation",
                    mid, group, status,
                )
                yield mid, f"status is {status!r}"
                continue
            missing = [cap for cap in requires if not entry_declares_capability(entry, cap)]
            if missing:
                logger.info(
                    "model %r in group %r does not declare %s — excluded from a "
                    "capability-scoped derivation",
                    mid, group, missing,
                )
                yield mid, (
                    "does not declare "
                    + ", ".join(f"capabilities.{c}: true" for c in missing)
                )
                continue
            yield mid, None

    def capability_rejections(
        self, group: str, *, requires: Tuple[str, ...] = ()
    ) -> List[Tuple[str, str]]:
        """Every member of *group* that was rejected, with the reason."""
        return [
            (mid, reason)
            for mid, reason in self._group_members(group, requires=requires)
            if reason is not None
        ]

    def iter_live_groups(
        self, *, requires: Tuple[str, ...] = ()
    ) -> Iterator[Tuple[str, List[str]]]:
        """Yield ``(group, live_member_ids)`` for every group with a live member."""
        for group in self.groups:
            live = self.live_group_ids(group, requires=requires)
            if live:
                yield group, live


def load_registry(path: Optional[Path] = None) -> Registry:
    """Read and parse the registry. The ONLY place its YAML is loaded."""
    import yaml as _yaml

    resolved = path if path is not None else find_registry()
    with open(resolved) as fh:
        doc = _yaml.safe_load(fh)
    if not isinstance(doc, dict):
        raise ValueError(
            "registry at %s did not parse to a mapping (got %s)" % (resolved, type(doc).__name__)
        )
    return Registry(resolved, doc)


# ── Per-entry derived facts ──────────────────────────────────────────────
#
# ROUTE decides the wire format and the credential; PROVIDER only says whose
# model it is. Both helpers below branch on `route` FIRST for that reason.
#
# They used to branch on `provider` alone in one of the two forked
# derivations, which was correct only because every openrouter row also
# carried provider: openrouter. A row with provider: anthropic +
# route: openrouter — the shape you get by moving an existing direct entry
# onto the aggregator — produced `anthropic/anthropic/claude-opus-5` against
# ANTHROPIC_API_KEY: a doubled prefix and the wrong credential, i.e. a 401 on
# a model the picker offers. Caught 2026-07-29 doing exactly that move, fixed
# in the generator, and never carried across to krepis — divergence 2 in this
# module's header.

def litellm_model(entry: dict) -> str:
    """Build the litellm model string: ``openai/deepseek-v4-flash`` etc."""
    route = entry.get("route", "")
    provider = entry.get("provider", "")
    model = entry.get("model", "")

    if route == "openrouter":
        # The vendor is already encoded in the slug (anthropic/claude-opus-5).
        return "openrouter/%s" % model
    if route == "egress_proxy":
        return "openai/%s" % model
    # route: direct — the wire format is the provider's own API.
    if provider == "anthropic":
        return "anthropic/%s" % model
    return "openai/%s" % model


def api_key_env(entry: dict) -> str:
    """Name of the environment variable holding this entry's credential."""
    route = entry.get("route", "")
    provider = entry.get("provider", "")

    if route == "openrouter":
        return "OPENROUTER_API_KEY"
    if route == "egress_proxy":
        # The egress proxy holds the real key and ignores this placeholder (R25).
        return EGRESS_PLACEHOLDER_ENV
    if provider == "anthropic":
        # Only reachable via route: direct. The proxy shim deliberately does NOT
        # resolve ANTHROPIC_API_KEY (config-I4456, model-router-policy R25) — so a
        # direct-route Anthropic entry will 401 until that ruling changes.
        return "ANTHROPIC_API_KEY"
    return EGRESS_PLACEHOLDER_ENV


def api_key_for(entry: dict, style: str = "value", overrides: Optional[Dict[str, str]] = None) -> str:
    """Render this entry's credential in the shape *style* asks for.

    ``overrides`` lets a caller supply a key it already holds (the in-process
    router is constructed with an OpenRouter key resolved from SSM) without
    reaching for the environment. It applies to ``value`` style only —
    a reference is a name, and substituting a secret for it would write the
    secret into the generated config file.
    """
    if style not in API_KEY_STYLES:
        raise ValueError("api key style must be one of %r, got %r" % (API_KEY_STYLES, style))
    name = api_key_env(entry)
    if style == "reference":
        return "os.environ/%s" % name
    if overrides and name in overrides and overrides[name]:
        return overrides[name]
    default = EGRESS_PLACEHOLDER_DEFAULT if name == EGRESS_PLACEHOLDER_ENV else ""
    return os.environ.get(name) or default


def extra_body(entry: dict) -> Optional[dict]:
    """Extract ``params.reasoning`` and OpenRouter provider pinning into
    litellm's ``extra_body``, or None.

    The string ``"null"`` is treated as absent: it is what a YAML author writes
    meaning "no reasoning", and forwarding it verbatim sends the literal four
    characters upstream as a reasoning directive.

    ``openrouter_provider_pinning`` (config#4532) is honoured only for an
    entry whose upstream is actually OpenRouter — OpenRouter requests without
    a pinned provider reselect the upstream per request, which invalidates
    prefix caches. When set, the generated config injects
    ``extra_body.provider`` so every request to this entry targets the same
    (set of) upstream provider(s).

    An entry reaches OpenRouter one of two ways: ``route: openrouter``
    (direct, real key in-process), or ``route: egress_proxy`` with
    ``upstream_host: openrouter.ai`` (proxied — the R25/R26-conformant
    shape, alpha-engine-config-I6286). Both must honour pinning; neither is
    ``provider``, which names the underlying model vendor (e.g.
    ``anthropic``) and is a different field from the routing path.
    """
    result: dict = {}
    params = entry.get("params") or {}
    reasoning = params.get("reasoning")
    if reasoning and reasoning != "null":
        result["reasoning"] = reasoning
    # `thinking` is a SEPARATE upstream control from `reasoning`, not a synonym,
    # and the difference decides whether a forced tool call is accepted at all.
    # Measured against api.deepseek.com through the egress proxy, 2026-08-21,
    # `deepseek-v4-flash` + forced `tool_choice`:
    #
    #   no thinking field            -> 400 Thinking mode does not support this tool_choice
    #   reasoning: {exclude: true}   -> 400 (identical)
    #   thinking: {type: disabled}   -> 200 with tool_calls
    #
    # `reasoning` asks the provider not to BILL or RETURN a reasoning trace;
    # `thinking` turns the mode off. Only the second one changes what the
    # request is allowed to contain. Until this passthrough existed the
    # registry could not express the one setting that works, which is why the
    # group's answer to a tool-calling consumer had to be a different vendor
    # (alpha-engine-config-I7897) rather than a different parameter
    # (alpha-engine-config-I7904).
    thinking = params.get("thinking")
    if thinking and thinking != "null":
        result["thinking"] = thinking
    targets_openrouter = (
        entry.get("route") == "openrouter"
        or entry.get("upstream_host") == "openrouter.ai"
    )
    if targets_openrouter:
        provider_pinning = entry.get("openrouter_provider_pinning")
        if provider_pinning and isinstance(provider_pinning, dict):
            result["provider"] = provider_pinning
    return result or None


def deployment_params(
    entry: dict,
    *,
    api_key_style: str = "value",
    api_key_overrides: Optional[Dict[str, str]] = None,
) -> dict:
    """The litellm params every consumer derives identically from one entry.

    Covers model string, credential, ``api_base`` and the multi-tenant egress
    proxy's ``X-Upstream-Host`` header, ``max_tokens`` and ``extra_body``.

    Deliberately NOT covered, because they are rendering choices rather than
    registry facts: ``rpm``/``tpm`` defaults and floors, and the per-deployment
    HTTP ``timeout``. Those belong to whichever surface is being emitted, and a
    consumer that needs the registry's own declared values reads them off
    *entry* (see :func:`declared_tpm`).
    """
    params: dict = {
        "model": litellm_model(entry),
        "api_key": api_key_for(entry, style=api_key_style, overrides=api_key_overrides),
    }

    if entry.get("route") == "egress_proxy":
        api_base = entry.get("api_base")
        if api_base:
            params["api_base"] = api_base
        else:
            # Fail loud rather than emitting a deployment that cannot reach
            # anything: an egress_proxy entry without a base URL is a registry
            # defect the R2 validator is supposed to catch.
            logger.warning(
                "egress_proxy entry %r has no api_base — requests will fail without a base URL",
                entry.get("id", entry.get("model", "<unknown>")),
            )

    upstream_host = entry.get("upstream_host")
    if upstream_host:
        params.setdefault("extra_headers", {})["X-Upstream-Host"] = upstream_host

    entry_params = entry.get("params") or {}
    if "max_tokens" in entry_params:
        params["max_tokens"] = entry_params["max_tokens"]

    eb = extra_body(entry)
    if eb:
        params["extra_body"] = eb

    return params


def declared_tpm(entry: dict, floor: int) -> Optional[int]:
    """The entry's ``tpm``, raised to *floor*, or None when undeclared.

    ``tpm`` reads as a per-minute budget. Under ``routing_strategy:
    usage-based-routing`` it is not one — litellm applies it as a PER-REQUEST
    ADMISSION GATE (``lowest_tpm_rpm.py``: ``if item_tpm + input_tokens >
    _deployment_tpm: continue``). Any deployment whose tpm is smaller than a
    single prompt is skipped, the selector returns None, and the group raises
    ``RouterRateLimitError: No deployments available … cooldown_list=[]``.
    Nothing is cooling down and nothing recovers, because the next request is
    the same size (alpha-engine-config-I5846).

    Clamping rather than raising is deliberate: a hard failure aborts config
    generation, and the proxy shim's degraded path then serves the
    last-known-good config — which is precisely the config carrying the
    unroutable tpm. The loud surface for a bad value is CI, which blocks it
    before it can be generated; at runtime the safe move is to route.

    Undeclared stays undeclared. A default invented here encodes no real
    provider limit while silently capping usable context, which is how the
    whole ``ultra`` chain became unroutable against a 76k-token prompt.
    """
    tpm = entry.get("tpm")
    if tpm is None:
        return None
    if tpm < floor:
        logger.warning(
            "model %r declares tpm=%s, below the %s admissible floor — a single prompt "
            "larger than this makes the deployment permanently unroutable under "
            "usage-based-routing. Raising to the floor.",
            entry.get("id"), tpm, floor,
        )
        return floor
    return tpm
