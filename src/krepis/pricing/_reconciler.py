"""
Generic upstream-price reconciler — fetch, normalise, compare.

**Why this exists.** Prices and context limits are *provider facts*, not our
facts. Hand-maintaining them guarantees drift. The two surfaces that consume
these facts — the alpha-engine-config registry and ``krepis.model_pricing.yaml``
— should derive from the same upstream-normalisation-and-comparison core so a
bugfix to the normaliser reaches both consumers at once.

This module implements the **fetch** (HTTP from litellm / OpenRouter),
**normalise** (per-token → per-1M, field-name canonicalisation), and **compare**
(field-by-field with tolerance bands) layers.  It knows nothing about the
registry schema (``pricing_overrides``, ``model_groups``, line-level YAML
editing) or about ``krepis.cost``'s ``PriceTable`` / ``PriceCard`` types —
those are the schema layers each consumer owns.

**Sources:**

- ``litellm`` price database — ``model_prices_and_context_window.json`` from the
  LiteLLM project. Authoritative for ``direct`` and ``egress_proxy`` routes.
- OpenRouter ``/api/v1/models`` — authoritative for ``openrouter`` routes.

**Canonical field names** (the dict keys ``normalize_litellm`` and
``normalize_openrouter`` produce):

    =====================  ====================================================
    Key                    Meaning
    =====================  ====================================================
    input_per_1m           USD per 1,000,000 input tokens
    output_per_1m          USD per 1,000,000 output tokens
    cache_read_per_1m      USD per 1,000,000 cached-read tokens
    max_context_tokens     Maximum input context length
    max_output_tokens      Maximum output (completion) token count
    =====================  ====================================================

Each consumer maps these to its own schema (registry: ``cost_per_1m_input``,
``cost_per_1m_output``, …; ``krepis.cost.PriceCard``: ``input_per_1m``,
``output_per_1m``, …).

**Tolerance bands:**

- LIMITS compare exactly (integers from upstream).
- PRICES get a 2% band because some upstreams are denomintated in non-USD
  currencies and OpenRouter converts at a periodically-refreshed FX rate.
  Without the band, a 0.35% FX swing on ``glm-5.2`` makes CI permanently
  flappy.  The band sits far above FX noise and far below every real error
  class (past errors: 4×, 10×, 40×).
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

_LITELLM_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)
_OPENROUTER_URL = "https://openrouter.ai/api/v1/models"
_HTTP_TIMEOUT = 30

# Tolerance for comparison, by field class.
_REL_TOL = 1e-9                # exact comparison (limits)
_PRICE_REL_TOL = 0.02          # 2% band (FX-denominated upstream prices)

# Canonical price-field names (what the normalisers produce).
_PRICE_FIELDS = (
    "input_per_1m",
    "output_per_1m",
    "cache_read_per_1m",
)
_LIMIT_FIELDS = (
    "max_context_tokens",
    "max_output_tokens",
)
_RECONCILED_FIELDS = _PRICE_FIELDS + _LIMIT_FIELDS

# Provider → litellm key prefixes to try, in order.
LITELLM_PREFIXES = {
    "deepseek": ("deepseek/", ""),
    "xai": ("xai/", ""),
    "moonshot": ("moonshot/", ""),
    "zhipu": ("zhipu/", ""),
    "anthropic": ("anthropic/", ""),
    "gemini": ("gemini/", ""),
}


# ── Errors ──────────────────────────────────────────────────────────────────


class ReconcileError(RuntimeError):
    """Fatal problem loading a source."""


class UnknownOpenRouterModel(ReconcileError):
    """A model slug is absent from OpenRouter's catalogue."""


# ── upstream loaders ────────────────────────────────────────────────────────


def _fetch_json(url: str) -> Any:
    """GET *url* over HTTPS and decode JSON. Any failure is fatal.

    The scheme is enforced rather than assumed: ``urlopen`` will happily
    open ``file:`` and other schemes, so a URL reaching here from anywhere
    but the two module constants could read local disk and pass it off as
    upstream pricing. This function decides prices — the cheap guard is
    worth more than the assumption that callers behave.
    """
    if not url.startswith("https://"):
        raise ReconcileError(
            f"refusing to fetch {url!r}: only https:// sources are permitted"
        )
    try:
        with urllib.request.urlopen(url, timeout=_HTTP_TIMEOUT) as resp:  # noqa: S310 — scheme enforced above
            if resp.status != 200:
                raise ReconcileError(f"{url} returned HTTP {resp.status}")
            return json.loads(resp.read().decode("utf-8"))
    except ReconcileError:
        raise
    except Exception as exc:  # network, DNS, TLS, JSON — all fatal
        raise ReconcileError(f"could not load {url}: {exc}") from exc


def load_litellm_prices(offline: str | None = None) -> dict:
    """Return the litellm price database keyed by litellm model name."""
    if offline:
        return json.loads(Path(offline).read_text())
    return _fetch_json(_LITELLM_URL)


def load_openrouter_models(offline: str | None = None) -> dict:
    """Return OpenRouter's model catalogue keyed by model id (slug)."""
    if offline:
        doc = json.loads(Path(offline).read_text())
    else:
        doc = _fetch_json(_OPENROUTER_URL)
    return {m["id"]: m for m in doc.get("data", []) if "id" in m}


# ── normalisation helpers ───────────────────────────────────────────────────


def per_1m(per_token: Any) -> float | None:
    """Convert an upstream per-token price to per-1M-tokens."""
    if per_token in (None, ""):
        return None
    try:
        return round(float(per_token) * 1_000_000, 10)
    except (TypeError, ValueError):
        return None


def normalise_litellm(card: dict) -> dict:
    """Normalise a litellm price card into canonical field names."""
    return {
        "input_per_1m": per_1m(card.get("input_cost_per_token")),
        "output_per_1m": per_1m(card.get("output_cost_per_token")),
        "cache_read_per_1m": per_1m(
            card.get("cache_read_input_token_cost")
            if card.get("cache_read_input_token_cost") is not None
            else card.get("input_cost_per_token_cache_hit")
        ),
        "max_context_tokens": card.get("max_input_tokens"),
        "max_output_tokens": card.get("max_output_tokens"),
    }


def normalise_openrouter(model: dict) -> dict:
    """Normalise an OpenRouter catalogue entry into canonical field names."""
    pricing = model.get("pricing", {}) or {}
    top = model.get("top_provider", {}) or {}
    return {
        "input_per_1m": per_1m(pricing.get("prompt")),
        "output_per_1m": per_1m(pricing.get("completion")),
        "cache_read_per_1m": per_1m(pricing.get("input_cache_read")),
        "max_context_tokens": model.get("context_length"),
        "max_output_tokens": top.get("max_completion_tokens"),
    }


# ── comparison ──────────────────────────────────────────────────────────────


def values_agree(ours: Any, theirs: Any, *, price_field: bool = False) -> bool:
    """True when *ours* matches *theirs* within the tolerance for the field type.

    Prices get a band (FX-denominated upstreams drift); limits are exact.
    """
    if ours is None or theirs is None:
        return ours is None and theirs is None
    try:
        a, b = float(ours), float(theirs)
    except (TypeError, ValueError):
        return ours == theirs
    if a == b:
        return True
    tol = _PRICE_REL_TOL if price_field else _REL_TOL
    scale = max(abs(a), abs(b))
    return abs(a - b) <= tol * scale


def upstream_for(
    provider: str,
    model: str,
    route: str,
    litellm_db: dict,
    openrouter_db: dict,
) -> tuple[str | None, str | None, dict | None]:
    """Resolve ``(provider, model, route)`` against the route-appropriate source.

    Returns ``(source_name, matched_key, canonical_fields)``; all three are
    ``None`` when the model appears in no upstream source (legitimately possible
    — those stay hand-maintained).

    Raises :exc:`UnknownOpenRouterModel` when an ``openrouter``-route model is
    absent from OpenRouter's catalogue — not merely unpriced, but dead.
    """
    if route == "openrouter":
        hit = openrouter_db.get(model)
        if hit is not None:
            return "openrouter", model, normalise_openrouter(hit)
        raise UnknownOpenRouterModel(
            f"model {model!r} is not in OpenRouter's catalogue — "
            "this entry can never be served. Check the slug against "
            "https://openrouter.ai/api/v1/models."
        )

    # direct / egress_proxy → what the provider charges us directly.
    for prefix in LITELLM_PREFIXES.get(provider, ("",)):
        key = f"{prefix}{model}"
        card = litellm_db.get(key)
        if card is not None:
            return "litellm", key, normalise_litellm(card)
    return None, None, None


# ── pricing-file comparison (for krepis CI) ─────────────────────────────────


def check_card_against_upstream(
    model_name: str,
    *,
    provider: str,
    route: str,
    upstream_model: str,
    card_fields: dict,
    litellm_db: dict,
    openrouter_db: dict,
) -> list[dict]:
    """Compare one rate card's prices against upstream.

    *card_fields* is a dict of price/limit values using canonical field names
    (``input_per_1m``, ``output_per_1m``, …). Returns a list of drift dicts
    (one per disagreeing field), or an empty list when the card agrees with
    upstream.

    Each drift dict has ``model_name``, ``field``, ``ours``, ``theirs``,
    ``source``, and ``key``.

    The card is matched against upstream using ``(provider, upstream_model,
    route)`` — *provider* and *route* are how the card is catalogued in the
    registry; for krepis-packaged cards (no registry entry), the caller
    supplies the metadata it would have carried in the registry.
    """
    try:
        source, key, upstream = upstream_for(
            provider, upstream_model, route, litellm_db, openrouter_db,
        )
    except UnknownOpenRouterModel:
        return [{
            "model_name": model_name,
            "field": "_upstream",
            "ours": "model_pricing.yaml entry",
            "theirs": "NOT IN OPENROUTER CATALOGUE",
            "source": "openrouter",
            "key": upstream_model,
        }]
    if upstream is None:
        return []  # hand-maintained, no upstream to compare against

    drifts = []
    for field in _PRICE_FIELDS + _LIMIT_FIELDS:
        ours = card_fields.get(field)
        theirs = upstream.get(field)
        if values_agree(ours, theirs, price_field=field in _PRICE_FIELDS):
            continue
        if theirs is None:
            continue  # upstream doesn't publish this field
        drifts.append({
            "model_name": model_name,
            "field": field,
            "ours": ours,
            "theirs": theirs,
            "source": source,
            "key": key,
        })
    return drifts
