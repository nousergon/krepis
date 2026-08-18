"""
Pricing primitives for the Alpha Engine cost-telemetry and rate-card pipelines.

``krepis.cost`` is the main module — ``PriceCard``/``PriceTable``/``ToolFee``
types plus cost-computation math.  This subpackage is split out from the flat
``krepis.`` namespace as new pricing capabilities are added:

- :mod:`krepis.pricing._reconciler` — generic upstream-price reconciler
  (fetch from litellm/OpenRouter, normalise, compare).  Used by
  ``alpha-engine-config/scripts/reconcile_llm_model_registry.py`` and by
  krepis's own ``model_pricing.yaml`` drift check.
"""
