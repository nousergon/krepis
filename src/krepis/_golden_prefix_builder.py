"""
Helper entry point for the G4 golden-prefix determinism test
(``prompt-caching-policy.md`` §7 G4).

Builds a representative prompt via the real ``build_messages_payload``
path, serializes it, and prints the output for cross-subprocess comparison.

Intended to be invoked via ``subprocess.run`` so the subprocess can set
``PYTHONHASHSEED`` BEFORE the first import — required for hash-order
nondeterminism to manifest.  Not a public API.
"""

import json
import sys
import textwrap

from krepis.anthropic_payload import build_messages_payload


def _representative_tools() -> list[dict]:
    """Return a realistic tool definition list covering schema shapes
    that are vulnerable to nondeterministic iteration."""
    return [
        {
            "name": "analyze_signal",
            "description": "Analyze a trading signal for quality and actionability.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "signal_id": {"type": "string"},
                    "confidence": {"type": "number"},
                    "metrics": {
                        "type": "object",
                        "properties": {
                            "sharpe_ratio": {"type": "number"},
                            "max_drawdown": {"type": "number"},
                        },
                        "required": ["sharpe_ratio"],
                    },
                },
                "required": ["signal_id", "confidence"],
            },
        },
        {
            "name": "search_research",
            "description": "Search the research corpus for relevant findings.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer"},
                    "date_from": {"type": "string"},
                },
                "required": ["query"],
            },
        },
        {
            "name": "compute_risk_metrics",
            "description": "Compute risk metrics for a given portfolio.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "portfolio_id": {"type": "string"},
                    "confidence_level": {"type": "number"},
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["portfolio_id"],
            },
        },
    ]


def _representative_system_prompt() -> str:
    return textwrap.dedent("""\
    You are a financial research assistant. Analyze the provided data and
    respond with structured findings. Always cite sources.

    Today's date: 2026-07-28
    Portfolio: Alpha Engine v4 (growth)
    Risk budget: 0.15 tracking error
    Max position size: 0.05
    Rebalance frequency: weekly
    """).strip()


def _representative_user_content() -> str:
    return (
        "Analyze the following signals for ticker SPY:\n"
        "1. Momentum score: 0.72\n"
        "2. Volatility regime: moderate (VIX 18.4)\n"
        "3. Correlation to portfolio: 0.65\n"
        "4. Expected return: 0.0823 (8.23% annualized)\n"
        "5. Max drawdown observed: -0.184\n"
        "6. Sharpe ratio target: 1.5\n"
        "7. Beta: 1.02, Alpha: 0.034\n"
    )


def build_and_serialize(*, sort_keys: bool = True) -> str:
    """Build a representative prompt and return its JSON serialization.

    Args:
        sort_keys: When True, ``json.dumps`` sorts keys so the output is
            independent of dict insertion order.  Set to False to test
            the ACTUAL insertion-order serialization the SDK would emit.

    Returns:
        JSON string of the payload.
    """
    payload = build_messages_payload(
        model="claude-sonnet-4-5",
        system_prompt=_representative_system_prompt(),
        user_content=_representative_user_content(),
        max_tokens=4096,
        tools=_representative_tools(),
        cache_system=True,
    )
    return json.dumps(payload, indent=2, sort_keys=sort_keys)


def main() -> None:
    """CLI entry point: prints the serialized payload to stdout.

    Usage:
        PYTHONHASHSEED=<N> python -m krepis._golden_prefix_builder
    """
    # Parse --sort-keys / --no-sort-keys from argv for hazard testing.
    sort_keys = True
    for arg in sys.argv[1:]:
        if arg == "--no-sort-keys":
            sort_keys = False

    print(build_and_serialize(sort_keys=sort_keys))


if __name__ == "__main__":
    main()
