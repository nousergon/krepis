#!/usr/bin/env python3
"""Live smoke test for the ``krepis.llm`` adapter — one tiny structured
call per configured provider, asserting schema-valid output + a priced
cost record.

OPT-IN: does nothing unless ``KREPIS_LIVE_LLM_SMOKE=1``. Each provider
leg additionally requires its API key env var and is skipped (loudly)
without it. Real tokens are billed — the payloads are tiny (~$0.001/leg).

Usage::

    KREPIS_LIVE_LLM_SMOKE=1 ANTHROPIC_API_KEY=... OPENROUTER_API_KEY=... \
        python scripts/live_llm_smoke.py
"""

from __future__ import annotations

import json
import os
import sys

from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from krepis.cost import record_llm_call  # noqa: E402
from krepis.llm import LLMClient  # noqa: E402
from krepis.llm_config import ModelSpec  # noqa: E402


class Answer(BaseModel):
    answer: int
    confidence: str


LEGS = [
    ModelSpec("anthropic", "claude-haiku-4-5", max_tokens=256),
    ModelSpec("openrouter", "deepseek/deepseek-v4-flash:floor", max_tokens=256),
    ModelSpec("openrouter", "moonshotai/kimi-k2.6", max_tokens=256),
]


def main() -> int:
    if os.environ.get("KREPIS_LIVE_LLM_SMOKE") != "1":
        print("KREPIS_LIVE_LLM_SMOKE != 1 — refusing to spend tokens. Set it to run.")
        return 0

    failures = 0
    for spec in LEGS:
        key_env = spec.resolved_api_key_env()
        if not os.environ.get(key_env):
            print(f"SKIP {spec.provider}:{spec.model} — {key_env} not set")
            continue
        try:
            result = LLMClient(spec).structured(
                system="You are a calculator. Answer precisely.",
                user_content="What is 6 * 7?",
                schema=Answer,
                schema_name="emit_answer",
            )
            record = record_llm_call(result)
            assert result.parsed is not None and result.parsed.answer == 42, (
                f"unexpected structured payload: {result.data}"
            )
            assert record["cost_usd"] >= 0.0
            print(
                f"OK   {spec.provider}:{spec.model} -> answer={result.parsed.answer} "
                f"cost=${record['cost_usd']:.6f} ({record['cost_source']}) "
                f"tokens={record['input_tokens']}/{record['output_tokens']}"
            )
        except Exception as exc:  # noqa: BLE001 — smoke reports every leg
            failures += 1
            print(f"FAIL {spec.provider}:{spec.model} -> {exc}")

    if failures:
        print(f"{failures} leg(s) failed")
        return 1
    print("live LLM smoke complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
