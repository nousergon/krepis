"""krepis — general-purpose primitives for production data and LLM pipelines.

krepis (Greek κρηπίς, the foundation course a structure stands on) is a
small, typed library of building blocks for services running on AWS:
structured logging, SSM-backed secrets, Telegram/SNS/Web Push alert routing,
bounded-backoff HTTP retry, S3 conditional-PUT writer locks, the Anthropic
payload chokepoint, the provider-agnostic LLM adapter (``krepis.llm`` —
Anthropic / OpenAI / OpenRouter behind one call surface with runtime
SSM-flippable model specs), LLM cost telemetry, trading-calendar/date
helpers, and the LLM-as-judge transport core (``krepis.judge`` — rubric
rendering, structured-output tool spec, batch custom_id codec, and the
judge-model pin/re-anchor registry).
"""

__version__ = "0.18.1"
