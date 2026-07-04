# krepis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)]()
[![Typed](https://img.shields.io/badge/typed-PEP_561-blue.svg)]()

**krepis** (Greek κρηπίς — the foundation course a structure stands on) is a
small, typed library of general-purpose Python primitives for building
production data and LLM pipelines on AWS.

## Install

```bash
pip install krepis
```

## What's inside

| Module | Purpose |
|---|---|
| `logging` | Structured logging + the flow-doctor secret-seeding chokepoint |
| `secrets` | SSM-backed secret resolution |
| `alerts` | Telegram / SNS alert routing with S3-marker dedup (incl. CLI) |
| `telegram` | Telegram transport |
| `email_sender` | SES / SMTP email transport |
| `http_retry` | Bounded-backoff HTTP retry with jitter |
| `locks` | S3 conditional-PUT writer locks |
| `cost` | LLM cost telemetry (token counts × price card → USD) |
| `model_metadata` | `ModelMetadata` value object (per-invocation model + token cost) |
| `anthropic_payload` | Anthropic request-payload construction + validation |
| `metrics` | Typed metric records |
| `trading_calendar` | NYSE trading-day calendar (pure stdlib) |
| `dates` | Calendar / trading-day dual-track date helpers |
| `ec2_spot` | EC2 spot-instance resilience helpers |
| `ssm_dispatcher` | SSM command dispatch |
| `ssm_log_capture` | SSM run-command log capture |
| `usage_pacing` | Linear-pace short-circuit against a rolling reset-window quota |

## License

MIT — see [LICENSE](LICENSE) and [NOTICE](NOTICE). Contributions accepted
under the [DCO](CONTRIBUTING.md) (`git commit -s`).

Part of the [Nous Ergon](https://github.com/nousergon) project.
