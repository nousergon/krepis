# Security Policy

## Reporting a vulnerability

If you find a security vulnerability in krepis, please report it privately:

- **Preferred:** open a [GitHub Security Advisory](https://github.com/nousergon/krepis/security/advisories/new). This keeps the discussion private until a fix ships.
- **Alternative:** email `security@nousergon.ai` with a description and reproduction steps.

Please **do not** open a public issue for security reports. I aim to acknowledge within 72 hours and ship a fix or mitigation within 14 days for high-severity issues.

## Scope

krepis is a library of general-purpose Python primitives installed via `pip`. The following are in scope:

- **Credential exposure**: any path that writes a secret to a log, error message, or serialized output.
- **SSM misconfiguration**: parameter paths that leak across regions, missing KMS encryption, confused-deputy risks in the SSM dispatcher.
- **Injection**: command injection in the SSM dispatcher or alerts CLI, template injection in Telegram/SNS/email formatting.
- **Dependency supply chain**: vulnerabilities in pinned or declared dependencies that krepis exposes through its public API.
- **SSRF**: any transport (`http_retry`, `anthropic`, `openai`, `webpush`, `email_sender`) that can be tricked into contacting an attacker-controlled host.

The following are **out of scope**:

- Vulnerabilities in upstream dependencies that have not been disclosed publicly — please report those to the upstream project first.
- Issues that require local filesystem access (krepis is a library; if your machine is compromised, the library's threat model has already failed).
- DoS via unbounded retry or memory consumption — krepis provides configuration knobs; set them to match your environment.

## Supported versions

Only the latest published version receives security patches. Pin to a specific version and monitor releases.
