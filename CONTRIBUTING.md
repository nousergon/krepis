# Contributing to krepis

Thanks for your interest in contributing. krepis is MIT-licensed and we
welcome issues and pull requests.

## Developer Certificate of Origin (DCO)

All contributions to krepis are accepted under the
[Developer Certificate of Origin 1.1](https://developercertificate.org/).
By signing off on your commits you certify that you wrote the code, or
otherwise have the right to submit it under the project's MIT license.

Sign off every commit with a `Signed-off-by` line matching the commit author:

```
git commit -s -m "your message"
```

which appends:

```
Signed-off-by: Your Name <your.email@example.com>
```

Pull requests whose commits are not signed off will be asked to amend before
merge. We do not require a separate CLA — the DCO is the inbound mechanism.

## Inbound = outbound

Contributions are licensed under the same MIT license that covers the project
(see `LICENSE`). Do not submit code you cannot license under MIT.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest -q
```

The test suite must be green on the full supported Python matrix
(3.9–3.13) before a PR is merged. New modules and behaviours need tests.

## Scope

krepis holds only **general-purpose** primitives — logging, secrets,
alerts, retry, locks, telemetry, calendar/date helpers, AWS plumbing.
Anything domain- or application-specific belongs in your own code, not
here. If a proposed addition is specific to one application rather than
broadly reusable, it is out of scope for krepis.
