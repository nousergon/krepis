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

## What a change has to satisfy

These are enforced in review, and they exist because krepis is a dependency
of pipelines that run unattended — a defect here surfaces several repos away,
long after the change that caused it.

**Fail loud.** A swallowed exception becomes a silent wrong answer somewhere
downstream. Bare `except: pass`, a silent `return None`, and graceful-degrade
on a writer are all defects — raise instead. Where a swallow is genuinely
correct, an inline comment must name what is being swallowed and where it is
recorded.

**The public API is additive-only.** Consumers pin krepis, so renaming or
removing an exported name breaks them at import time — in a way their own
tests will not catch until deploy. Emit both names for a release, migrate,
then remove. Never a same-commit rename.

**Tests are hermetic.** No network, no AWS calls, no wall-clock dependence.
Anything reaching outside the process gets a fixture or a fake.

**Tests pin contracts, not incidental values.** A test asserting a specific
upstream price breaks on a legitimate price change and teaches people to
ignore it. Assert the behaviour, derive the expected value where it is data.

## Scope

krepis holds only **general-purpose** primitives — logging, secrets,
alerts, retry, locks, telemetry, calendar/date helpers, AWS plumbing.
Anything domain- or application-specific belongs in your own code, not
here. If a proposed addition is specific to one application rather than
broadly reusable, it is out of scope for krepis.
