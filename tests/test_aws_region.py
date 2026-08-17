"""Tests for :mod:`krepis.aws_region` — the single region-resolution chain.

alpha-engine-config-I7428: SSM ``AWS-RunShellScript`` runs with no
``AWS_REGION``/``AWS_DEFAULT_REGION`` and no per-user boto profile. A bare
``boto3.client("cloudwatch")`` there raises ``NoRegionError`` — S3 tolerates a
missing region for most operations, CloudWatch and Step Functions do not,
which is why this read as an intermittent, service-specific fault. These
tests reproduce the missing-environment condition directly (clearing both env
vars) rather than only asserting against the resolver's return value, and a
source-scan guard prevents a fifth fork of the region-fallback chain or a new
bare regional-service client from reintroducing the defect.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from krepis import aws_region

SRC_DIR = Path(__file__).resolve().parent.parent / "src" / "krepis"

# S3 is the one AWS service whose client tolerates no region for most
# operations (per the issue's own root-cause note) — the only exempted
# service name for a bare `boto3.client(...)` call.
_REGIONLESS_EXEMPT = {"s3"}


def _iter_boto3_client_calls(text: str) -> list[str]:
    """Return each `boto3.client(...)` / `_boto3.client(...)` call's full
    argument text (paren-balanced), for every occurrence in `text`."""
    calls = []
    for match in re.finditer(r"_?boto3\.client\(", text):
        start = match.end()
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
            i += 1
        calls.append(text[start : i - 1])
    return calls


def test_no_bare_boto3_client_for_a_regional_service() -> None:
    """Every `boto3.client("<service>")` call for a non-S3 service must pass
    `region_name=` — the exact shape that regressed to a bare
    `boto3.client("cloudwatch")` in `stage_coverage.py:606` before this fix.
    A future call site that forgets `region_name=` fails THIS test, not a
    live SSM run three hours into a weekly pipeline."""
    violations: list[str] = []
    for path in sorted(SRC_DIR.glob("*.py")):
        if path.name == "aws_region.py":
            continue  # its docstring quotes the bare-call example verbatim
        text = path.read_text(encoding="utf-8")
        for call_args in _iter_boto3_client_calls(text):
            service_match = re.match(r"""\s*["'](\w[\w-]*)["']""", call_args)
            if not service_match:
                continue  # dynamic service name — can't statically check
            service = service_match.group(1)
            if service in _REGIONLESS_EXEMPT:
                continue
            if "region_name" not in call_args:
                violations.append(f"{path.name}: boto3.client({call_args!r})")
    assert not violations, (
        "bare boto3.client(...) for a regional service with no region_name= "
        "— this is the alpha-engine-config-I7428 defect class:\n"
        + "\n".join(violations)
    )


class _RaisingSession:
    def get_config_variable(self, _name: str) -> str | None:
        raise RuntimeError("no config file on this box")


def test_resolve_region_prefers_aws_region_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "ap-south-1")
    assert aws_region.resolve_region() == "eu-west-1"


def test_resolve_region_falls_back_to_aws_default_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.setenv("AWS_DEFAULT_REGION", "ap-south-1")
    assert aws_region.resolve_region() == "ap-south-1"


def test_resolve_region_falls_back_to_botocore_session_when_env_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reproduces the exact SSM AWS-RunShellScript environment: neither
    AWS_REGION nor AWS_DEFAULT_REGION set."""
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    monkeypatch.setattr(aws_region, "_from_botocore_session", lambda: "us-west-2")
    assert aws_region.resolve_region() == "us-west-2"


def test_resolve_region_falls_back_to_imds_when_session_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    monkeypatch.setattr(aws_region, "_from_botocore_session", lambda: None)
    monkeypatch.setattr(aws_region, "_from_imds", lambda: "us-east-2")
    assert aws_region.resolve_region() == "us-east-2"


def test_resolve_region_never_returns_none_on_a_bare_box(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reproduction of alpha-engine-config-I7428's live failure: no env,
    no profile, no IMDS (off-EC2 / unreachable) — the exact shape that made
    `boto3.client("cloudwatch")` raise NoRegionError on every SSM box run.
    Post-fix this resolves to DEFAULT_REGION rather than propagating an
    unresolvable state."""
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    monkeypatch.setattr(aws_region, "_from_botocore_session", lambda: None)
    monkeypatch.setattr(aws_region, "_from_imds", lambda: None)
    region = aws_region.resolve_region()
    assert region == aws_region.DEFAULT_REGION
    assert region  # never empty


def test_from_botocore_session_swallows_a_broken_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import botocore.session

    monkeypatch.setattr(
        botocore.session, "get_session", lambda: _RaisingSession()
    )
    assert aws_region._from_botocore_session() is None


def test_from_imds_swallows_an_unreachable_metadata_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from botocore.utils import InstanceMetadataRegionFetcher

    def _raise(self: Any) -> str:
        raise RuntimeError("connection refused — not on EC2")

    monkeypatch.setattr(
        InstanceMetadataRegionFetcher, "retrieve_region", _raise
    )
    assert aws_region._from_imds() is None


def test_boto3_client_built_with_resolve_region_never_raises_noregionerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: a real (network-free) boto3 client construction, in the
    exact env shape SSM AWS-RunShellScript hands the launcher, must not raise
    NoRegionError once built through `resolve_region()`."""
    import boto3

    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    monkeypatch.setattr(aws_region, "_from_botocore_session", lambda: None)
    monkeypatch.setattr(aws_region, "_from_imds", lambda: None)

    client = boto3.client("cloudwatch", region_name=aws_region.resolve_region())
    assert client.meta.region_name == aws_region.DEFAULT_REGION
