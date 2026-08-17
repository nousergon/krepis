"""The ONE region resolver — every boto3 client in krepis builds through it.

**Why this exists.** `alpha-engine-config-I7428`: SSM ``AWS-RunShellScript``
runs with no ``AWS_REGION`` / ``AWS_DEFAULT_REGION`` in the environment and no
per-user boto profile on the box. A bare ``boto3.client("cloudwatch")`` there
raises ``botocore.exceptions.NoRegionError`` — but S3 (which can resolve
without a region for most operations) does not, so the failure read as an
intermittent, service-specific fault rather than a missing environment. Two
of ``krepis.stage_coverage``'s per-stage output assertions degraded to
``UNMEASURED`` on **every** box run as a result — the exact "detector that
never fires" shape `principles.md` names.

**Why one resolver.** Per `policy-shared-code`, the region-from-env fallback
chain existed independently in three places before this fix
(``fleet_events._region``, two inline copies in ``alerts.py``, two inline
``os.environ.get("AWS_REGION", "us-east-1")`` in ``router.py``) — past the
second-adoption trigger — and a fourth call site (``stage_coverage.py``)
carried no fallback at all, which is the box that actually broke. A forked
resolver is exactly the shape that lets one copy get the fix and the others
keep the bug.

**Resolution order** — each step only runs if the previous one came up empty:

1. ``AWS_REGION``
2. ``AWS_DEFAULT_REGION``
3. The botocore session's own config resolution (picks up ``~/.aws/config``
   ``region =`` under the active profile, when one exists).
4. EC2 Instance Metadata Service (IMDS). This is the one source that is
   correct **by construction** on a box, rather than correct because the
   fleet happens to be single-region today — every alpha-engine EC2 instance
   answers IMDS with the region it actually runs in, unaffected by which
   fleet remains single-region tomorrow. Bounded to a single 1s-timeout
   attempt: IMDS being unreachable (not on EC2 at all — a laptop, CI) must
   not stall a resolution that is about to fall through to a default anyway.
5. :data:`DEFAULT_REGION` (``us-east-1``) — the fleet's one region today.

This chain cannot return ``None`` or ``""``. That is deliberate: it retires
"no region resolvable" as an ``UNMEASURED`` condition. A caller that still
sees ``NoRegionError`` after building its client through
:func:`resolve_region` is exercising a bug in this resolver, not an
unmeasurable environment — that is a defect to fix here, not a fact to
observe-mode around at the call site.
"""

from __future__ import annotations

import logging
import os
from typing import Final

logger = logging.getLogger(__name__)

#: The fleet's one region today. Not a guess: every alpha-engine resource
#: (S3 buckets, Step Functions, Lambda, EC2) is provisioned in this region;
#: see ``infrastructure-ownership-policy.md``.
DEFAULT_REGION: Final[str] = "us-east-1"

#: IMDS is a single link-local hop; a working fetch resolves in well under
#: this. Kept short so a non-EC2 caller (laptop, CI, a spot box mid-teardown)
#: falls through to :data:`DEFAULT_REGION` quickly rather than stalling.
_IMDS_TIMEOUT_SECONDS: Final[float] = 1.0


def _from_botocore_session() -> str | None:
    try:
        import botocore.session
    except ImportError:  # pragma: no cover - botocore ships with boto3
        return None
    try:
        return botocore.session.get_session().get_config_variable("region")
    except Exception:  # noqa: BLE001 - config resolution is best-effort
        logger.debug("aws_region: botocore session region lookup failed", exc_info=True)
        return None


def _from_imds() -> str | None:
    try:
        from botocore.utils import InstanceMetadataRegionFetcher
    except ImportError:  # pragma: no cover - botocore ships with boto3
        return None
    try:
        region = InstanceMetadataRegionFetcher(
            timeout=_IMDS_TIMEOUT_SECONDS, num_attempts=1
        ).retrieve_region()
        return region or None
    except Exception:  # noqa: BLE001 - IMDS unreachable off-EC2 is routine
        logger.debug("aws_region: IMDS region lookup failed", exc_info=True)
        return None


def resolve_region() -> str:
    """Return a region string. Never ``None``, never empty, never raises.

    Order: ``AWS_REGION`` -> ``AWS_DEFAULT_REGION`` -> botocore session
    config -> IMDS -> :data:`DEFAULT_REGION`. Every ``boto3.client(...)`` call
    for a regional service anywhere in ``krepis`` must pass
    ``region_name=resolve_region()`` — a bare ``boto3.client("<service>")``
    for a regional service is the exact defect `alpha-engine-config-I7428`
    fixed; ``tests/test_aws_region.py::test_no_bare_boto3_client_for_regional_service``
    guards the regression.
    """
    env_region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if env_region:
        return env_region

    session_region = _from_botocore_session()
    if session_region:
        return session_region

    imds_region = _from_imds()
    if imds_region:
        return imds_region

    return DEFAULT_REGION
