"""
EC2 spot-launch capacity-resilience chokepoint.

Consolidation substrate for the spot-launch pattern that previously
appeared as three mirrored copies of the same fragility across the
alpha-engine fleet — each repo's launcher script (``spot_data_weekly.sh``
in alpha-engine-data, ``spot_train.sh`` in alpha-engine-predictor,
``spot_backtest.sh`` in alpha-engine-backtester) independently encoded
the same hardcoded ``--subnet-id`` (single AZ, us-east-1f) +
``--instance-type c5.large`` (single SKU) + N retries-with-backoff. When
AWS ran out of c5.large capacity in us-east-1f, every spot-launching
state failed simultaneously with no resilience.

**Why now (2026-05-22 evening):** The post-trap-fix dry-pass of the
Saturday SF (``postfix-keystone-20260522T232655Z``) hit
``InsufficientInstanceCapacity`` on the Evaluator's spot launch in
us-east-1f. The 2 earlier spots (Backtester + Parity) happened to clear
because AWS capacity rolled between the launches; Evaluator drew the
short straw. The defect class is "any single Saturday SF run has a
non-trivial chance of hitting capacity in at least one of the 3+ spot
states." The Friday-PM dry-pass exposed it (third in a row caught break
of the dry-pass safety net — first was the trap escape, second was the
keystone merge order, third is this).

**Why a CLI, not a bash function:**

Per ``~/Development/CLAUDE.md`` SOTA sub-sub-rule — "when mirroring a
pattern across repos, consider lifting it into ``nousergon-lib``...
Pure-Bash primitives can stay mirrored unless re-expressible as a Python
CLI entry callable from Bash, in which case the CLI re-expression is
the institutional path." Third repo with the same fragility is well
past the second-recurrence trigger. The CLI shape mirrors
:mod:`krepis.alerts` + :mod:`krepis.ssm_log_capture`
precedent.

**Strategy:**

The function iterates ``(instance_type, subnet)`` combinations in the
order given, attempting :func:`RunInstances` against each. On
``InsufficientInstanceCapacity`` / ``InsufficientHostCapacity`` /
``Unsupported`` (instance type not in AZ) → rotate to the next
combination. On any other error (auth, quota, AMI not found) → raise.

Caller controls the rotation order by listing types/subnets. Default
shape we use in the fleet:

- types: ``[c5.large, m5.large, c6i.large, c5a.large]`` (all 2 vCPU /
  ~4-8 GB RAM; capacity-resilient set chosen 2026-05-22)
- subnets: all default-VPC subnets across us-east-1{a,b,c,d,e,f}

**Public API:**

- :func:`launch` — Python API returning ``InstanceId`` on success,
  raising :class:`SpotCapacityExhausted` if every combination hit a
  capacity error, or :class:`SpotLaunchError` on any other error.
- CLI: ``python -m krepis.ec2_spot launch --types ... --subnets ...``.
  Returns ``InstanceId`` on stdout. Exits non-zero on failure;
  capacity-exhaustion exits 64 (distinguishable from generic failure).
- :func:`classify_termination` — classify why a (spot) instance terminated:
  ``reclaim`` (AWS reclaimed → caller should relaunch on a fresh spot),
  ``other`` (real crash / OOM / timeout → do NOT blind-retry), or ``unknown``.
  CLI: ``python -m krepis.ec2_spot classify-termination
  --instance-id <id>`` prints ``classification<TAB>state<TAB>reason_code<TAB>
  transition_reason``. The fleet-wide chokepoint for the spot-reclaim
  classification that previously lived (buggy) in ``spot_backtest.sh`` and was
  absent from ``spot_train.sh`` / the data spot launchers.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Final, Sequence

logger = logging.getLogger(__name__)

# Error codes (RunInstances) that mean "this combination is out of
# capacity, try another." Anything else is a hard error.
CAPACITY_ERROR_CODES: Final[frozenset[str]] = frozenset(
    {
        "InsufficientInstanceCapacity",
        "InsufficientHostCapacity",
        "SpotMaxPriceTooLow",  # spot-specific; AWS returns this when
                                # the AZ's spot price exceeds bid
        "Unsupported",          # instance type not offered in AZ
        "InvalidAvailabilityZone",
    }
)

CAPACITY_EXIT_CODE: Final[int] = 64

# ── Spot-reclaim classification ──────────────────────────────────────────────
# A mid-run AWS spot reclaim surfaces to a dispatcher as a generic command
# failure with no traceback. The authoritative signal is the instance's
# ``StateReason.Code`` — AWS sets ``Server.SpotInstanceTermination`` (or
# ``Server.InsufficientInstanceCapacity``) when it reclaims. Earlier, each
# spot launcher tried to classify this from ``StateTransitionReason`` alone
# (which only shows the human ``"Service initiated (<ts>)"`` form, NEVER the
# code) and matched against ``Server.SpotInstanceTermination`` — a field/value
# mismatch that could never hit, so two real backtester reclaims on 2026-06-06
# hard-failed instead of relaunching. This chokepoint reads the RIGHT field.
SPOT_RECLAIM_REASON_CODES: Final[frozenset[str]] = frozenset(
    {"Server.SpotInstanceTermination", "Server.InsufficientInstanceCapacity"}
)
# The MOST authoritative reclaim signal is the Spot Instance Request's
# Status.Code (queried first below). These are the SIR status codes that mean
# AWS reclaimed/never-maintained the instance for capacity/price reasons — a
# strict superset of what spot_data_weekly.sh already classified on, so this
# chokepoint never regresses the best existing launcher.
SPOT_RECLAIM_SIR_STATUS_CODES: Final[frozenset[str]] = frozenset(
    {
        "instance-terminated-no-capacity",
        "instance-terminated-by-price",
        "instance-terminated-capacity-oversubscribed",
        "instance-stopped-no-capacity",
        "instance-stopped-by-price",
        "instance-stopped-capacity-oversubscribed",
        "marked-for-termination",
    }
)
# Belt-and-suspenders: a worker already in one of these states whose
# StateTransitionReason contains "Service initiated" was torn down by AWS out
# from under a still-running dispatcher. A genuine in-instance crash/OOM leaves
# the instance ``running`` until the dispatcher terminates it, so this can never
# mis-fire on a real bug.
_RECLAIM_TRANSITION_STATES: Final[frozenset[str]] = frozenset({"shutting-down", "terminated"})
_RECLAIM_TRANSITION_MARKER: Final[str] = "Service initiated"


class SpotLaunchError(Exception):
    """Non-capacity RunInstances failure (auth, quota, AMI not found, …)."""


class SpotCapacityExhausted(SpotLaunchError):
    """Every (instance_type, subnet) combination returned a capacity error."""


def _build_run_instances_kwargs(
    *,
    image_id: str,
    instance_type: str,
    key_name: str,
    security_group_ids: list[str],
    subnet_id: str,
    iam_instance_profile: str,
    spot: bool,
    volume_size_gb: int,
    volume_type: str,
    shutdown_behavior: str,
    tag_name: str | None,
) -> dict:
    kwargs: dict = {
        "ImageId": image_id,
        "InstanceType": instance_type,
        "KeyName": key_name,
        "SecurityGroupIds": security_group_ids,
        "SubnetId": subnet_id,
        "IamInstanceProfile": {"Name": iam_instance_profile},
        "MinCount": 1,
        "MaxCount": 1,
        "InstanceInitiatedShutdownBehavior": shutdown_behavior,
        "BlockDeviceMappings": [
            {
                "DeviceName": "/dev/xvda",
                "Ebs": {"VolumeSize": volume_size_gb, "VolumeType": volume_type},
            }
        ],
    }
    if spot:
        kwargs["InstanceMarketOptions"] = {
            "MarketType": "spot",
            "SpotOptions": {
                "SpotInstanceType": "one-time",
                "InstanceInterruptionBehavior": "terminate",
            },
        }
    if tag_name:
        kwargs["TagSpecifications"] = [
            {
                "ResourceType": "instance",
                "Tags": [{"Key": "Name", "Value": tag_name}],
            }
        ]
    return kwargs


def launch(
    instance_types: Sequence[str],
    subnets: Sequence[str],
    *,
    image_id: str,
    key_name: str,
    security_group_ids: Sequence[str],
    iam_instance_profile: str,
    spot: bool = True,
    volume_size_gb: int = 30,
    volume_type: str = "gp3",
    shutdown_behavior: str = "terminate",
    tag_name: str | None = None,
    region: str = "us-east-1",
) -> str:
    """Launch a spot, rotating across instance_types × subnets on capacity error.

    Returns:
        Instance ID of the first successful launch.

    Raises:
        SpotCapacityExhausted: every (type, subnet) combination returned
            a capacity error. Caller can wait + retry, or escalate.
        SpotLaunchError: any other RunInstances error (auth, quota,
            AMI not found, …) — these don't retry, they raise loud.
        ValueError: empty instance_types or subnets list.
    """
    if not instance_types:
        raise ValueError("instance_types must be non-empty")
    if not subnets:
        raise ValueError("subnets must be non-empty")

    import boto3
    from botocore.exceptions import ClientError

    ec2 = boto3.client("ec2", region_name=region)
    sg_ids = list(security_group_ids)

    capacity_attempts: list[str] = []
    for instance_type in instance_types:
        for subnet_id in subnets:
            kwargs = _build_run_instances_kwargs(
                image_id=image_id,
                instance_type=instance_type,
                key_name=key_name,
                security_group_ids=sg_ids,
                subnet_id=subnet_id,
                iam_instance_profile=iam_instance_profile,
                spot=spot,
                volume_size_gb=volume_size_gb,
                volume_type=volume_type,
                shutdown_behavior=shutdown_behavior,
                tag_name=tag_name,
            )
            try:
                resp = ec2.run_instances(**kwargs)
            except ClientError as exc:
                err = exc.response.get("Error", {})
                code = err.get("Code", "UnknownError")
                msg = err.get("Message", str(exc))
                if code in CAPACITY_ERROR_CODES:
                    capacity_attempts.append(f"{instance_type}@{subnet_id}: {code}")
                    logger.warning(
                        "ec2_spot: %s in %s for %s — rotating",
                        code,
                        subnet_id,
                        instance_type,
                    )
                    print(
                        f"ec2_spot: {code} for {instance_type}@{subnet_id} — rotating",
                        file=sys.stderr,
                    )
                    continue
                raise SpotLaunchError(
                    f"RunInstances failed with non-capacity error "
                    f"{code} ({instance_type}@{subnet_id}): {msg}"
                ) from exc

            instance_id = resp["Instances"][0]["InstanceId"]
            logger.info(
                "ec2_spot: launched %s as %s in %s",
                instance_type,
                instance_id,
                subnet_id,
            )
            print(
                f"ec2_spot: launched {instance_type} as {instance_id} in {subnet_id}",
                file=sys.stderr,
            )
            return instance_id

    raise SpotCapacityExhausted(
        f"every (instance_type, subnet) combination returned a capacity error "
        f"({len(capacity_attempts)} attempts): "
        + "; ".join(capacity_attempts)
    )


def classify_termination(instance_id: str, *, region: str = "us-east-1") -> dict[str, str]:
    """Classify why a (spot) instance is terminating/terminated.

    Returns a dict with keys ``classification`` (``"reclaim"`` | ``"other"`` |
    ``"unknown"``), ``state``, ``reason_code``, ``transition_reason``.

    ``"reclaim"`` means AWS reclaimed the spot — the caller should relaunch on a
    fresh spot rather than treat the failure as terminal. ``"other"`` is any
    other terminal cause (real crash / OOM / delivery timeout / user shutdown):
    the caller must NOT blind-retry it. ``"unknown"`` if the instance cannot be
    described (it may already be gone).

    Classification is reclaim iff ANY of (in authority order):

    1. the Spot Instance Request's ``Status.Code`` is one of
       :data:`SPOT_RECLAIM_SIR_STATUS_CODES` (the most authoritative signal —
       the ``sir_code`` field in the result), OR
    2. the instance's ``StateReason.Code`` is one of
       :data:`SPOT_RECLAIM_REASON_CODES` (the ``reason_code`` field), OR
    3. the instance is shutting-down/terminated with a "Service initiated"
       ``StateTransitionReason`` (see module notes — the field-mismatch fix).
    """
    import boto3
    from botocore.exceptions import ClientError

    ec2 = boto3.client("ec2", region_name=region)
    result = {
        "classification": "unknown",
        "state": "",
        "reason_code": "",
        "transition_reason": "",
        "sir_code": "",
    }

    # 1. Spot Instance Request Status.Code — the authoritative reclaim signal,
    #    queryable even after the instance is gone. Best-effort: on-demand
    #    instances have no SIR (empty), and a describe failure just falls
    #    through to the instance-level checks.
    try:
        sir = ec2.describe_spot_instance_requests(
            Filters=[{"Name": "instance-id", "Values": [instance_id]}]
        )
        reqs = sir.get("SpotInstanceRequests") or []
        if reqs:
            result["sir_code"] = (reqs[0].get("Status") or {}).get("Code", "") or ""
    except ClientError as exc:
        logger.warning(
            "ec2_spot: describe-spot-instance-requests failed for %s: %s",
            instance_id,
            exc,
        )

    # 2/3. Instance State + StateReason.Code + StateTransitionReason.
    described = False
    try:
        resp = ec2.describe_instances(InstanceIds=[instance_id])
        reservations = resp.get("Reservations") or []
        instances = reservations[0].get("Instances") if reservations else None
        if instances:
            described = True
            inst = instances[0]
            result["state"] = (inst.get("State") or {}).get("Name", "") or ""
            result["reason_code"] = (inst.get("StateReason") or {}).get("Code", "") or ""
            result["transition_reason"] = inst.get("StateTransitionReason", "") or ""
    except ClientError as exc:
        logger.warning(
            "ec2_spot: describe-instances failed for %s: %s", instance_id, exc
        )

    # If neither the SIR nor the instance could be read, we genuinely don't know.
    if not result["sir_code"] and not described:
        return result  # classification stays "unknown"

    is_reclaim = (
        result["sir_code"] in SPOT_RECLAIM_SIR_STATUS_CODES
        or any(c in result["reason_code"] for c in SPOT_RECLAIM_REASON_CODES)
        or (
            result["state"] in _RECLAIM_TRANSITION_STATES
            and _RECLAIM_TRANSITION_MARKER in result["transition_reason"]
        )
    )
    result["classification"] = "reclaim" if is_reclaim else "other"
    return result


def _split_csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m krepis.ec2_spot",
        description=(
            "Launch an EC2 spot with capacity-resilient rotation across "
            "instance types and subnets. The institutional replacement for "
            "the hardcoded single-subnet + single-instance-type pattern "
            "mirrored across the alpha-engine fleet's spot launchers."
        ),
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    launch_p = subparsers.add_parser(
        "launch",
        help="Launch a spot with rotating (type, subnet) combinations.",
    )
    launch_p.add_argument(
        "--types",
        required=True,
        help=(
            "Comma-separated instance types to try in order "
            "(e.g., 'c5.large,m5.large,c6i.large'). First success wins."
        ),
    )
    launch_p.add_argument(
        "--subnets",
        required=True,
        help=(
            "Comma-separated subnet IDs to try in order. Each is an AZ "
            "(default-VPC subnets in us-east-1 span 1a-1f)."
        ),
    )
    launch_p.add_argument("--image-id", required=True, help="AMI ID.")
    launch_p.add_argument("--key-name", required=True, help="EC2 key pair name.")
    launch_p.add_argument(
        "--security-group",
        required=True,
        action="append",
        help=(
            "Security group ID. Pass multiple times for >1 SG: "
            "--security-group sg-A --security-group sg-B"
        ),
    )
    launch_p.add_argument(
        "--iam-profile",
        required=True,
        help="IAM instance profile NAME (not ARN).",
    )
    launch_p.add_argument(
        "--no-spot",
        action="store_true",
        help="Launch on-demand instead of spot.",
    )
    launch_p.add_argument(
        "--volume-size",
        type=int,
        default=30,
        help="Root EBS volume size in GB (default: 30).",
    )
    launch_p.add_argument(
        "--volume-type",
        default="gp3",
        help="Root EBS volume type (default: gp3).",
    )
    launch_p.add_argument(
        "--shutdown-behavior",
        default="terminate",
        choices=("terminate", "stop"),
        help="Instance-initiated shutdown behavior (default: terminate).",
    )
    launch_p.add_argument(
        "--name",
        default=None,
        help="Name tag applied to the launched instance.",
    )
    launch_p.add_argument(
        "--region",
        default=os.environ.get("AWS_REGION", "us-east-1"),
        help="AWS region (default: $AWS_REGION or us-east-1).",
    )

    classify_p = subparsers.add_parser(
        "classify-termination",
        help=(
            "Classify why a (spot) instance terminated: reclaim | other | "
            "unknown. Prints TAB-separated 'classification<TAB>state<TAB>"
            "reason_code<TAB>transition_reason' on stdout for bash callers."
        ),
    )
    classify_p.add_argument("--instance-id", required=True, help="EC2 instance ID.")
    classify_p.add_argument(
        "--region",
        default=os.environ.get("AWS_REGION", "us-east-1"),
        help="AWS region (default: $AWS_REGION or us-east-1).",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING)

    if args.cmd == "classify-termination":
        result = classify_termination(args.instance_id, region=args.region)
        # TAB-separated, fixed field order — bash:
        #   IFS=$'\t' read -r verdict state rcode treason sir < <(python -m ... )
        print(
            "\t".join(
                (
                    result["classification"],
                    result["state"],
                    result["reason_code"],
                    result["transition_reason"],
                    result["sir_code"],
                )
            )
        )
        return 0

    try:
        instance_id = launch(
            instance_types=_split_csv(args.types),
            subnets=_split_csv(args.subnets),
            image_id=args.image_id,
            key_name=args.key_name,
            security_group_ids=args.security_group,
            iam_instance_profile=args.iam_profile,
            spot=not args.no_spot,
            volume_size_gb=args.volume_size,
            volume_type=args.volume_type,
            shutdown_behavior=args.shutdown_behavior,
            tag_name=args.name,
            region=args.region,
        )
    except SpotCapacityExhausted as exc:
        print(f"ec2_spot: capacity exhausted: {exc}", file=sys.stderr)
        return CAPACITY_EXIT_CODE
    except SpotLaunchError as exc:
        print(f"ec2_spot: launch failed: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"ec2_spot: bad input: {exc}", file=sys.stderr)
        return 2

    # InstanceId on stdout — bash callers capture this via $(...)
    print(instance_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
