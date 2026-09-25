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
combination. On an account-wide spot quota error (config#2698 —
``MaxSpotInstanceCountExceeded`` / ``SpotInstanceRequestLimitExceeded`` /
``MaxSpotFleetRequestCountExceeded``) → raise :class:`SpotQuotaExceededError`
immediately, skipping rotation (every remaining combination draws on the
same account-wide quota and would fail identically). On any other error
(auth, AMI not found) → raise :class:`SpotLaunchError`.

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
- :func:`relaunch_decision` / ``relaunch-decision`` CLI — the bounded mid-run
  relaunch DECISION wrapped around classification: given the attempt number, a
  ``MAX_SPOT_ATTEMPTS`` budget, and (optionally) the outer SF
  ``executionTimeout`` coupling guard, decide whether the launcher should exec a
  fresh spot. Lifts the divergent inline copies in ``spot_data_weekly.sh`` (#349)
  and ``spot_backtest.sh`` (#283/#289) and supplies the missing predictor
  adopter. CLI exits 0 = relaunch, 75 = hold (fail loud).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from dataclasses import dataclass
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

# Error codes that mean "spot is unavailable ACCOUNT-WIDE" (a quota, not a
# per-(type, subnet) capacity shortfall). Deliberately kept OUT of
# CAPACITY_ERROR_CODES (config#2698): capacity semantics are "try the next
# combination" — rotating never clears a quota error, every remaining
# attempt in the loop would fail identically. Quota semantics are "spot is
# closed account-wide, go on-demand" — the caller falls back exactly as it
# already does for full capacity exhaustion (see
# krepis.SpotQuotaExceededError / nousergon_lib.spot_dispatch.launch_with_fallback).
QUOTA_ERROR_CODES: Final[frozenset[str]] = frozenset(
    {
        "MaxSpotInstanceCountExceeded",
        "SpotInstanceRequestLimitExceeded",
        "MaxSpotFleetRequestCountExceeded",
    }
)

CAPACITY_EXIT_CODE: Final[int] = 64

# CLI ``launch`` exit code for a quota error — distinct from CAPACITY_EXIT_CODE
# (64) so a bash caller can tell "rotate/wait" (capacity) apart from "spot is
# closed account-wide" (quota) instead of both collapsing into the same code.
QUOTA_EXIT_CODE: Final[int] = 65

# relaunch-decision CLI, LEGACY exit-code contract: exit 0 = RELAUNCH, this =
# HOLD (do not relaunch, fail loud). Distinct from 64 (capacity) and 1 (generic)
# so a bash caller can branch on it unambiguously: `python -m ...
# relaunch-decision ...; case $? in 0) exec ...;; 75) exit "$orig";; esac`.
#
# DEPRECATED AS THE DEFAULT ANSWER SHAPE — prefer `--json` (see below).
#
# Why: "hold" is not an error. It is the ordinary, expected verdict for every
# failure that is not an AWS reclaim, i.e. the large majority of calls. Encoding
# it as a NON-ZERO EXIT put the API's normal answer into the one channel that
# `set -e` treats as fatal, and every bash adopter wrote the natural
# `VAR="$(cmd)"` — a simple command whose status IS the substitution's. Errexit
# then fired ON THE NORMAL ANSWER, inside the EXIT trap that was calling it, so
# `terminate-instances` was never reached and the spot instance leaked. The
# abort was silent because `set -e` does not re-enter a trap it is running.
#
# Measured 2026-08-12 across the fleet: SIX call sites in FIVE repos, all
# written the same way, none of them guarded. Three were live leaks
# (crucible-predictor, crucible-backtester ×2 — the second reached by ten
# per-stage launchers — and crucible-dashboard); two more (nousergon-data ×2,
# crucible-research) escaped only because their single caller happened to write
# `|| reason=""`, which suppresses errexit through the whole call. Six of six
# adopters made the same mistake, which makes it an API defect, not six user
# errors: an interface whose normal answer is indistinguishable from a failure
# will be misused by every correct-looking caller.
#
# The exit code is retained verbatim for the callers already handling it — a
# change here would break them silently — but new callers should ask for
# `--json`, where the verdict is a FIELD and the exit status means only whether
# the CLI could answer at all.
NO_RELAUNCH_EXIT_CODE: Final[int] = 75

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


class SpotQuotaExceededError(SpotLaunchError):
    """RunInstances returned an account-wide spot quota error (one of
    :data:`QUOTA_ERROR_CODES`). Raised immediately on the FIRST quota error —
    unlike :class:`SpotCapacityExhausted`, rotation is skipped entirely
    because every remaining (type, subnet) combination draws on the same
    account-wide quota and would fail identically. A sibling of
    ``SpotCapacityExhausted`` (both are ``SpotLaunchError``, not each other)
    so callers can branch on the two distinct semantics: "try on-demand
    because every combo is out of capacity" vs. "try on-demand because spot
    itself is closed account-wide" — see ``nousergon_lib.spot_dispatch.
    launch_with_fallback``, which catches both and falls back to on-demand,
    paging only on the quota case (quota pressure needs an operator's eyes;
    ordinary capacity rotation exhaustion does not)."""


class IdempotencyConflict(SpotLaunchError):
    """RunInstances refused a ``ClientToken`` it had already seen with
    different parameters (``IdempotentParameterMismatch``). The token already
    launched an instance, so this is never grounds for launching another one:
    :func:`launch_self_starting` answers it by finding that instance."""


# ── Idempotent, self-starting launch (alpha-engine-config-I11597) ────────────
#
# The dispatcher pattern this replaces for a one-shot job was
# ``launch`` -> wait for SSM Online -> ``send-command``: two non-atomic steps
# inside a timeout-bounded Lambda. On 2026-09-23 (alpha-engine-config-I11532)
# SSM registered slowly, the Lambda was killed between the launch and the
# send, and the async retry found a running box that had never been given
# its job. The box idled; the day's run was lost.
#
# A self-starting box carries its own work: ONE RunInstances whose user-data
# installs and starts the job as a systemd unit at boot
# (``krepis.spot_bootstrap.render_self_starting_user_data``). There is no
# second step to lose. What remains is making the one step replay-safe, which
# is what EC2's ``ClientToken`` is for.

#: RunInstances error code for a reused ClientToken with different parameters.
IDEMPOTENT_MISMATCH_CODE: Final[str] = "IdempotentParameterMismatch"

#: EC2 caps a ClientToken at 64 ASCII characters.
CLIENT_TOKEN_MAX_LEN: Final[int] = 64

#: EC2's user-data ceiling: 16 KB of RAW data (the limit applies before the
#: base64 encoding boto3 adds). A job whose script would exceed it must have
#: its user-data FETCH the script (a git checkout or an S3 object) instead of
#: inlining it.
USER_DATA_MAX_BYTES: Final[int] = 16 * 1024

#: Launch provenance tags. The VALUES are a contract shared with
#: ``nousergon_lib.spot_dispatch`` (LAUNCH_MARKET_TAG / LAUNCH_REASON_TAG and
#: its REASON_* vocabulary, alpha-engine-config-I5727): consumers classify
#: launches by these strings, so a self-starting launch must be countable in
#: exactly the same terms as an SSM-dispatched one.
#: ``tests/test_ec2_spot_self_starting.py`` pins the literals.
LAUNCH_MARKET_TAG: Final[str] = "LaunchMarket"
LAUNCH_REASON_TAG: Final[str] = "LaunchReason"
REASON_SPOT_OK: Final[str] = "spot_ok"
REASON_CAPACITY: Final[str] = "capacity_exhausted"
REASON_QUOTA: Final[str] = "quota_exceeded"
REASON_FORCED: Final[str] = "force_on_demand"

#: Filter values per DescribeInstances call when probing by client token.
#: Conservative: a dispatcher rotating 9 types x 6 subnets x 2 markets has 108
#: tokens, and one oversized filter must not become a probe failure.
_PROBE_CHUNK = 50


def _check_user_data(user_data: str | None) -> None:
    if user_data is None:
        return
    if not user_data.strip():
        raise ValueError("user_data must be non-empty when given")
    size = len(user_data.encode("utf-8"))
    if size > USER_DATA_MAX_BYTES:
        raise ValueError(
            f"user_data is {size} bytes; EC2 accepts at most "
            f"{USER_DATA_MAX_BYTES}. Have the user-data fetch the job script "
            "(from the box's git checkout or S3) instead of inlining it."
        )


def client_token(
    idempotency_key: str, *, spot: bool, instance_type: str, subnet_id: str
) -> str:
    """The RunInstances ``ClientToken`` for one launch attempt.

    Deterministic in all four inputs. The attempt's parameters are part of it
    because EC2 rejects a token reused with DIFFERENT parameters
    (``IdempotentParameterMismatch``): one token per (market, type, subnet)
    lets rotation proceed normally, while a replay that reaches the same
    attempt gets the same instance back instead of launching a second one.
    """
    if not idempotency_key:
        raise ValueError("idempotency_key must be non-empty")
    market = "spot" if spot else "on-demand"
    digest = hashlib.sha256(
        f"{idempotency_key}|{market}|{instance_type}|{subnet_id}".encode("utf-8")
    ).hexdigest()
    # 7 + 48 = 55 characters: inside CLIENT_TOKEN_MAX_LEN, pure ASCII.
    return f"krepis-{digest[:48]}"


def _all_tokens(
    idempotency_key: str, instance_types: Sequence[str], subnets: Sequence[str]
) -> dict[str, str]:
    """``{token: market}`` for every attempt a self-starting launch can make."""
    return {
        client_token(
            idempotency_key, spot=spot, instance_type=itype, subnet_id=subnet
        ): ("spot" if spot else "on-demand")
        for spot in (True, False)
        for itype in instance_types
        for subnet in subnets
    }


def find_by_idempotency_key(
    idempotency_key: str,
    instance_types: Sequence[str],
    subnets: Sequence[str],
    *,
    region: str = "us-east-1",
) -> tuple[str, str] | None:
    """``(instance_id, market)`` of the instance an earlier launch with this
    key created, in ANY state, or ``None`` if the probe found none.

    Needed on top of the per-attempt ``ClientToken``: a replay walks the same
    rotation, but capacity may have come back in a pool the first call was
    refused, so the replay's first successful attempt can carry a different
    token than the original's. Every token the key could have produced is
    looked up by DescribeInstances' ``client-token`` filter.

    A terminated instance counts: the key already had its launch, and whether
    that box finished is for the completion marker and the reaper to judge,
    not grounds for a second launch.

    Raises whatever DescribeInstances raised. A failed probe is not "no
    instance" — :func:`launch_self_starting` records it and relies on the
    per-attempt tokens alone.
    """
    import boto3

    tokens = _all_tokens(idempotency_key, instance_types, subnets)
    ec2 = boto3.client("ec2", region_name=region)
    ordered = sorted(tokens)
    for start in range(0, len(ordered), _PROBE_CHUNK):
        chunk = ordered[start : start + _PROBE_CHUNK]
        next_token: str | None = None
        while True:
            call: dict = {"Filters": [{"Name": "client-token", "Values": chunk}]}
            if next_token:
                call["NextToken"] = next_token
            resp = ec2.describe_instances(**call)
            for reservation in resp.get("Reservations", []):
                for inst in reservation.get("Instances", []):
                    iid = inst.get("InstanceId")
                    if not iid:
                        continue
                    market = tokens.get(inst.get("ClientToken", ""))
                    if market is None:
                        tags = {
                            t.get("Key"): t.get("Value")
                            for t in inst.get("Tags", []) or []
                        }
                        market = tags.get(LAUNCH_MARKET_TAG, "unknown")
                    return iid, market
            next_token = resp.get("NextToken")
            if not next_token:
                break
    return None


@dataclass(frozen=True)
class SelfStartingLaunch:
    """What :func:`launch_self_starting` did.

    ``replayed`` is True when the key had ALREADY launched this instance — the
    caller is a retry of an invocation that got as far as RunInstances.
    ``probe_degraded`` is True when the lookup by key failed and the launch
    relied on the per-attempt ClientTokens alone (recorded, never silent).
    """

    instance_id: str
    market: str
    replayed: bool
    probe_degraded: bool


def launch_self_starting(
    instance_types: Sequence[str],
    subnets: Sequence[str],
    *,
    idempotency_key: str,
    user_data: str,
    image_id: str,
    key_name: str,
    security_group_ids: Sequence[str],
    iam_instance_profile: str,
    tag_name: str,
    volume_size_gb: int = 30,
    extra_tags: dict[str, str] | None = None,
    force_on_demand: bool = False,
    region: str = "us-east-1",
) -> SelfStartingLaunch:
    """Launch a box that runs its own job from user-data, replay-safely.

    One call does the whole dispatch: there is no SSM wait and no command to
    send, so there is no half-done state for a killed caller to leave behind.
    Built for a Lambda dispatcher passing its invocation's request id as
    ``idempotency_key``: Lambda's async retries of one event carry that
    event's request id, so a retry gets back the box its killed predecessor
    launched instead of launching a second one.

    Order of operations:

    1. :func:`find_by_idempotency_key` — if this key already launched a box,
       return it (``replayed=True``) and launch nothing.
    2. Spot, rotating types x subnets (:func:`launch`), each attempt with its
       own ClientToken.
    3. On capacity exhaustion or an account-wide spot quota, on-demand across
       the same rotation — the fallback ``nousergon_lib.spot_dispatch.
       launch_with_fallback`` applies, with the same ``LaunchMarket`` /
       ``LaunchReason`` provenance tags and the same operator page on a quota
       ceiling. ``force_on_demand`` skips straight to this step.
    4. :class:`IdempotencyConflict` from either step means a token already
       launched a box with different parameters: that box is looked up and
       returned (``replayed=True``); if it cannot be found, the conflict
       raises rather than launching again.

    ``user_data`` is normally ``krepis.spot_bootstrap.
    render_self_starting_user_data(...)``. It is readable through
    ``DescribeInstanceAttribute``, so it must carry references, never secrets.

    Raises :class:`SpotLaunchError` (or a subclass) when spot and on-demand are
    both exhausted, or on any non-capacity RunInstances error.
    """
    if not idempotency_key:
        raise ValueError("idempotency_key must be non-empty")
    if not tag_name:
        raise ValueError("tag_name must be non-empty")
    if not user_data:
        raise ValueError("user_data must be non-empty")
    _check_user_data(user_data)

    probe_degraded = False
    try:
        found = find_by_idempotency_key(
            idempotency_key, instance_types, subnets, region=region
        )
    except Exception as exc:  # noqa: BLE001 - recorded on the result + logged; tokens still hold
        # The per-attempt ClientTokens remain in force, so a replay reaching
        # the same attempt still gets the same box back; what is lost is only
        # the cross-attempt case. Recorded on the result the caller returns.
        logger.error(
            "ec2_spot: idempotency probe for %s failed (%s: %s) — launching "
            "with per-attempt ClientTokens only",
            tag_name,
            type(exc).__name__,
            exc,
        )
        probe_degraded = True
        found = None
    if found is not None:
        iid, market = found
        logger.warning(
            "ec2_spot: idempotency key already launched %s (%s) for %s — "
            "returning it, launching nothing",
            iid,
            market,
            tag_name,
        )
        return SelfStartingLaunch(iid, market, True, probe_degraded)

    def attempt(spot: bool, reason: str) -> str:
        tags = dict(extra_tags or {})
        # Library keys win on collision: measured facts about the launch.
        tags[LAUNCH_MARKET_TAG] = "spot" if spot else "on-demand"
        tags[LAUNCH_REASON_TAG] = reason
        return launch(
            list(instance_types),
            list(subnets),
            image_id=image_id,
            key_name=key_name,
            security_group_ids=list(security_group_ids),
            iam_instance_profile=iam_instance_profile,
            spot=spot,
            volume_size_gb=volume_size_gb,
            shutdown_behavior="terminate",
            tag_name=tag_name,
            extra_tags=tags,
            region=region,
            user_data=user_data,
            idempotency_key=idempotency_key,
        )

    try:
        if force_on_demand:
            logger.warning("ec2_spot: force_on_demand set — launching ON-DEMAND directly")
            return SelfStartingLaunch(
                attempt(False, REASON_FORCED), "on-demand", False, probe_degraded
            )
        try:
            return SelfStartingLaunch(
                attempt(True, REASON_SPOT_OK), "spot", False, probe_degraded
            )
        except SpotCapacityExhausted:
            logger.warning(
                "ec2_spot: spot capacity exhausted for %s — relaunching ON-DEMAND",
                tag_name,
            )
            return SelfStartingLaunch(
                attempt(False, REASON_CAPACITY), "on-demand", False, probe_degraded
            )
        except SpotQuotaExceededError as exc:
            logger.warning(
                "ec2_spot: spot quota exceeded (%s) — relaunching ON-DEMAND", exc
            )
            _page_quota(tag_name, region, exc)
            return SelfStartingLaunch(
                attempt(False, REASON_QUOTA), "on-demand", False, probe_degraded
            )
    except IdempotencyConflict:
        found = find_by_idempotency_key(
            idempotency_key, instance_types, subnets, region=region
        )
        if found is None:
            raise
        iid, market = found
        logger.warning(
            "ec2_spot: ClientToken conflict resolved to the key's existing "
            "instance %s (%s) — launching nothing",
            iid,
            market,
        )
        return SelfStartingLaunch(iid, market, True, probe_degraded)


def _page_quota(tag_name: str, region: str, exc: Exception) -> None:
    """Same operator page ``nousergon_lib.spot_dispatch`` sends: a quota
    ceiling only clears when a human requests an increase."""
    from krepis import alerts

    alerts.publish(
        f"EC2 spot quota exceeded for {tag_name!r} in {region} — "
        f"falling back to on-demand: {exc}",
        severity="warning",
        source="krepis.ec2_spot.launch_self_starting",
        dedup_key=f"spot-quota-exceeded-{region}",
    )


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
    extra_tags: dict[str, str] | None = None,
    user_data: str | None = None,
    client_token: str | None = None,
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
    # Merge Name + extra_tags into ONE TagSpecifications entry so every
    # discriminator tag rides the same RunInstances call atomically (root fix
    # for alpha-engine-config#2292 / config#2267 site 2: a POST-LAUNCH
    # create_tags call leaves a seconds-wide window where the box exists
    # untagged — invisible to a dedupe guard, undiscoverable by the
    # orphan-reaper's completion-marker key derivation). extra_tags wins on a
    # key collision with Name (an explicit caller-supplied tag is more
    # specific than the generic Name convenience param) — last-Key-wins is
    # also how AWS itself would apply a duplicate-key Tags list.
    tags = []
    if tag_name:
        tags.append({"Key": "Name", "Value": tag_name})
    if extra_tags:
        tags.extend({"Key": k, "Value": v} for k, v in extra_tags.items())
    if tags:
        # Tag the EBS volume too (same RunInstances call, second
        # TagSpecifications entry) — the volume created via
        # BlockDeviceMappings is otherwise untagged and invisible to every
        # cost-allocation-tag CUR query forever (alpha-engine-config-I11273).
        # A post-launch create_tags call would reintroduce the untagged-box
        # race this function was built to close above.
        kwargs["TagSpecifications"] = [
            {
                "ResourceType": "instance",
                "Tags": tags,
            },
            {
                "ResourceType": "volume",
                "Tags": tags,
            },
        ]
    # Both keys are ABSENT unless asked for, so every existing caller's
    # RunInstances request is byte-for-byte what it was before they existed.
    if user_data is not None:
        # boto3 base64-encodes UserData for RunInstances itself; pass it raw.
        kwargs["UserData"] = user_data
    if client_token is not None:
        kwargs["ClientToken"] = client_token
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
    extra_tags: dict[str, str] | None = None,
    region: str = "us-east-1",
    user_data: str | None = None,
    idempotency_key: str | None = None,
) -> str:
    """Launch a spot, rotating across instance_types × subnets on capacity error.

    Args:
        extra_tags: additional ``{key: value}`` instance tags merged into the
            SAME ``TagSpecifications`` entry as the ``Name`` tag, so they ride
            the RunInstances call atomically — the box is never observably
            untagged. This is the root fix for the post-launch ``create_tags``
            race (alpha-engine-config#2292): a caller that previously wrote
            discriminator tags via a separate, post-launch ``create_tags``
            call (with its own bounded retry) should pass them here instead
            and delete that retry path entirely — one mechanism, not two.
        user_data: script the instance runs at first boot (cloud-init). At
            most :data:`USER_DATA_MAX_BYTES` raw bytes. Readable by anyone
            holding ``ec2:DescribeInstanceAttribute`` — never put a secret in
            it; pass references (SSM parameter names, S3 keys) instead.
        idempotency_key: makes the launch replay-safe (alpha-engine-config-
            I11597). Every RunInstances attempt carries a ``ClientToken``
            derived from this key AND the attempt's (market, type, subnet) —
            see :func:`client_token` — so a caller retried with the same key
            gets back the instance its earlier attempt created instead of a
            second box. Rotation never reuses a token across different
            parameters, which EC2 would reject as ``IdempotentParameterMismatch``.
            Callers that also rotate markets should use
            :func:`launch_self_starting`, which probes every token first.

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
    _check_user_data(user_data)

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
                extra_tags=extra_tags,
                user_data=user_data,
                client_token=(
                    client_token(
                        idempotency_key,
                        spot=spot,
                        instance_type=instance_type,
                        subnet_id=subnet_id,
                    )
                    if idempotency_key is not None
                    else None
                ),
            )
            try:
                resp = ec2.run_instances(**kwargs)
            except ClientError as exc:
                err = exc.response.get("Error", {})
                code = err.get("Code", "UnknownError")
                msg = err.get("Message", str(exc))
                if code in QUOTA_ERROR_CODES:
                    # Account-wide — every remaining (type, subnet)
                    # combination would fail identically, so rotation is
                    # skipped outright (config#2698).
                    logger.warning(
                        "ec2_spot: %s (account-wide spot quota) for %s@%s — "
                        "not rotating, caller should fall back to on-demand",
                        code,
                        instance_type,
                        subnet_id,
                    )
                    print(
                        f"ec2_spot: {code} (spot quota exceeded) for "
                        f"{instance_type}@{subnet_id} — not rotating",
                        file=sys.stderr,
                    )
                    raise SpotQuotaExceededError(
                        f"RunInstances failed with account-wide spot quota "
                        f"error {code} ({instance_type}@{subnet_id}): {msg}"
                    ) from exc
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
                if code == IDEMPOTENT_MISMATCH_CODE:
                    # This attempt's token already launched something, with
                    # different parameters (e.g. a tag value that moved between
                    # the original call and the retry). The box exists; it is
                    # the caller's to find, never a reason to launch another.
                    raise IdempotencyConflict(
                        f"RunInstances refused a reused ClientToken with "
                        f"different parameters ({instance_type}@{subnet_id}): "
                        f"{msg}"
                    ) from exc
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


# ── Bounded mid-run relaunch decision ────────────────────────────────────────
# classify_termination answers "was this a reclaim?". The relaunch *decision*
# that wraps it — "given the verdict and how many attempts we've already burned,
# should the launcher exec a fresh spot, and does the attempt budget still fit
# the outer Step-Functions executionTimeout?" — was duplicated, with divergent
# conventions, in every adopter:
#   * alpha-engine-data/spot_data_weekly.sh (#349): forward SPOT_ATTEMPT counter
#     vs MAX_SPOT_ATTEMPTS, relaunch iff reason non-empty AND attempts remain.
#   * alpha-engine-backtester/spot_backtest.sh (#283/#289): decrementing
#     RECLAIM_RELAUNCH_MAX budget, relaunch iff reclaim AND budget > 0.
#   * predictor spot_train.sh: NO relaunch at all (the open gap, issue #883).
# Two divergent in-repo copies + one missing adopter is exactly the ≥2-consumer
# lift-to-lib trigger. This chokepoint owns the decision so all three launchers
# collapse to: classify → ask the lib → exec-relaunch (the exec itself stays in
# bash; it must replace the launcher's own PID and cannot be lifted).
DEFAULT_MAX_SPOT_ATTEMPTS: Final[int] = 2  # one relaunch; matches #349 default

# Outer Step-Functions executionTimeout (seconds) per orchestrated state, used
# to guard the MAX_SPOT_ATTEMPTS ↔ SF-budget coupling the issue calls out: each
# relaunch costs a fresh boot (~7 min) plus a worst-case full re-run, so attempts
# beyond what the SF budget can absorb are dead budget that silently expire the
# state. A launcher passes ``--sf-execution-timeout`` (the SF budget it runs
# under) and a per-attempt wall-time estimate; we refuse to advise a relaunch the
# budget cannot fit. Known fleet budgets (from the Saturday SF definitions):
SF_EXECUTION_TIMEOUTS: Final[dict[str, int]] = {
    "DataPhase1": 5400,
    "MorningEnrich": 5400,
    "RAGIngestion": 3600,
}


def relaunch_decision(
    *,
    classification: str,
    attempt: int,
    max_attempts: int = DEFAULT_MAX_SPOT_ATTEMPTS,
    sf_execution_timeout: int | None = None,
    per_attempt_seconds: int | None = None,
) -> dict[str, object]:
    """Decide whether a launcher should relaunch on a fresh spot.

    Pure decision logic (no AWS calls) so it is trivially testable and the
    caller stays in control of the AWS describe (via :func:`classify_termination`)
    and the ``exec`` re-launch. Inputs:

    * ``classification`` — the verdict from :func:`classify_termination`
      (``"reclaim"`` | ``"other"`` | ``"unknown"``). Only ``"reclaim"`` is
      retryable; ``"other"`` (real crash/OOM/timeout) and ``"unknown"`` must
      fail loud so a blind retry never masks a genuine bug.
    * ``attempt`` — 1-based count of the attempt that just finished (the first
      run is attempt 1).
    * ``max_attempts`` — total attempts allowed including the first
      (default :data:`DEFAULT_MAX_SPOT_ATTEMPTS` = 2, i.e. one relaunch).
    * ``sf_execution_timeout`` / ``per_attempt_seconds`` — optional coupling
      guard. When BOTH are given, the next attempt is only advised if
      ``(attempt + 1) * per_attempt_seconds <= sf_execution_timeout`` — so
      raising ``max_attempts`` past what the outer SF budget can absorb can
      never silently produce dead attempts (the issue's explicit requirement).

    Returns a dict with:

    * ``relaunch`` (bool) — True iff the launcher should exec a fresh spot.
    * ``reason`` (str) — short machine-readable cause for logging/metrics.
    * ``attempts_remaining`` (int) — attempts left AFTER this one.
    * ``next_attempt`` (int) — the attempt number the relaunch would be.
    """
    attempts_remaining = max(0, max_attempts - attempt)
    next_attempt = attempt + 1

    if classification != "reclaim":
        return {
            "relaunch": False,
            "reason": f"not-reclaim:{classification or 'empty'}",
            "attempts_remaining": attempts_remaining,
            "next_attempt": next_attempt,
        }
    if attempts_remaining <= 0:
        return {
            "relaunch": False,
            "reason": f"budget-exhausted:{attempt}/{max_attempts}",
            "attempts_remaining": 0,
            "next_attempt": next_attempt,
        }
    if sf_execution_timeout is not None and per_attempt_seconds is not None:
        projected = next_attempt * per_attempt_seconds
        if projected > sf_execution_timeout:
            return {
                "relaunch": False,
                "reason": (
                    f"sf-budget-exceeded:{projected}s>{sf_execution_timeout}s "
                    f"(per_attempt={per_attempt_seconds}s next_attempt={next_attempt})"
                ),
                "attempts_remaining": attempts_remaining,
                "next_attempt": next_attempt,
            }
    return {
        "relaunch": True,
        "reason": "reclaim",
        "attempts_remaining": attempts_remaining,
        "next_attempt": next_attempt,
    }


def _split_csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_extra_tags(pairs: list[str] | None) -> dict[str, str] | None:
    """Parse repeated ``--extra-tag KEY=VALUE`` CLI args into a dict."""
    if not pairs:
        return None
    tags: dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise ValueError(f"--extra-tag must be KEY=VALUE, got: {pair!r}")
        tags[key] = value
    return tags


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
        "--extra-tag",
        dest="extra_tags",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help=(
            "Additional instance tag, KEY=VALUE. Pass multiple times for "
            ">1 tag: --extra-tag repo=foo --extra-tag sha=abc123. Rides the "
            "same RunInstances TagSpecifications entry as --name, so it is "
            "atomic with launch (no post-launch create_tags race)."
        ),
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

    decide_p = subparsers.add_parser(
        "relaunch-decision",
        help=(
            "Classify a terminated spot AND decide whether the launcher should "
            "relaunch a fresh spot (bounded by --max-attempts, optionally gated "
            "on the outer SF executionTimeout). PREFER --json: the verdict is a "
            "field and the exit status means only whether the CLI could answer. "
            "Without --json the legacy contract applies — exits 0 = RELAUNCH, "
            f"{NO_RELAUNCH_EXIT_CODE} = DO NOT relaunch, printing "
            "'relaunch|hold<TAB>reason<TAB>classification<TAB>attempts_remaining' "
            "— which makes the ordinary 'hold' answer fatal under `set -e`."
        ),
    )
    decide_p.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help=(
            "Emit the decision as a JSON object and exit 0 whenever a decision "
            "was reached, INCLUDING 'hold'. Fields: relaunch (bool), verdict "
            "('relaunch'|'hold'), reason, classification, attempts_remaining. "
            "A non-zero exit then means the CLI could not answer at all (bad "
            "input, AWS error) — never a verdict. Recommended for all callers."
        ),
    )
    decide_p.add_argument("--instance-id", required=True, help="EC2 instance ID.")
    decide_p.add_argument(
        "--region",
        default=os.environ.get("AWS_REGION", "us-east-1"),
        help="AWS region (default: $AWS_REGION or us-east-1).",
    )
    decide_p.add_argument(
        "--attempt",
        type=int,
        required=True,
        help="1-based number of the attempt that just finished (first run = 1).",
    )
    decide_p.add_argument(
        "--max-attempts",
        type=int,
        default=int(os.environ.get("MAX_SPOT_ATTEMPTS", DEFAULT_MAX_SPOT_ATTEMPTS)),
        help=(
            "Total attempts allowed incl. the first "
            f"(default $MAX_SPOT_ATTEMPTS or {DEFAULT_MAX_SPOT_ATTEMPTS})."
        ),
    )
    decide_p.add_argument(
        "--sf-execution-timeout",
        type=int,
        default=None,
        help=(
            "Outer Step-Functions executionTimeout (s) this launcher runs under. "
            "When given with --per-attempt-seconds, refuses a relaunch the SF "
            "budget cannot absorb (MAX_SPOT_ATTEMPTS ↔ SF-timeout coupling guard)."
        ),
    )
    decide_p.add_argument(
        "--per-attempt-seconds",
        type=int,
        default=None,
        help="Worst-case wall time (s) of one attempt incl. boot; pairs with --sf-execution-timeout.",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING)

    if args.cmd == "relaunch-decision":
        classified = classify_termination(args.instance_id, region=args.region)
        decision = relaunch_decision(
            classification=classified["classification"],
            attempt=args.attempt,
            max_attempts=args.max_attempts,
            sf_execution_timeout=args.sf_execution_timeout,
            per_attempt_seconds=args.per_attempt_seconds,
        )
        verdict = "relaunch" if decision["relaunch"] else "hold"

        if args.as_json:
            # The RECOMMENDED contract: the verdict is DATA, and the exit
            # status carries exactly one bit of meaning — could the CLI answer
            # at all. "hold" is the ordinary answer for every non-reclaim
            # failure, so putting it in the exit status made the normal path
            # fatal under `set -e` for six of six bash adopters (see
            # NO_RELAUNCH_EXIT_CODE). Reaching a decision is a SUCCESS whatever
            # the decision is.
            print(
                json.dumps(
                    {
                        "relaunch": bool(decision["relaunch"]),
                        "verdict": verdict,
                        "reason": str(decision["reason"]),
                        "classification": classified["classification"],
                        "attempts_remaining": decision["attempts_remaining"],
                    },
                    sort_keys=True,
                )
            )
            return 0

        # LEGACY contract, retained byte-for-byte. Changing it would silently
        # invert the branch in every caller that currently tests for 75.
        print(
            "\t".join(
                (
                    verdict,
                    str(decision["reason"]),
                    classified["classification"],
                    str(decision["attempts_remaining"]),
                )
            )
        )
        return 0 if decision["relaunch"] else NO_RELAUNCH_EXIT_CODE

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
        extra_tags = _parse_extra_tags(getattr(args, "extra_tags", None))
    except ValueError as exc:
        print(f"ec2_spot: bad input: {exc}", file=sys.stderr)
        return 2

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
            extra_tags=extra_tags,
            region=args.region,
        )
    except SpotCapacityExhausted as exc:
        print(f"ec2_spot: capacity exhausted: {exc}", file=sys.stderr)
        return CAPACITY_EXIT_CODE
    except SpotQuotaExceededError as exc:
        print(f"ec2_spot: spot quota exceeded: {exc}", file=sys.stderr)
        return QUOTA_EXIT_CODE
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
