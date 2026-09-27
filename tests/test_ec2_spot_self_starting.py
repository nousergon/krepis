"""Idempotent, self-starting launches (alpha-engine-config-I11597).

The incident these pin (alpha-engine-config-I11532, 2026-09-23): a dispatcher
Lambda launched a box, was killed while waiting for SSM, and its async retry
found a running box that never received its job. A self-starting box carries
its job in user-data, so the only step left is RunInstances, and these tests
pin that step as replay-safe: the same idempotency key never produces a second
instance, whichever path the replay takes.

``_FakeEc2`` implements the two EC2 behaviours the design rests on, rather than
a MagicMock that would accept any call:

* RunInstances with a ``ClientToken`` it has seen returns the SAME instance
  when the parameters match, and ``IdempotentParameterMismatch`` when they do
  not;
* DescribeInstances honours the ``client-token`` filter.
"""

from __future__ import annotations

import copy
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from krepis import ec2_spot
from krepis.ec2_spot import (
    CLIENT_TOKEN_MAX_LEN,
    USER_DATA_MAX_BYTES,
    IdempotencyConflict,
    SelfStartingLaunch,
    SpotCapacityExhausted,
    SpotLaunchError,
    client_token,
    find_by_idempotency_key,
    launch,
    launch_self_starting,
)

_TYPES = ["c5.large", "m5.large"]
_SUBNETS = ["subnet-A", "subnet-B"]
_USER_DATA = "#!/bin/bash\necho job\n"
_BASE = dict(
    image_id="ami-1",
    key_name="k",
    security_group_ids=["sg-1"],
    iam_instance_profile="p",
    tag_name="alpha-engine-test-spot",
    region="us-east-1",
)


def _err(code: str) -> ClientError:
    return ClientError(
        {"Error": {"Code": code, "Message": f"simulated {code}"}}, "RunInstances"
    )


class _FakeEc2:
    """EC2 with real ClientToken semantics and a capacity switchboard.

    ``refuse`` is a set of ``(market, type, subnet)`` that answer
    InsufficientInstanceCapacity; tests change it between calls to model
    capacity coming back between an original launch and its replay.
    """

    def __init__(self) -> None:
        self.instances: dict[str, dict] = {}  # id -> instance record
        self.by_token: dict[str, tuple[str, dict]] = {}  # token -> (id, params)
        self.refuse: set[tuple[str, str, str]] = set()
        self.quota_on_spot = False
        self.describe_raises: Exception | None = None
        self.describe_hides = False  # eventual consistency: nothing visible yet
        self.run_calls: list[dict] = []

    def run_instances(self, **kwargs):
        self.run_calls.append(copy.deepcopy(kwargs))
        market = "spot" if "InstanceMarketOptions" in kwargs else "on-demand"
        if market == "spot" and self.quota_on_spot:
            raise _err("MaxSpotInstanceCountExceeded")
        key = (market, kwargs["InstanceType"], kwargs["SubnetId"])
        token = kwargs.get("ClientToken")
        params = {k: v for k, v in kwargs.items() if k != "ClientToken"}
        if token and token in self.by_token:
            iid, seen = self.by_token[token]
            if seen != params:
                raise _err("IdempotentParameterMismatch")
            return {"Instances": [{"InstanceId": iid}]}
        if key in self.refuse:
            raise _err("InsufficientInstanceCapacity")
        iid = f"i-{len(self.instances) + 1:04d}"
        self.instances[iid] = {
            "InstanceId": iid,
            "ClientToken": token or "",
            "State": {"Name": "pending"},
            "Tags": [
                t
                for spec in kwargs.get("TagSpecifications", [])
                if spec["ResourceType"] == "instance"
                for t in spec["Tags"]
            ],
            "UserData": kwargs.get("UserData"),
            "market": market,
        }
        if token:
            self.by_token[token] = (iid, params)
        return {"Instances": [{"InstanceId": iid}]}

    def describe_instances(self, **kwargs):
        if self.describe_raises is not None:
            raise self.describe_raises
        if self.describe_hides:
            return {"Reservations": []}
        wanted = None
        for f in kwargs.get("Filters", []):
            if f["Name"] == "client-token":
                assert len(f["Values"]) <= 50, "probe must chunk its filter values"
                wanted = set(f["Values"])
        hits = [
            {k: v for k, v in inst.items() if k not in ("UserData", "market")}
            for inst in self.instances.values()
            if wanted is None or inst["ClientToken"] in wanted
        ]
        return {"Reservations": [{"Instances": hits}]} if hits else {"Reservations": []}


@pytest.fixture
def ec2():
    fake = _FakeEc2()
    boto3 = MagicMock()
    boto3.client.return_value = fake
    with patch.dict("sys.modules", {"boto3": boto3}):
        yield fake


def _launch(key: str = "req-1", **kw) -> SelfStartingLaunch:
    return launch_self_starting(
        _TYPES, _SUBNETS, idempotency_key=key, user_data=_USER_DATA, **{**_BASE, **kw}
    )


class TestClientToken:
    def test_deterministic_and_within_ec2_limit(self):
        a = client_token("req-1", spot=True, instance_type="c5.large", subnet_id="s")
        assert a == client_token("req-1", spot=True, instance_type="c5.large", subnet_id="s")
        assert len(a) <= CLIENT_TOKEN_MAX_LEN
        assert a.isascii()

    def test_every_attempt_parameter_changes_the_token(self):
        """One token per (market, type, subnet): rotation must never reuse a
        token with different parameters, which EC2 refuses."""
        base = dict(spot=True, instance_type="c5.large", subnet_id="s")
        tokens = {
            client_token("req-1", **base),
            client_token("req-2", **base),
            client_token("req-1", **{**base, "spot": False}),
            client_token("req-1", **{**base, "instance_type": "m5.large"}),
            client_token("req-1", **{**base, "subnet_id": "t"}),
        }
        assert len(tokens) == 5

    def test_empty_key_is_refused(self):
        with pytest.raises(ValueError):
            client_token("", spot=True, instance_type="c5.large", subnet_id="s")


class TestLaunchPlumbing:
    def test_plain_launch_sends_neither_new_key(self, ec2):
        """Existing callers' RunInstances requests are unchanged."""
        launch(_TYPES, _SUBNETS, **_BASE)
        assert "UserData" not in ec2.run_calls[0]
        assert "ClientToken" not in ec2.run_calls[0]

    def test_user_data_and_token_ride_the_run_instances_call(self, ec2):
        launch(_TYPES, _SUBNETS, user_data=_USER_DATA, idempotency_key="req-1", **_BASE)
        call = ec2.run_calls[0]
        assert call["UserData"] == _USER_DATA
        assert call["ClientToken"] == client_token(
            "req-1", spot=True, instance_type="c5.large", subnet_id="subnet-A"
        )

    def test_user_data_over_the_ec2_limit_is_refused_before_any_call(self, ec2):
        with pytest.raises(ValueError, match="fetch the job script"):
            launch(_TYPES, _SUBNETS, user_data="x" * (USER_DATA_MAX_BYTES + 1), **_BASE)
        assert ec2.run_calls == []

    def test_blank_user_data_is_refused(self, ec2):
        with pytest.raises(ValueError):
            launch(_TYPES, _SUBNETS, user_data="  \n", **_BASE)

    def test_parameter_mismatch_is_an_idempotency_conflict(self, ec2):
        launch(_TYPES, _SUBNETS, idempotency_key="req-1", extra_tags={"d": "1"}, **_BASE)
        with pytest.raises(IdempotencyConflict):
            launch(_TYPES, _SUBNETS, idempotency_key="req-1", extra_tags={"d": "2"}, **_BASE)
        assert issubclass(IdempotencyConflict, SpotLaunchError)


class TestReplayLaunchesNothing:
    def test_same_key_returns_the_same_instance(self, ec2):
        first = _launch()
        second = _launch()
        assert first == SelfStartingLaunch("i-0001", "spot", False, False)
        assert second == SelfStartingLaunch("i-0001", "spot", True, False)
        assert len(ec2.instances) == 1
        assert len(ec2.run_calls) == 1, "a replay found by the probe calls RunInstances not at all"

    def test_a_different_key_launches_its_own_box(self, ec2):
        _launch("req-1")
        other = _launch("req-2")
        assert other.instance_id == "i-0002"
        assert other.replayed is False

    def test_replay_the_probe_cannot_see_still_gets_the_same_instance(self, ec2):
        """DescribeInstances is eventually consistent. With the probe blind,
        the replay's RunInstances carries the same ClientToken and EC2 hands
        back the original instance."""
        first = _launch()
        ec2.describe_hides = True
        second = _launch()
        assert second.instance_id == first.instance_id
        assert len(ec2.instances) == 1

    def test_probe_failure_is_recorded_and_tokens_still_hold(self, ec2):
        first = _launch()
        ec2.describe_raises = RuntimeError("EC2 API degraded")
        second = _launch()
        assert second.instance_id == first.instance_id
        assert second.probe_degraded is True
        assert len(ec2.instances) == 1

    def test_capacity_that_came_back_does_not_create_a_second_box(self, ec2):
        """The case the per-attempt tokens alone would miss: the original got
        its box on the SECOND pool, the first pool has capacity again by the
        replay, and a replay walking the rotation would launch there under a
        new token. The probe finds the original first."""
        ec2.refuse = {("spot", "c5.large", "subnet-A")}
        first = _launch()
        ec2.refuse = set()
        second = _launch()
        assert second.instance_id == first.instance_id
        assert second.replayed is True
        assert len(ec2.instances) == 1

    def test_on_demand_fallback_is_found_on_replay(self, ec2):
        ec2.refuse = {("spot", t, s) for t in _TYPES for s in _SUBNETS}
        first = _launch()
        assert first.market == "on-demand"
        tags = {t["Key"]: t["Value"] for t in ec2.instances[first.instance_id]["Tags"]}
        assert tags["LaunchMarket"] == "on-demand"
        assert tags["LaunchReason"] == "capacity_exhausted"
        ec2.refuse = set()
        second = _launch()
        assert (second.instance_id, second.market, second.replayed) == (
            first.instance_id, "on-demand", True
        )
        assert len(ec2.instances) == 1

    def test_conflict_resolves_to_the_existing_instance(self, ec2):
        """A tag that moved between original and replay (e.g. a date) makes
        RunInstances refuse the token; the box is found, not relaunched."""
        first = _launch(extra_tags={"day": "2026-09-23"})
        ec2.describe_hides = True  # the up-front probe misses it
        real_find = ec2_spot.find_by_idempotency_key
        calls = {"n": 0}

        def find(*a, **k):
            calls["n"] += 1
            ec2.describe_hides = calls["n"] == 1
            return real_find(*a, **k)

        with patch.object(ec2_spot, "find_by_idempotency_key", find):
            second = _launch(extra_tags={"day": "2026-09-24"})
        assert second.instance_id == first.instance_id
        assert second.replayed is True
        assert len(ec2.instances) == 1

    def test_conflict_with_nothing_findable_raises(self, ec2):
        _launch(extra_tags={"day": "a"})
        ec2.describe_hides = True
        with pytest.raises(IdempotencyConflict):
            _launch(extra_tags={"day": "b"})
        assert len(ec2.instances) == 1


class TestSelfStartingLaunch:
    def test_user_data_and_provenance_ride_the_launch(self, ec2):
        res = _launch(extra_tags={"thinktank-run-token": "abc", "LaunchMarket": "caller"})
        inst = ec2.instances[res.instance_id]
        assert inst["UserData"] == _USER_DATA
        tags = {t["Key"]: t["Value"] for t in inst["Tags"]}
        assert tags["Name"] == "alpha-engine-test-spot"
        assert tags["thinktank-run-token"] == "abc"
        assert tags["LaunchMarket"] == "spot", "library provenance wins on collision"
        assert tags["LaunchReason"] == "spot_ok"
        assert ec2.run_calls[0]["InstanceInitiatedShutdownBehavior"] == "terminate"

    def test_force_on_demand(self, ec2):
        res = _launch(force_on_demand=True)
        assert res.market == "on-demand"
        tags = {t["Key"]: t["Value"] for t in ec2.instances[res.instance_id]["Tags"]}
        assert tags["LaunchReason"] == "force_on_demand"
        assert all("InstanceMarketOptions" not in c for c in ec2.run_calls)

    def test_quota_falls_back_to_on_demand_and_pages(self, ec2):
        ec2.quota_on_spot = True
        with patch("krepis.alerts.publish") as publish:
            res = _launch()
        assert res.market == "on-demand"
        tags = {t["Key"]: t["Value"] for t in ec2.instances[res.instance_id]["Tags"]}
        assert tags["LaunchReason"] == "quota_exceeded"
        publish.assert_called_once()
        assert publish.call_args.kwargs["dedup_key"] == "spot-quota-exceeded-us-east-1"

    def test_everything_exhausted_raises(self, ec2):
        ec2.refuse = {
            (m, t, s) for m in ("spot", "on-demand") for t in _TYPES for s in _SUBNETS
        }
        with pytest.raises(SpotCapacityExhausted):
            _launch()
        assert ec2.instances == {}

    @pytest.mark.parametrize(
        "kw",
        [
            {"idempotency_key": ""},
            {"user_data": ""},
            {"tag_name": ""},
        ],
    )
    def test_required_inputs(self, ec2, kw):
        args = dict(idempotency_key="req-1", user_data=_USER_DATA, **_BASE)
        args.update(kw)
        with pytest.raises(ValueError):
            launch_self_starting(_TYPES, _SUBNETS, **args)
        assert ec2.run_calls == []

    def test_provenance_vocabulary_matches_the_shared_contract(self):
        """Same strings nousergon_lib.spot_dispatch writes (I5727): a
        self-starting launch must be countable in the same terms."""
        assert ec2_spot.LAUNCH_MARKET_TAG == "LaunchMarket"
        assert ec2_spot.LAUNCH_REASON_TAG == "LaunchReason"
        assert {
            ec2_spot.REASON_SPOT_OK,
            ec2_spot.REASON_CAPACITY,
            ec2_spot.REASON_QUOTA,
            ec2_spot.REASON_FORCED,
        } == {"spot_ok", "capacity_exhausted", "quota_exceeded", "force_on_demand"}


class TestFindByIdempotencyKey:
    def test_none_when_nothing_launched(self, ec2):
        assert find_by_idempotency_key("req-1", _TYPES, _SUBNETS) is None

    def test_chunks_and_follows_pagination(self):
        """108 tokens for a 9x6 rotation: the probe must split them and follow
        NextToken rather than send one oversized filter."""
        types = [f"t{i}.large" for i in range(9)]
        subnets = [f"subnet-{i}" for i in range(6)]
        pages = iter(
            [
                {"Reservations": [], "NextToken": "n1"},
                {"Reservations": []},
                {"Reservations": []},
                {"Reservations": [{"Instances": [{"InstanceId": "i-x", "ClientToken": "other",
                                                  "Tags": [{"Key": "LaunchMarket", "Value": "spot"}]}]}]},
            ]
        )
        client = MagicMock()
        client.describe_instances.side_effect = lambda **kw: next(pages)
        boto3 = MagicMock()
        boto3.client.return_value = client
        with patch.dict("sys.modules", {"boto3": boto3}):
            found = find_by_idempotency_key("req-1", types, subnets)
        assert found == ("i-x", "spot")
        calls = client.describe_instances.call_args_list
        assert calls[1].kwargs["NextToken"] == "n1"
        assert all(len(c.kwargs["Filters"][0]["Values"]) <= 50 for c in calls)

    def test_probe_errors_propagate(self, ec2):
        ec2.describe_raises = RuntimeError("boom")
        with pytest.raises(RuntimeError):
            find_by_idempotency_key("req-1", _TYPES, _SUBNETS)
