"""
Unit tests for ``krepis.ec2_spot``.

Pins the capacity-resilience contract that the 3 spot launchers will
rely on after the 2026-05-22 lift from inline-bash-run-instances to
lib CLI:

* first success short-circuits — no further attempts
* InsufficientInstanceCapacity rotates (type, subnet) iteration order
* exhaustion raises SpotCapacityExhausted listing every attempt
* non-capacity errors (Auth, AMI not found, quota) raise SpotLaunchError
  immediately (NO retry — those don't get better by rotating)
* CLI returns InstanceId on stdout, exits 0 on success
* CLI exits CAPACITY_EXIT_CODE (64) on capacity exhaustion — distinct
  from generic failure exit 1 — so bash callers can distinguish
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from krepis import ec2_spot


def _capacity_error(code: str = "InsufficientInstanceCapacity"):
    from botocore.exceptions import ClientError

    return ClientError(
        error_response={
            "Error": {"Code": code, "Message": f"simulated {code}"},
        },
        operation_name="RunInstances",
    )


def _other_error(code: str = "AuthFailure"):
    from botocore.exceptions import ClientError

    return ClientError(
        error_response={
            "Error": {"Code": code, "Message": f"simulated {code}"},
        },
        operation_name="RunInstances",
    )


@pytest.fixture
def fake_boto3():
    """boto3 stub returning a configurable RunInstances mock."""
    ec2_client = MagicMock()
    fake = MagicMock()
    fake.client.return_value = ec2_client
    return fake, ec2_client


_BASE_KWARGS = dict(
    image_id="ami-deadbeef",
    key_name="test-key",
    security_group_ids=["sg-1"],
    iam_instance_profile="test-profile",
    region="us-east-1",
)


class TestLaunchHappyPath:
    def test_first_combination_succeeds_returns_instance_id(self, fake_boto3):
        fake, ec2 = fake_boto3
        ec2.run_instances.return_value = {"Instances": [{"InstanceId": "i-aaa"}]}
        with patch.dict("sys.modules", {"boto3": fake}):
            instance_id = ec2_spot.launch(
                instance_types=["c5.large"],
                subnets=["subnet-A"],
                **_BASE_KWARGS,
            )
        assert instance_id == "i-aaa"
        ec2.run_instances.assert_called_once()

    def test_kwargs_shape_matches_existing_launcher(self, fake_boto3):
        fake, ec2 = fake_boto3
        ec2.run_instances.return_value = {"Instances": [{"InstanceId": "i-shape"}]}
        with patch.dict("sys.modules", {"boto3": fake}):
            ec2_spot.launch(
                instance_types=["c5.large"],
                subnets=["subnet-A"],
                tag_name="alpha-engine-test-20260522",
                **_BASE_KWARGS,
            )
        kwargs = ec2.run_instances.call_args.kwargs
        assert kwargs["ImageId"] == "ami-deadbeef"
        assert kwargs["InstanceType"] == "c5.large"
        assert kwargs["KeyName"] == "test-key"
        assert kwargs["SecurityGroupIds"] == ["sg-1"]
        assert kwargs["SubnetId"] == "subnet-A"
        assert kwargs["IamInstanceProfile"] == {"Name": "test-profile"}
        assert kwargs["MinCount"] == 1
        assert kwargs["MaxCount"] == 1
        assert kwargs["InstanceInitiatedShutdownBehavior"] == "terminate"
        # Spot config
        market = kwargs["InstanceMarketOptions"]
        assert market["MarketType"] == "spot"
        assert market["SpotOptions"]["SpotInstanceType"] == "one-time"
        assert market["SpotOptions"]["InstanceInterruptionBehavior"] == "terminate"
        # Block device (gp3, 30 GB default)
        bdm = kwargs["BlockDeviceMappings"][0]
        assert bdm["DeviceName"] == "/dev/xvda"
        assert bdm["Ebs"] == {"VolumeSize": 30, "VolumeType": "gp3"}
        # Name tag
        tags = kwargs["TagSpecifications"][0]
        assert tags["ResourceType"] == "instance"
        assert tags["Tags"] == [
            {"Key": "Name", "Value": "alpha-engine-test-20260522"}
        ]

    def test_no_spot_omits_market_options(self, fake_boto3):
        fake, ec2 = fake_boto3
        ec2.run_instances.return_value = {"Instances": [{"InstanceId": "i-od"}]}
        with patch.dict("sys.modules", {"boto3": fake}):
            ec2_spot.launch(
                instance_types=["c5.large"],
                subnets=["subnet-A"],
                spot=False,
                **_BASE_KWARGS,
            )
        kwargs = ec2.run_instances.call_args.kwargs
        assert "InstanceMarketOptions" not in kwargs


class TestRotation:
    """The whole point of this module — rotate on capacity errors."""

    def test_rotates_subnet_within_same_type(self, fake_boto3):
        fake, ec2 = fake_boto3
        ec2.run_instances.side_effect = [
            _capacity_error(),
            _capacity_error(),
            {"Instances": [{"InstanceId": "i-third-subnet"}]},
        ]
        with patch.dict("sys.modules", {"boto3": fake}):
            instance_id = ec2_spot.launch(
                instance_types=["c5.large"],
                subnets=["subnet-A", "subnet-B", "subnet-C"],
                **_BASE_KWARGS,
            )
        assert instance_id == "i-third-subnet"
        assert ec2.run_instances.call_count == 3
        # 3rd call used subnet-C
        assert (
            ec2.run_instances.call_args_list[2].kwargs["SubnetId"] == "subnet-C"
        )

    def test_rotates_to_next_type_after_exhausting_subnets(self, fake_boto3):
        fake, ec2 = fake_boto3
        ec2.run_instances.side_effect = [
            _capacity_error(),
            _capacity_error(),
            _capacity_error(),  # All 3 subnets exhausted for c5.large
            {"Instances": [{"InstanceId": "i-m5"}]},  # m5.large in subnet-A
        ]
        with patch.dict("sys.modules", {"boto3": fake}):
            instance_id = ec2_spot.launch(
                instance_types=["c5.large", "m5.large"],
                subnets=["subnet-A", "subnet-B", "subnet-C"],
                **_BASE_KWARGS,
            )
        assert instance_id == "i-m5"
        # 4th attempt = m5.large in subnet-A (first of new type)
        last = ec2.run_instances.call_args_list[3].kwargs
        assert last["InstanceType"] == "m5.large"
        assert last["SubnetId"] == "subnet-A"

    @pytest.mark.parametrize(
        "code",
        [
            "InsufficientInstanceCapacity",
            "InsufficientHostCapacity",
            "Unsupported",
            "InvalidAvailabilityZone",
        ],
    )
    def test_each_capacity_code_rotates(self, fake_boto3, code):
        fake, ec2 = fake_boto3
        ec2.run_instances.side_effect = [
            _capacity_error(code),
            {"Instances": [{"InstanceId": "i-after-rotation"}]},
        ]
        with patch.dict("sys.modules", {"boto3": fake}):
            instance_id = ec2_spot.launch(
                instance_types=["c5.large"],
                subnets=["subnet-A", "subnet-B"],
                **_BASE_KWARGS,
            )
        assert instance_id == "i-after-rotation"


class TestFailureModes:
    def test_capacity_exhausted_raises_with_full_attempt_list(self, fake_boto3):
        fake, ec2 = fake_boto3
        # 2 types × 3 subnets = 6 attempts, all capacity errors
        ec2.run_instances.side_effect = [_capacity_error()] * 6
        with patch.dict("sys.modules", {"boto3": fake}):
            with pytest.raises(ec2_spot.SpotCapacityExhausted) as exc_info:
                ec2_spot.launch(
                    instance_types=["c5.large", "m5.large"],
                    subnets=["subnet-A", "subnet-B", "subnet-C"],
                    **_BASE_KWARGS,
                )
        msg = str(exc_info.value)
        assert "6 attempts" in msg
        # Every (type, subnet) shows up in the error message
        for t in ("c5.large", "m5.large"):
            for s in ("subnet-A", "subnet-B", "subnet-C"):
                assert f"{t}@{s}" in msg
        assert ec2.run_instances.call_count == 6

    def test_non_capacity_error_raises_immediately_no_rotation(self, fake_boto3):
        fake, ec2 = fake_boto3
        ec2.run_instances.side_effect = [_other_error("AuthFailure")]
        with patch.dict("sys.modules", {"boto3": fake}):
            with pytest.raises(ec2_spot.SpotLaunchError) as exc_info:
                ec2_spot.launch(
                    instance_types=["c5.large"],
                    subnets=["subnet-A", "subnet-B", "subnet-C"],
                    **_BASE_KWARGS,
                )
        assert "AuthFailure" in str(exc_info.value)
        # ONE attempt — non-capacity error did not retry
        assert ec2.run_instances.call_count == 1

    def test_empty_types_raises_value_error(self):
        with pytest.raises(ValueError, match="instance_types"):
            ec2_spot.launch(
                instance_types=[],
                subnets=["subnet-A"],
                **_BASE_KWARGS,
            )

    def test_empty_subnets_raises_value_error(self):
        with pytest.raises(ValueError, match="subnets"):
            ec2_spot.launch(
                instance_types=["c5.large"],
                subnets=[],
                **_BASE_KWARGS,
            )

    def test_capacity_exhausted_is_a_spot_launch_error(self):
        """Catching SpotLaunchError should also catch the capacity subclass —
        bash callers that wrap can distinguish via exit codes instead."""
        assert issubclass(
            ec2_spot.SpotCapacityExhausted, ec2_spot.SpotLaunchError
        )


class TestCli:
    def test_launch_subcommand_prints_instance_id_to_stdout(
        self, fake_boto3, capfd
    ):
        fake, ec2 = fake_boto3
        ec2.run_instances.return_value = {"Instances": [{"InstanceId": "i-cli-ok"}]}
        with patch.dict("sys.modules", {"boto3": fake}):
            rc = ec2_spot.main(
                [
                    "launch",
                    "--types",
                    "c5.large",
                    "--subnets",
                    "subnet-A",
                    "--image-id",
                    "ami-deadbeef",
                    "--key-name",
                    "k",
                    "--security-group",
                    "sg-1",
                    "--iam-profile",
                    "p",
                ]
            )
        assert rc == 0
        out = capfd.readouterr().out.strip()
        assert out == "i-cli-ok"  # ONLY the InstanceId on stdout

    def test_capacity_exhaustion_returns_64(self, fake_boto3):
        fake, ec2 = fake_boto3
        ec2.run_instances.side_effect = [_capacity_error()] * 6
        with patch.dict("sys.modules", {"boto3": fake}):
            rc = ec2_spot.main(
                [
                    "launch",
                    "--types",
                    "c5.large,m5.large",
                    "--subnets",
                    "subnet-A,subnet-B,subnet-C",
                    "--image-id",
                    "ami-X",
                    "--key-name",
                    "k",
                    "--security-group",
                    "sg-1",
                    "--iam-profile",
                    "p",
                ]
            )
        assert rc == ec2_spot.CAPACITY_EXIT_CODE
        assert rc == 64

    def test_non_capacity_failure_returns_1(self, fake_boto3):
        fake, ec2 = fake_boto3
        ec2.run_instances.side_effect = [_other_error("AuthFailure")]
        with patch.dict("sys.modules", {"boto3": fake}):
            rc = ec2_spot.main(
                [
                    "launch",
                    "--types",
                    "c5.large",
                    "--subnets",
                    "subnet-A",
                    "--image-id",
                    "ami-X",
                    "--key-name",
                    "k",
                    "--security-group",
                    "sg-1",
                    "--iam-profile",
                    "p",
                ]
            )
        assert rc == 1

    def test_multiple_security_groups_via_repeated_flag(self, fake_boto3):
        fake, ec2 = fake_boto3
        ec2.run_instances.return_value = {"Instances": [{"InstanceId": "i-multi-sg"}]}
        with patch.dict("sys.modules", {"boto3": fake}):
            ec2_spot.main(
                [
                    "launch",
                    "--types",
                    "c5.large",
                    "--subnets",
                    "subnet-A",
                    "--image-id",
                    "ami-X",
                    "--key-name",
                    "k",
                    "--security-group",
                    "sg-1",
                    "--security-group",
                    "sg-2",
                    "--iam-profile",
                    "p",
                ]
            )
        kwargs = ec2.run_instances.call_args.kwargs
        assert kwargs["SecurityGroupIds"] == ["sg-1", "sg-2"]

    def test_no_spot_flag_disables_market_options(self, fake_boto3):
        fake, ec2 = fake_boto3
        ec2.run_instances.return_value = {"Instances": [{"InstanceId": "i-od-cli"}]}
        with patch.dict("sys.modules", {"boto3": fake}):
            ec2_spot.main(
                [
                    "launch",
                    "--types",
                    "c5.large",
                    "--subnets",
                    "subnet-A",
                    "--image-id",
                    "ami-X",
                    "--key-name",
                    "k",
                    "--security-group",
                    "sg-1",
                    "--iam-profile",
                    "p",
                    "--no-spot",
                ]
            )
        assert "InstanceMarketOptions" not in ec2.run_instances.call_args.kwargs

    def test_name_flag_adds_name_tag(self, fake_boto3):
        fake, ec2 = fake_boto3
        ec2.run_instances.return_value = {"Instances": [{"InstanceId": "i-tagged"}]}
        with patch.dict("sys.modules", {"boto3": fake}):
            ec2_spot.main(
                [
                    "launch",
                    "--types",
                    "c5.large",
                    "--subnets",
                    "subnet-A",
                    "--image-id",
                    "ami-X",
                    "--key-name",
                    "k",
                    "--security-group",
                    "sg-1",
                    "--iam-profile",
                    "p",
                    "--name",
                    "alpha-engine-backtest-20260522",
                ]
            )
        kwargs = ec2.run_instances.call_args.kwargs
        assert kwargs["TagSpecifications"][0]["Tags"] == [
            {"Key": "Name", "Value": "alpha-engine-backtest-20260522"}
        ]

    def test_missing_subcommand_errors(self):
        with pytest.raises(SystemExit):
            ec2_spot.main([])

    def test_help_exits_clean(self):
        with pytest.raises(SystemExit) as exc:
            ec2_spot.main(["--help"])
        assert exc.value.code == 0


class TestCsvSplitting:
    """The CLI splits --types and --subnets on commas with trim. Bash
    callers pass `--types c5.large,m5.large` directly."""

    def test_split_basic(self):
        assert ec2_spot._split_csv("a,b,c") == ["a", "b", "c"]

    def test_split_trims_whitespace(self):
        assert ec2_spot._split_csv("a, b , c") == ["a", "b", "c"]

    def test_split_drops_empty(self):
        assert ec2_spot._split_csv("a,,b,") == ["a", "b"]


class TestModuleEntrypoint:
    def test_module_has_main_guard(self):
        assert callable(ec2_spot.main)


def _describe_resp(state, *, reason_code=None, transition_reason=""):
    """Shape a describe-instances response for one instance."""
    inst = {
        "State": {"Name": state},
        "StateTransitionReason": transition_reason,
    }
    if reason_code is not None:
        inst["StateReason"] = {"Code": reason_code, "Message": reason_code}
    return {"Reservations": [{"Instances": [inst]}]}


class TestClassifyTermination:
    """Pins the spot-reclaim classifier lifted from spot_backtest.sh (the
    2026-06-06 field-mismatch fix: classify on StateReason.Code, not the
    StateTransitionReason-only string)."""

    def _run(self, fake_boto3, resp=None, raise_exc=None, sir_code=None):
        fake, ec2 = fake_boto3
        # SIR query defaults to "no spot request found" unless a code is given.
        if sir_code is None:
            ec2.describe_spot_instance_requests.return_value = {"SpotInstanceRequests": []}
        else:
            ec2.describe_spot_instance_requests.return_value = {
                "SpotInstanceRequests": [{"Status": {"Code": sir_code}}]
            }
        if raise_exc is not None:
            ec2.describe_instances.side_effect = raise_exc
        else:
            ec2.describe_instances.return_value = (
                resp if resp is not None else {"Reservations": []}
            )
        with patch.dict("sys.modules", {"boto3": fake}):
            return ec2_spot.classify_termination("i-abc", region="us-east-1")

    def test_sir_status_no_capacity_is_reclaim(self, fake_boto3):
        # Most authoritative signal — the Spot Instance Request status — even
        # when the instance is already gone (no Reservations).
        r = self._run(fake_boto3, {"Reservations": []}, sir_code="instance-terminated-no-capacity")
        assert r["classification"] == "reclaim"
        assert r["sir_code"] == "instance-terminated-no-capacity"

    def test_sir_marked_for_termination_is_reclaim(self, fake_boto3):
        r = self._run(fake_boto3, _describe_resp("running"), sir_code="marked-for-termination")
        assert r["classification"] == "reclaim"

    def test_sir_fulfilled_running_is_other(self, fake_boto3):
        # A healthy/fulfilled SIR on a running instance is NOT a reclaim.
        r = self._run(fake_boto3, _describe_resp("running"), sir_code="fulfilled")
        assert r["classification"] == "other"

    def test_statereason_code_spot_termination_is_reclaim(self, fake_boto3):
        r = self._run(fake_boto3, _describe_resp("shutting-down", reason_code="Server.SpotInstanceTermination"))
        assert r["classification"] == "reclaim"
        assert r["reason_code"] == "Server.SpotInstanceTermination"

    def test_insufficient_capacity_code_is_reclaim(self, fake_boto3):
        r = self._run(fake_boto3, _describe_resp("terminated", reason_code="Server.InsufficientInstanceCapacity"))
        assert r["classification"] == "reclaim"

    def test_service_initiated_shutting_down_is_reclaim(self, fake_boto3):
        # The exact 2026-06-06 signature: no StateReason.Code, only the
        # "Service initiated" transition reason on a shutting-down worker.
        r = self._run(fake_boto3, _describe_resp("shutting-down", transition_reason="Service initiated (2026-06-06 16:04:58 GMT)"))
        assert r["classification"] == "reclaim"

    def test_service_initiated_terminated_is_reclaim(self, fake_boto3):
        r = self._run(fake_boto3, _describe_resp("terminated", transition_reason="Service initiated (x)"))
        assert r["classification"] == "reclaim"

    def test_running_real_crash_is_other(self, fake_boto3):
        # A genuine in-instance failure leaves the worker running until the
        # dispatcher terminates it — must NOT classify as reclaim (no blind retry).
        r = self._run(fake_boto3, _describe_resp("running", transition_reason=""))
        assert r["classification"] == "other"

    def test_user_initiated_shutdown_is_other(self, fake_boto3):
        r = self._run(fake_boto3, _describe_resp("terminated", reason_code="Client.UserInitiatedShutdown", transition_reason="User initiated"))
        assert r["classification"] == "other"

    def test_describe_error_with_no_sir_is_unknown(self, fake_boto3):
        # Neither the SIR nor the instance is readable → genuinely unknown.
        r = self._run(fake_boto3, raise_exc=_other_error("RequestLimitExceeded"))
        assert r["classification"] == "unknown"

    def test_no_instances_no_sir_is_unknown(self, fake_boto3):
        r = self._run(fake_boto3, {"Reservations": []})
        assert r["classification"] == "unknown"

    def test_cli_emits_tab_separated_line(self, fake_boto3, capsys):
        fake, ec2 = fake_boto3
        ec2.describe_spot_instance_requests.return_value = {
            "SpotInstanceRequests": [{"Status": {"Code": "instance-terminated-no-capacity"}}]
        }
        ec2.describe_instances.return_value = _describe_resp(
            "shutting-down", reason_code="Server.SpotInstanceTermination",
            transition_reason="Service initiated (ts)",
        )
        with patch.dict("sys.modules", {"boto3": fake}):
            rc = ec2_spot.main(["classify-termination", "--instance-id", "i-abc", "--region", "us-east-1"])
        assert rc == 0
        out = capsys.readouterr().out.strip()
        fields = out.split("\t")
        assert fields[0] == "reclaim"
        assert fields[1] == "shutting-down"
        assert fields[2] == "Server.SpotInstanceTermination"
        assert fields[4] == "instance-terminated-no-capacity"  # sir_code appended
