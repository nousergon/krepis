"""Contract tests for the spot-teardown evidence chokepoint (I7442).

The load-bearing property: a failure's staging prefix — which holds SSM's own
upload of the full remote stdout/stderr — is never deleted while handling that
failure unless it has first been copied somewhere durable.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from krepis import spot_evidence


class FakeS3:
    """Minimal in-memory S3 with injectable failures on copy and delete."""

    def __init__(self, objects=None, fail_copy=False, fail_delete=False):
        self.objects = dict(objects or {})
        self.fail_copy = fail_copy
        self.fail_delete = fail_delete
        self.deleted = []
        self.copied = []

    def list_objects_v2(self, Bucket, Prefix, ContinuationToken=None):
        keys = sorted(k for k in self.objects if k.startswith(Prefix))
        return {"Contents": [{"Key": k} for k in keys], "IsTruncated": False}

    def copy_object(self, Bucket, Key, CopySource):
        if self.fail_copy:
            raise RuntimeError("AccessDenied")
        self.objects[Key] = self.objects[CopySource["Key"]]
        self.copied.append(Key)

    def delete_objects(self, Bucket, Delete):
        if self.fail_delete:
            raise RuntimeError("AccessDenied")
        for obj in Delete["Objects"]:
            self.objects.pop(obj["Key"], None)
            self.deleted.append(obj["Key"])
        return {}


STAGING = "s3://alpha-engine-research/tmp/spot_predictor-backtest/20260815T123311Z-i-08a4371deec28ef07"
PREFIX = "tmp/spot_predictor-backtest/20260815T123311Z-i-08a4371deec28ef07"

# The exact key shape SSM writes under OutputS3KeyPrefix, and the exact prefix
# the 2026-08-15 teardown emptied.
SSM_OUT = PREFIX + "/ssm-output/cmd-1/i-08a/awsrunShellScript/0.awsrunShellScript/stdout"


def _objects():
    return {
        SSM_OUT: b"bash: line 16: 26748 Killed  python -u backtest.py\n",
        PREFIX + "/config.yaml": b"x: 1\n",
    }


class TestParseS3Uri:
    def test_splits_bucket_and_prefix(self):
        assert spot_evidence.parse_s3_uri("s3://b/p/q") == ("b", "p/q")

    @pytest.mark.parametrize("bad", ["", "b/p", "s3://b", "s3://b/", "s3://"])
    def test_refuses_anything_that_is_not_a_prefix_under_a_bucket(self, bad):
        """A bucket-wide value must never reach a recursive delete."""
        with pytest.raises(spot_evidence.StagingUriError):
            spot_evidence.parse_s3_uri(bad)


class TestFailedRun:
    def test_evidence_is_copied_before_staging_is_deleted(self):
        s3 = FakeS3(_objects())
        result = spot_evidence.teardown(
            STAGING,
            slug="predictor-backtest",
            exit_code=1,
            s3_client=s3,
            now=datetime(2026, 8, 15, tzinfo=timezone.utc),
        )
        assert result["preserved"] is True
        assert result["deleted"] is True
        durable = (
            "_spot_evidence/predictor-backtest/2026-08-15/"
            "20260815T123311Z-i-08a4371deec28ef07"
        )
        preserved = [k for k in s3.objects if k.startswith(durable)]
        assert any(k.endswith("/stdout") for k in preserved)
        assert b"Killed" in s3.objects[
            durable + "/ssm-output/cmd-1/i-08a/awsrunShellScript/0.awsrunShellScript/stdout"
        ]
        assert SSM_OUT not in s3.objects
        assert result["durable_uri"].endswith(durable + "/")

    def test_a_failed_copy_leaves_staging_in_place(self):
        """The whole point: never delete the only remaining copy."""
        s3 = FakeS3(_objects(), fail_copy=True)
        result = spot_evidence.teardown(
            STAGING, slug="predictor-backtest", exit_code=1, s3_client=s3
        )
        assert result["preserved"] is False
        assert result["deleted"] is False
        assert s3.deleted == []
        assert SSM_OUT in s3.objects
        assert "LEFT IN PLACE" in result["detail"]

    def test_an_empty_staging_prefix_still_cleans(self):
        s3 = FakeS3({})
        result = spot_evidence.teardown(STAGING, slug="s", exit_code=1, s3_client=s3)
        assert result["preserved"] is True
        assert result["deleted"] is True

    def test_a_failed_delete_is_reported_not_raised(self):
        s3 = FakeS3(_objects(), fail_delete=True)
        result = spot_evidence.teardown(STAGING, slug="s", exit_code=1, s3_client=s3)
        assert result["preserved"] is True
        assert result["deleted"] is False


class TestSuccessfulRun:
    def test_nothing_is_retained_when_the_run_succeeded(self):
        """Cost: a green weekly run must not start accruing log storage."""
        s3 = FakeS3(_objects())
        result = spot_evidence.teardown(STAGING, slug="s", exit_code=0, s3_client=s3)
        assert result["preserved"] is False
        assert result["durable_uri"] is None
        assert result["deleted"] is True
        assert s3.copied == []
        assert s3.objects == {}


class TestNeverRaises:
    def test_a_bad_staging_uri_is_refused_without_touching_anything(self):
        s3 = FakeS3(_objects())
        result = spot_evidence.teardown(
            "not-a-uri", slug="s", exit_code=1, s3_client=s3
        )
        assert result["deleted"] is False
        assert s3.deleted == []
        assert "refused" in result["detail"]

    def test_dry_run_touches_nothing(self):
        s3 = FakeS3(_objects())
        result = spot_evidence.teardown(
            STAGING, slug="s", exit_code=1, s3_client=s3, dry_run=True
        )
        assert s3.copied == [] and s3.deleted == []
        assert "would preserve" in result["detail"]


class TestOrderingIsStructural:
    """The ordering must be a property of the call graph, not a comment."""

    def test_the_staging_delete_has_only_its_two_guarded_call_sites(self):
        import inspect

        source = inspect.getsource(spot_evidence)
        # One definition plus the two guarded calls in `teardown` (the success
        # branch, where there is no failure whose evidence could be lost, and
        # the branch reached only after `_preserve` returned ok). A third
        # occurrence means someone has reintroduced an unguarded delete —
        # retention sweeps must use `_delete_prefix`.
        assert source.count("_delete_staging(") == 3

    def test_delete_is_unreachable_when_preserve_fails(self):
        calls = []
        real_delete = spot_evidence._delete_staging

        def spy(*a, **kw):
            calls.append(a)
            return real_delete(*a, **kw)

        s3 = FakeS3(_objects(), fail_copy=True)
        spot_evidence._delete_staging = spy
        try:
            spot_evidence.teardown(STAGING, slug="s", exit_code=1, s3_client=s3)
        finally:
            spot_evidence._delete_staging = real_delete
        assert calls == []


class TestCli:
    def test_render_lines(self):
        assert "preserved" in spot_evidence._render(
            {"preserved": True, "durable_uri": "s3://b/k/", "detail": "d"}
        )
        assert "spot_evidence:" in spot_evidence._render(
            {"preserved": False, "exit_code": 0, "detail": "deleted"}
        )
        assert "spot_evidence:" in spot_evidence._render(
            {"preserved": False, "exit_code": 1, "detail": "LEFT IN PLACE"}
        )

    def test_main_never_returns_non_zero(self, capsys, monkeypatch):
        """A janitor that can overwrite the workload's exit status is the
        same defect class one layer up."""
        monkeypatch.setattr(
            spot_evidence,
            "teardown",
            lambda *a, **kw: {
                "preserved": False,
                "exit_code": 1,
                "detail": "EVIDENCE PRESERVATION FAILED",
                "durable_uri": None,
            },
        )
        rc = spot_evidence.main(
            ["teardown", "--staging", STAGING, "--slug", "s", "--exit-code", "1"]
        )
        assert rc == 0
        assert "EVIDENCE PRESERVATION FAILED" in capsys.readouterr().out

    def test_main_json(self, capsys, monkeypatch):
        monkeypatch.setattr(
            spot_evidence, "teardown", lambda *a, **kw: {"preserved": True, "k": 1}
        )
        rc = spot_evidence.main(
            [
                "teardown",
                "--staging",
                STAGING,
                "--slug",
                "s",
                "--exit-code",
                "1",
                "--json",
            ]
        )
        assert rc == 0
        import json as _json

        assert _json.loads(capsys.readouterr().out)["preserved"] is True

    def test_main_dry_run_end_to_end_without_credentials(self, capsys):
        rc = spot_evidence.main(
            [
                "teardown",
                "--staging",
                STAGING,
                "--slug",
                "s",
                "--exit-code",
                "0",
                "--dry-run",
            ]
        )
        assert rc == 0
        assert "spot_evidence:" in capsys.readouterr().out


class TestPagination:
    def test_more_than_one_page_of_keys_is_handled(self):
        class Paged(FakeS3):
            def __init__(self, objects):
                super().__init__(objects)
                self.pages = 0

            def list_objects_v2(self, Bucket, Prefix, ContinuationToken=None):
                keys = sorted(k for k in self.objects if k.startswith(Prefix))
                if ContinuationToken is None and len(keys) > 1:
                    self.pages += 1
                    return {
                        "Contents": [{"Key": keys[0]}],
                        "IsTruncated": True,
                        "NextContinuationToken": "t",
                    }
                return {
                    "Contents": [{"Key": k} for k in keys],
                    "IsTruncated": False,
                }

        s3 = Paged(_objects())
        result = spot_evidence.teardown(STAGING, slug="s", exit_code=1, s3_client=s3)
        assert result["preserved"] is True


class _PruneS3(FakeS3):
    """FakeS3 that also answers Delimiter='/' CommonPrefixes listings."""

    def list_objects_v2(self, Bucket, Prefix, ContinuationToken=None, Delimiter=None):
        if Delimiter != "/":
            return super().list_objects_v2(Bucket, Prefix, ContinuationToken)
        children = set()
        for key in self.objects:
            if not key.startswith(Prefix):
                continue
            rest = key[len(Prefix) :]
            if "/" in rest:
                children.add(Prefix + rest.split("/", 1)[0] + "/")
        return {
            "CommonPrefixes": [{"Prefix": c} for c in sorted(children)],
            "IsTruncated": False,
        }


NOW = datetime(2026, 8, 15, tzinfo=timezone.utc)


class TestRetentionSweep:
    """Retires crucible-backtester#675's per-repo bash prune (config-I7396)."""

    def _objects_with_history(self):
        objects = _objects()
        objects["tmp/spot_predictor-backtest/20260101T000000Z-i-old/ssm-output/x"] = b"o"
        objects["tmp/spot_predictor-backtest/20260814T000000Z-i-new/ssm-output/x"] = b"n"
        objects["_spot_evidence/predictor-backtest/2026-01-01/run/x"] = b"o"
        objects["_spot_evidence/predictor-backtest/2026-08-10/run/x"] = b"n"
        return objects

    def test_stale_staging_and_stale_evidence_are_pruned(self):
        s3 = _PruneS3(self._objects_with_history())
        result = spot_evidence.teardown(
            STAGING, slug="predictor-backtest", exit_code=0, s3_client=s3, now=NOW
        )
        assert result["pruned"] == 2
        assert not any("20260101T000000Z" in k for k in s3.objects)
        assert not any("2026-01-01" in k for k in s3.objects)

    def test_recent_prefixes_survive(self):
        s3 = _PruneS3(self._objects_with_history())
        spot_evidence.teardown(
            STAGING, slug="predictor-backtest", exit_code=0, s3_client=s3, now=NOW
        )
        assert any("20260814T000000Z" in k for k in s3.objects)
        assert any("2026-08-10" in k for k in s3.objects)

    def test_this_run_s_freshly_preserved_evidence_is_never_pruned(self):
        s3 = _PruneS3(self._objects_with_history())
        result = spot_evidence.teardown(
            STAGING, slug="predictor-backtest", exit_code=1, s3_client=s3, now=NOW
        )
        assert result["preserved"] is True
        assert any(k.startswith(result["durable_uri"][len("s3://alpha-engine-research/"):]) for k in s3.objects)

    def test_an_unparseable_leaf_is_left_alone(self):
        """Deleting on a failure to understand a key is how a sweep becomes an
        outage."""
        objects = _objects()
        objects["tmp/spot_predictor-backtest/manual-scratch/x"] = b"keep"
        s3 = _PruneS3(objects)
        spot_evidence.teardown(
            STAGING, slug="predictor-backtest", exit_code=0, s3_client=s3, now=NOW
        )
        assert "tmp/spot_predictor-backtest/manual-scratch/x" in s3.objects

    def test_no_prune_flag_skips_the_sweep(self):
        s3 = _PruneS3(self._objects_with_history())
        result = spot_evidence.teardown(
            STAGING,
            slug="predictor-backtest",
            exit_code=0,
            s3_client=s3,
            now=NOW,
            prune=False,
        )
        assert result["pruned"] is None
        assert any("20260101T000000Z" in k for k in s3.objects)

    def test_a_broken_prune_never_fails_the_teardown(self):
        class BadList(_PruneS3):
            def list_objects_v2(self, Bucket, Prefix, ContinuationToken=None, Delimiter=None):
                if Delimiter == "/":
                    raise RuntimeError("AccessDenied")
                return super().list_objects_v2(Bucket, Prefix, ContinuationToken)

        s3 = BadList(self._objects_with_history())
        result = spot_evidence.teardown(
            STAGING, slug="predictor-backtest", exit_code=1, s3_client=s3, now=NOW
        )
        assert result["preserved"] is True and result["deleted"] is True

    @pytest.mark.parametrize(
        "leaf,expected",
        [
            ("20260815T123311Z-i-08a", 2026),
            ("2026-08-15", 2026),
            ("manual-scratch", None),
            ("", None),
        ],
    )
    def test_leaf_date_parsing(self, leaf, expected):
        got = spot_evidence._leaf_date("root/" + leaf + "/")
        assert (got.year if got else None) == expected
