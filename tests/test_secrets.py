"""
Unit tests for ``krepis.secrets``.

Locks down the resolution-order contract: cache priority, source-toggle
behavior (env / ssm / auto), SSM-unavailable latching, and the required /
default / SecretNotFoundError surface.
"""

from __future__ import annotations

from unittest.mock import patch

import boto3
import pytest
from botocore.exceptions import EndpointConnectionError
from moto import mock_aws

from krepis import secrets as secrets_mod
from krepis.secrets import (
    SOURCE_TOGGLE_ENV,
    SSM_PREFIX,
    SecretNotFoundError,
    clear_cache,
    get_secret,
)


@pytest.fixture(autouse=True)
def _reset_cache():
    """Every test starts with an empty cache + re-armed SSM probe."""
    clear_cache()
    yield
    clear_cache()


@pytest.fixture
def ssm_seeded():
    """moto-mocked SSM with a small seeded keyset under /alpha-engine/."""
    with mock_aws():
        client = boto3.client("ssm", region_name="us-east-1")
        client.put_parameter(
            Name="/alpha-engine/POLYGON_API_KEY",
            Type="SecureString",
            Value="poly-secret-from-ssm",
        )
        client.put_parameter(
            Name="/alpha-engine/ANTHROPIC_API_KEY",
            Type="SecureString",
            Value="ant-secret-from-ssm",
        )
        yield client


# ── Name validation ─────────────────────────────────────────────────────────


class TestNameValidation:
    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            get_secret("")

    def test_name_with_slash_raises(self):
        with pytest.raises(ValueError, match="must not contain '/'"):
            get_secret("alpha-engine/POLYGON_API_KEY")


# ── Resolution order: source=env ────────────────────────────────────────────


class TestSourceEnv:
    def test_env_only_when_source_is_env(self, monkeypatch, ssm_seeded):
        """source=env must skip SSM entirely, even if SSM has the value."""
        monkeypatch.setenv(SOURCE_TOGGLE_ENV, "env")
        monkeypatch.setenv("POLYGON_API_KEY", "from-env")
        assert get_secret("POLYGON_API_KEY") == "from-env"

    def test_env_only_missing_raises(self, monkeypatch, ssm_seeded):
        """source=env + secret missing from env must raise even if SSM has it."""
        monkeypatch.setenv(SOURCE_TOGGLE_ENV, "env")
        monkeypatch.delenv("POLYGON_API_KEY", raising=False)
        with pytest.raises(SecretNotFoundError, match="POLYGON_API_KEY"):
            get_secret("POLYGON_API_KEY")

    def test_unknown_source_falls_back_to_auto(self, monkeypatch, ssm_seeded):
        """Unknown source toggle value logs a warning and uses auto."""
        monkeypatch.setenv(SOURCE_TOGGLE_ENV, "vault")
        # SSM (auto path) wins.
        assert get_secret("POLYGON_API_KEY") == "poly-secret-from-ssm"


# ── Resolution order: source=ssm ────────────────────────────────────────────


class TestSourceSsm:
    def test_ssm_only_when_source_is_ssm(self, monkeypatch, ssm_seeded):
        monkeypatch.setenv(SOURCE_TOGGLE_ENV, "ssm")
        # Env has a value but source=ssm must ignore it.
        monkeypatch.setenv("POLYGON_API_KEY", "from-env-ignored")
        assert get_secret("POLYGON_API_KEY") == "poly-secret-from-ssm"

    def test_ssm_only_missing_raises_even_if_env_has_it(self, monkeypatch, ssm_seeded):
        """source=ssm must NOT fall back to env when SSM misses."""
        monkeypatch.setenv(SOURCE_TOGGLE_ENV, "ssm")
        monkeypatch.setenv("NEW_KEY", "from-env-ignored")
        with pytest.raises(SecretNotFoundError):
            get_secret("NEW_KEY")


# ── Resolution order: source=auto (default) ─────────────────────────────────


class TestSourceAuto:
    def test_ssm_wins_over_env(self, monkeypatch, ssm_seeded):
        """auto: SSM read is tried first; env is the fallback."""
        monkeypatch.delenv(SOURCE_TOGGLE_ENV, raising=False)
        monkeypatch.setenv("POLYGON_API_KEY", "from-env-fallback-unused")
        assert get_secret("POLYGON_API_KEY") == "poly-secret-from-ssm"

    def test_env_used_when_ssm_misses(self, monkeypatch, ssm_seeded):
        """auto: if SSM has no key, env is the fallback."""
        monkeypatch.delenv(SOURCE_TOGGLE_ENV, raising=False)
        monkeypatch.setenv("DEV_ONLY_KEY", "from-env-fallback")
        assert get_secret("DEV_ONLY_KEY") == "from-env-fallback"

    def test_missing_everywhere_required_raises(self, monkeypatch, ssm_seeded):
        monkeypatch.delenv(SOURCE_TOGGLE_ENV, raising=False)
        monkeypatch.delenv("GHOST_KEY", raising=False)
        with pytest.raises(SecretNotFoundError, match="GHOST_KEY"):
            get_secret("GHOST_KEY")

    def test_missing_everywhere_optional_returns_none(self, monkeypatch, ssm_seeded):
        monkeypatch.delenv(SOURCE_TOGGLE_ENV, raising=False)
        monkeypatch.delenv("GHOST_KEY", raising=False)
        assert get_secret("GHOST_KEY", required=False) is None

    def test_missing_everywhere_optional_with_default(self, monkeypatch, ssm_seeded):
        monkeypatch.delenv(SOURCE_TOGGLE_ENV, raising=False)
        monkeypatch.delenv("GHOST_KEY", raising=False)
        assert get_secret("GHOST_KEY", required=False, default="fallback") == "fallback"


# ── Cache behavior ──────────────────────────────────────────────────────────


class TestCache:
    def test_second_call_hits_cache_not_ssm(self, monkeypatch, ssm_seeded):
        """SSM is called exactly once per (process, secret) — subsequent reads
        return the cached value even after the underlying SSM param changes."""
        monkeypatch.delenv(SOURCE_TOGGLE_ENV, raising=False)
        first = get_secret("POLYGON_API_KEY")
        assert first == "poly-secret-from-ssm"
        # Mutate SSM under the cache; the cached read should still win.
        ssm_seeded.put_parameter(
            Name="/alpha-engine/POLYGON_API_KEY",
            Type="SecureString",
            Value="rotated-secret",
            Overwrite=True,
        )
        assert get_secret("POLYGON_API_KEY") == "poly-secret-from-ssm"

    def test_clear_cache_re_reads(self, monkeypatch, ssm_seeded):
        monkeypatch.delenv(SOURCE_TOGGLE_ENV, raising=False)
        assert get_secret("POLYGON_API_KEY") == "poly-secret-from-ssm"
        ssm_seeded.put_parameter(
            Name="/alpha-engine/POLYGON_API_KEY",
            Type="SecureString",
            Value="rotated-secret",
            Overwrite=True,
        )
        clear_cache()
        assert get_secret("POLYGON_API_KEY") == "rotated-secret"


# ── SSM-unavailable latching ────────────────────────────────────────────────


class TestSsmUnavailable:
    def test_boto3_missing_falls_through_to_env(self, monkeypatch):
        """When boto3 import fails, fall through to env without raising."""
        monkeypatch.delenv(SOURCE_TOGGLE_ENV, raising=False)
        monkeypatch.setenv("LOCAL_KEY", "from-env-no-boto")
        import builtins

        real_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            if name == "boto3":
                raise ImportError("boto3 not installed (simulated)")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=blocked_import):
            assert get_secret("LOCAL_KEY") == "from-env-no-boto"
        # Latch should be set — subsequent calls skip SSM entirely.
        assert secrets_mod._ssm_unavailable is True

    def test_endpoint_error_latches_unavailable(self, monkeypatch):
        """A botocore network error latches _ssm_unavailable for the process."""
        monkeypatch.delenv(SOURCE_TOGGLE_ENV, raising=False)
        monkeypatch.setenv("LOCAL_KEY", "from-env-after-network-fail")

        def boom(**kwargs):
            raise EndpointConnectionError(endpoint_url="https://ssm.us-east-1.amazonaws.com")

        with patch("boto3.client") as mock_client:
            mock_client.return_value.get_parameter.side_effect = boom
            assert get_secret("LOCAL_KEY") == "from-env-after-network-fail"
        assert secrets_mod._ssm_unavailable is True

    def test_parameter_not_found_does_not_latch(self, monkeypatch, ssm_seeded):
        """ParameterNotFound is a per-key miss, not a global SSM failure —
        must not latch the unavailable flag (other secrets still resolve)."""
        monkeypatch.delenv(SOURCE_TOGGLE_ENV, raising=False)
        monkeypatch.setenv("MISSING_KEY", "from-env")
        # First call: SSM misses on MISSING_KEY → env fallback wins.
        assert get_secret("MISSING_KEY") == "from-env"
        # Latch should NOT be set — SSM was healthy, just empty.
        assert secrets_mod._ssm_unavailable is False
        # Second call to a different key should still try SSM and succeed.
        assert get_secret("POLYGON_API_KEY") == "poly-secret-from-ssm"


# ── Region resolution ──────────────────────────────────────────────────────


class TestRegion:
    def test_aws_region_env_var_used(self, monkeypatch, ssm_seeded):
        monkeypatch.delenv(SOURCE_TOGGLE_ENV, raising=False)
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        assert get_secret("POLYGON_API_KEY") == "poly-secret-from-ssm"

    def test_aws_default_region_fallback(self, monkeypatch, ssm_seeded):
        """When AWS_REGION is unset, fall back to AWS_DEFAULT_REGION."""
        monkeypatch.delenv(SOURCE_TOGGLE_ENV, raising=False)
        monkeypatch.delenv("AWS_REGION", raising=False)
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
        assert get_secret("POLYGON_API_KEY") == "poly-secret-from-ssm"
