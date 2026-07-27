"""Tests for ``krepis.cache_minimums``.

The values themselves are the point. A wrong minimum fails silently in
whichever direction it is wrong — too low places markers that never cache,
too high declines caching that would have worked — so the table is pinned
against the published values rather than merely exercised.
"""
from __future__ import annotations

import pytest

from krepis.cache_minimums import (
    CacheMinimumLookupError,
    cache_minimum,
    clears_cache_minimum,
    known_models,
    require_cache_minimum,
)


class TestPublishedValues:
    """Pin the table. These are provider facts with a doc link in the YAML."""

    @pytest.mark.parametrize(
        "model,expected",
        [
            ("claude-fable-5", 512),
            ("claude-opus-5", 512),
            ("claude-opus-4-8", 1024),
            ("claude-sonnet-5", 1024),
            ("claude-sonnet-4-6", 1024),
            ("claude-opus-4-7", 2048),
            ("claude-opus-4-6", 4096),
            ("claude-haiku-4-5", 4096),
        ],
    )
    def test_value(self, model, expected):
        assert cache_minimum(model) == expected

    def test_minimums_are_not_monotonic_across_generations(self):
        """Guards the trap directly.

        If someone 'tidies' this table on the assumption that newer means
        lower, or that Haiku is cheapest therefore smallest, this fails.
        """
        assert cache_minimum("claude-opus-5") < cache_minimum("claude-opus-4-7")
        assert cache_minimum("claude-opus-4-7") < cache_minimum("claude-opus-4-6")
        # Haiku — the mechanical tier — carries the HIGHEST minimum.
        assert cache_minimum("claude-haiku-4-5") == max(
            cache_minimum(m) for m in known_models()
        )

    def test_regression_the_two_values_that_were_wrong_in_the_wild(self):
        """crucible-research/scripts/measure_cache_prefixes.py carried
        opus-4-7 as 4096 and sonnet-4-6 as 2048 — both too high, which
        suppressed caching rollout on three call sites."""
        assert cache_minimum("claude-opus-4-7") == 2048, "was wrongly 4096"
        assert cache_minimum("claude-sonnet-4-6") == 1024, "was wrongly 2048"


class TestResolution:
    def test_dated_snapshot_resolves_to_its_family(self):
        assert cache_minimum("claude-sonnet-4-6-20250514") == 1024
        assert cache_minimum("claude-haiku-4-5-20251001") == 4096

    def test_longest_prefix_wins_not_shortest(self):
        """``claude-opus-4-5`` and ``claude-opus-4-8`` share a prefix; a
        shorter family key must never swallow a more specific one."""
        assert cache_minimum("claude-opus-4-8") == 1024
        assert cache_minimum("claude-opus-4-5") == 4096

    def test_unknown_model_is_none_not_zero(self):
        assert cache_minimum("deepseek-v4-flash") is None
        assert cache_minimum("some-model-that-does-not-exist") is None

    def test_empty_and_none_are_none(self):
        assert cache_minimum(None) is None
        assert cache_minimum("") is None


class TestRequire:
    def test_returns_value_when_known(self):
        assert require_cache_minimum("claude-opus-5") == 512

    def test_raises_on_unknown_rather_than_defaulting(self):
        with pytest.raises(CacheMinimumLookupError) as exc:
            require_cache_minimum("deepseek-v4-pro")
        # The message has to tell the caller what to do about it.
        assert "cache_minimums.yaml" in str(exc.value)


class TestClears:
    def test_at_the_boundary_is_inclusive(self):
        assert clears_cache_minimum("claude-opus-5", 512) is True
        assert clears_cache_minimum("claude-opus-5", 511) is False

    def test_haiku_rejects_a_prefix_opus_would_cache(self):
        """The concrete cost of assuming tier predicts the minimum."""
        assert clears_cache_minimum("claude-opus-5", 3000) is True
        assert clears_cache_minimum("claude-haiku-4-5", 3000) is False

    def test_unknown_model_is_none_not_a_default(self):
        """Tri-state on purpose: neither True nor False is safe to assume,
        and both would fail invisibly."""
        assert clears_cache_minimum("deepseek-v4-flash", 10_000) is None


class TestTable:
    def test_known_models_is_longest_first(self):
        lengths = [len(m) for m in known_models()]
        assert lengths == sorted(lengths, reverse=True)

    def test_every_value_is_a_positive_int(self):
        assert all(
            isinstance(cache_minimum(m), int) and cache_minimum(m) > 0
            for m in known_models()
        )
