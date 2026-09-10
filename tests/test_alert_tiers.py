"""Delivery-tier resolution — alpha-engine-config-I6751 Phase 1.

These are CONTRACT tests, not value tests: none of them assert that a named
fleet class carries a named tier (that is the registry's job and it changes on
merge). They assert the ROUTING RULES — which are what regressed for a year
while every "severity tiering fix" left SNS byte-identical at every severity.
"""

from __future__ import annotations

import json

import pytest

from krepis import alert_tiers


def _registry(tmp_path, monkeypatch, entries, *, schema_version=1):
    doc = {
        "schema_version": schema_version,
        "generated_at": "2026-09-09T00:00:00+00:00",
        "source_digest": "sha256:test",
        "colliding_sources": [],
        "entries": entries,
    }
    path = tmp_path / "alert_tier_registry.json"
    path.write_text(json.dumps(doc))
    monkeypatch.setenv(alert_tiers.REGISTRY_PATH_ENV, str(path))
    alert_tiers.reset_cache()
    return path


@pytest.fixture(autouse=True)
def _clean_cache():
    alert_tiers.reset_cache()
    yield
    alert_tiers.reset_cache()


def test_exact_source_resolves_its_declared_tier(tmp_path, monkeypatch):
    _registry(tmp_path, monkeypatch, [
        {"class": "hygiene_thing", "source": "scan-unlisted-state",
         "tier": "tracked-only", "severities": ["warning"]},
    ])
    d = alert_tiers.resolve_tier("scan-unlisted-state", "critical")
    # The severity is CRITICAL and the tier is still tracked-only: severity
    # has stopped deciding delivery. This assertion is the whole epic.
    assert d.tier == alert_tiers.TIER_TRACKED_ONLY
    assert d.alert_class == "hygiene_thing"
    assert d.registry_drift is False


def test_unknown_source_pages_and_is_marked_as_drift(tmp_path, monkeypatch):
    _registry(tmp_path, monkeypatch, [
        {"class": "known", "source": "a", "tier": "tracked-only",
         "severities": ["info"]},
    ])
    d = alert_tiers.resolve_tier("a-source-nobody-declared", "info")
    assert d.tier == alert_tiers.TIER_PAGE
    assert d.registry_drift is True


def test_missing_source_pages(tmp_path, monkeypatch):
    _registry(tmp_path, monkeypatch, [])
    for empty in (None, ""):
        d = alert_tiers.resolve_tier(empty, "info")
        assert d.tier == alert_tiers.TIER_PAGE
        assert d.registry_drift is True


def test_unreadable_registry_fails_upward_to_page(tmp_path, monkeypatch):
    monkeypatch.setenv(alert_tiers.REGISTRY_PATH_ENV, str(tmp_path / "nope.json"))
    alert_tiers.reset_cache()
    d = alert_tiers.resolve_tier("anything", "info")
    assert d.tier == alert_tiers.TIER_PAGE
    assert d.registry_drift is True


def test_unknown_schema_version_is_refused_rather_than_guessed(tmp_path, monkeypatch):
    _registry(tmp_path, monkeypatch, [
        {"class": "x", "source": "s", "tier": "tracked-only",
         "severities": ["info"]},
    ], schema_version=99)
    d = alert_tiers.resolve_tier("s", "info")
    assert d.tier == alert_tiers.TIER_PAGE
    assert d.registry_drift is True


def test_dynamic_row_resolves_from_severity(tmp_path, monkeypatch):
    _registry(tmp_path, monkeypatch, [
        {"class": "varies", "source": "sweep", "tier": "dynamic",
         "severities": ["dynamic"]},
    ])
    assert alert_tiers.resolve_tier("sweep", "critical").tier == alert_tiers.TIER_PAGE
    assert alert_tiers.resolve_tier("sweep", "error").tier == alert_tiers.TIER_NOTIFY_SILENT
    assert alert_tiers.resolve_tier("sweep", "warning").tier == alert_tiers.TIER_TRACKED_ONLY
    # An unrecognised severity on a dynamic row is not a licence to go quiet.
    assert alert_tiers.resolve_tier("sweep", "banana").tier == alert_tiers.TIER_PAGE


def test_wildcard_row_matches_by_prefix_and_exact_wins(tmp_path, monkeypatch):
    _registry(tmp_path, monkeypatch, [
        {"class": "broad", "source": "research:*", "tier": "tracked-only",
         "severities": ["warning"]},
        {"class": "specific", "source": "research:cut_promotion",
         "tier": "page", "severities": ["error"]},
    ])
    assert alert_tiers.resolve_tier("research:anything", "info").alert_class == "broad"
    assert alert_tiers.resolve_tier("research:cut_promotion", "info").alert_class == "specific"


def test_longest_wildcard_wins_over_a_broader_one(tmp_path, monkeypatch):
    _registry(tmp_path, monkeypatch, [
        {"class": "broad", "source": "r:*", "tier": "page",
         "severities": ["error"]},
        {"class": "narrow", "source": "r:sub:*", "tier": "tracked-only",
         "severities": ["warning"]},
    ])
    assert alert_tiers.resolve_tier("r:sub:thing", "info").alert_class == "narrow"


def test_colliding_rows_take_the_strictest_tier(tmp_path, monkeypatch):
    # `metron` is knowingly shared by two classes (alpha-engine-config-I8995).
    # A resolver that picked arbitrarily could silence a real page.
    _registry(tmp_path, monkeypatch, [
        {"class": "quiet", "source": "metron", "tier": "tracked-only",
         "severities": ["warning"]},
        {"class": "loud", "source": "metron", "tier": "page",
         "severities": ["error"]},
    ])
    d = alert_tiers.resolve_tier("metron", "error")
    assert d.tier == alert_tiers.TIER_PAGE
    assert "strictest" in d.reason


def test_unknown_tier_string_in_the_registry_pages(tmp_path, monkeypatch):
    _registry(tmp_path, monkeypatch, [
        {"class": "typo", "source": "s", "tier": "traked-only",
         "severities": ["info"]},
    ])
    assert alert_tiers.resolve_tier("s", "info").tier == alert_tiers.TIER_PAGE


def test_page_after_consecutive_is_carried_through(tmp_path, monkeypatch):
    _registry(tmp_path, monkeypatch, [
        {"class": "hourly", "source": "box-health", "tier": "page",
         "severities": ["critical"], "page_after_consecutive": 2},
    ])
    assert alert_tiers.resolve_tier("box-health", "critical").page_after_consecutive == 2


def test_muted_topic_arn_rewrites_the_fleet_default_only():
    from krepis import alerts

    assert alerts._muted_topic_arn(
        "arn:aws:sns:us-east-1:711398986525:alpha-engine-alerts"
    ) == f"arn:aws:sns:us-east-1:711398986525:{alert_tiers.MUTED_SNS_TOPIC_NAME}"
    # A topic with no declared muted sibling keeps its email leg rather than
    # having its durable record published into a topic the caller's role may
    # not be granted. `crucible-v2-pages` is the live example, and crucible-v2
    # phase 2's page-counting clauses must keep observing it unchanged through
    # 2026-09-19.
    assert alerts._muted_topic_arn(
        "arn:aws:sns:us-east-1:711398986525:crucible-v2-pages"
    ) is None
    assert alerts._muted_topic_arn(None) is None


def test_episode_override_can_retier_one_episode_of_a_class(tmp_path, monkeypatch):
    """A class that is legitimately two conditions sharing one source.

    Measured cases: the EOD reconciler's scheduled vendor substitution vs a
    genuinely bad provisional print (alpha-engine-config-I10360), and a
    crucible-v2 fault-injection replay vs a live failure
    (alpha-engine-config-I10366). The MECHANISM ships now; no registry row
    declares an override until each is ruled.
    """
    _registry(tmp_path, monkeypatch, [
        {"class": "recon", "source": "eod-recon", "tier": "page",
         "severities": ["warning"],
         "episode_overrides": [
             {"when": {"close_source": "substitution"}, "tier": "tracked-only"},
         ]},
    ])
    assert alert_tiers.resolve_tier("eod-recon", "warning").tier == alert_tiers.TIER_PAGE
    held = alert_tiers.resolve_tier(
        "eod-recon", "warning", {"close_source": "substitution"},
    )
    assert held.tier == alert_tiers.TIER_TRACKED_ONLY
    assert "episode override" in held.reason
    # A partial attribute match must NOT retier — an override is an exact
    # claim about an episode, not a fuzzy one.
    assert alert_tiers.resolve_tier(
        "eod-recon", "warning", {"close_source": "yfinance"},
    ).tier == alert_tiers.TIER_PAGE
