"""Shared test fixtures for ``krepis``.

``publish`` (``alerts.py``) gained a ``PYTEST_CURRENT_TEST`` guard that
short-circuits real SNS / Telegram fan-out from inside any test process
(the cross-repo chokepoint, L4566). This lib's OWN alert tests, however,
deliberately exercise the real fan-out logic against mocked boto3 /
Telegram transports — so they must opt back in via the
``ALPHA_ENGINE_ALLOW_TEST_ALERTS`` escape hatch. Enable it for the whole
lib suite here; the one test that verifies the guard itself unsets it.
"""
import pytest


@pytest.fixture(autouse=True)
def _allow_real_alert_fanout_under_mocked_transports(monkeypatch):
    monkeypatch.setenv("ALPHA_ENGINE_ALLOW_TEST_ALERTS", "1")
    yield


@pytest.fixture(autouse=True)
def _no_ssm_master_key_lookup_from_tests(monkeypatch):
    """Cut the one router leg that shells out to AWS.

    ``_resolve_litellm_master_key`` falls back to ``aws ssm get-parameter``. On a
    machine with no working credentials that call fails, and six tests in
    ``test_router.py`` were green only because of it — they asserted "no master
    key is resolvable here", which is a fact about the laptop, not about krepis.
    Giving the laptop a non-expiring machine identity (nous-ergon-ops-PR237)
    turned all six red on the same run.

    CI never had credentials, so CI never saw it: the suite would have stayed
    green while being wrong. Neutralising the leg makes the outcome the same
    everywhere. A test that wants the SSM path patches
    ``krepis.router._litellm_master_key_from_ssm`` directly.
    """
    monkeypatch.setattr(
        "krepis.router._litellm_master_key_from_ssm", lambda: None, raising=True
    )
    yield
