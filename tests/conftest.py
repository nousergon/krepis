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
