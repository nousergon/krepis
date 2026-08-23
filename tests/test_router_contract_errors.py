"""A 4xx refusal must fail fast; only availability may consume a fallback.

alpha-engine-config-I7904. Two things are asserted here that nothing asserted
before:

* the **classification** — which failures are permanent contract errors and
  which stay in the availability class, including the 4xx codes that are
  availability after all;
* the **negative control** — a champion that returns a real ``400`` fails with
  ITS OWN message and the fallback chain is never entered, while a champion
  that returns a ``503`` still fails over exactly as it always did.

A classification that has never been observed classifying is unproven, so the
negative control drives the real ``litellm.Router`` object krepis builds, at
the real seam, and spies on litellm's own fallback entry point.
"""

from __future__ import annotations

import asyncio

import pytest

from krepis.llm_errors import (
    PermanentContractError,
    as_permanent_contract_error,
    http_status_of,
    is_permanent_contract_error,
    raise_if_permanent_contract_error,
    split_fallback_noise,
)

#: The message litellm actually surfaced, verbatim, on the run that produced
#: alpha-engine-config-I7904. Kept as a literal: the whole defect is that the
#: LAST clause is the one a reader takes away, and a paraphrase would lose the
#: shape that makes it misleading.
I7904_SURFACED = (
    "litellm.BadRequestError: OpenAIException - Thinking mode does not support "
    "this tool_choice.\n"
    "Received Model Group=low-deepseek-v4-flash-low\n"
    "Available Model Group Fallbacks=['glm-4.7-flash']\n"
    "Error doing the fallback: litellm.RateLimitError - Rate limit reached for "
    "requests\n"
    "No fallback model group found for original model_group=glm-4.7-flash. "
    "Fallbacks=None"
)


class _StatusError(Exception):
    def __init__(self, status: int, message: str = "boom") -> None:
        super().__init__(message)
        self.status_code = status


class TestClassification:
    @pytest.mark.parametrize("status", [400, 401, 403, 404, 413, 422])
    def test_a_request_refusal_is_permanent(self, status):
        assert is_permanent_contract_error(_StatusError(status)) is True

    @pytest.mark.parametrize("status", [408, 409, 425, 429, 500, 502, 503, 529])
    def test_availability_stays_availability(self, status):
        """The 4xx codes in this list describe a transient SERVER condition.

        Classifying 429 as permanent would silently delete rate-limit failover,
        which is the one thing fallback chains are unambiguously for.
        """
        assert is_permanent_contract_error(_StatusError(status)) is False

    def test_an_unclassifiable_failure_keeps_its_retries(self):
        """Fail-open. "We could not tell" must never suppress a failover."""
        assert http_status_of(RuntimeError("connection reset")) is None
        assert is_permanent_contract_error(RuntimeError("connection reset")) is False

    def test_status_is_read_from_the_sdk_message_when_no_attribute_carries_it(self):
        assert http_status_of(RuntimeError("Error code: 400 - {'error': ...}")) == 400

    def test_a_three_digit_number_that_is_not_a_status_is_not_read_as_one(self):
        assert http_status_of(RuntimeError("model deepseek-v4-flash-0731 rejected")) is None


class TestTheSurfacedCauseIsTheChampionsOwn:
    def test_the_fallback_noise_is_split_off_the_cause(self):
        cause, aside = split_fallback_noise(I7904_SURFACED)
        assert cause == (
            "litellm.BadRequestError: OpenAIException - Thinking mode does not "
            "support this tool_choice."
        )
        assert "RateLimitError" in aside

    def test_the_error_leads_with_the_cause_and_labels_the_rest_not_the_cause(self):
        exc = as_permanent_contract_error(
            _StatusError(400, I7904_SURFACED),
            deployment="low-deepseek-v4-flash-low",
            model_group="low",
        )
        text = str(exc)
        # The cause appears before any mention of the fallback's own failure.
        assert text.index("Thinking mode does not support") < text.index("RateLimitError")
        assert "NOT THE CAUSE" in text
        assert exc.status_code == 400
        assert exc.deployment == "low-deepseek-v4-flash-low"

    def test_raise_if_permanent_is_a_no_op_on_an_availability_failure(self):
        raise_if_permanent_contract_error(_StatusError(503))  # must not raise

    def test_raise_if_permanent_raises_on_a_refusal(self):
        with pytest.raises(PermanentContractError):
            raise_if_permanent_contract_error(_StatusError(400, "bad tool_choice"))


# ── Negative control: drive the real Router at the real seam ─────────────

def _router_with_a_chain():
    """A two-deployment Router built by krepis, with a declared fallback."""
    from krepis.router import _contract_aware_router_class

    cls = _contract_aware_router_class()
    return cls(
        model_list=[
            {
                "model_name": "champion",
                "litellm_params": {"model": "openai/x", "api_key": "k", "api_base": "http://127.0.0.1:1"},
            },
            {
                "model_name": "backup",
                "litellm_params": {"model": "openai/y", "api_key": "k", "api_base": "http://127.0.0.1:1"},
            },
        ],
        fallbacks=[{"champion": ["backup"]}],
    )


def _drive(router, exc, monkeypatch):
    """Run one completion whose champion raises *exc*; report what happened.

    Returns ``(raised, fallback_entered)``.
    """
    import litellm.router as _lr

    entered: list[str] = []

    async def _boom(*args, **kwargs):
        raise exc

    async def _spy_fallback(*args, **kwargs):
        entered.append(kwargs.get("fallback_model_group") or "?")
        return "FELL-BACK"

    monkeypatch.setattr(router, "async_function_with_retries", _boom)
    monkeypatch.setattr(_lr, "run_async_fallback", _spy_fallback)

    try:
        result = asyncio.run(
            router.async_function_with_fallbacks(
                model="champion", messages=[{"role": "user", "content": "hi"}]
            )
        )
        return result, entered
    except BaseException as raised:  # noqa: BLE001 — the outcome under test
        return raised, entered


class TestTheNegativeControl:
    def test_the_override_is_installed_on_the_object_that_serves_calls(self):
        """The seam is a litellm internal name. If a release renames it, this
        goes red — which is the loud failure. A silently restored
        fallback-on-400 would be the quiet one."""
        router = _router_with_a_chain()
        assert getattr(router, "krepis_contract_aware", False) is True
        assert (
            type(router).async_function_with_fallbacks_common_utils
            is not __import__("litellm").Router.async_function_with_fallbacks_common_utils
        )

    def test_a_400_from_the_champion_never_reaches_the_fallback(self, monkeypatch):
        router = _router_with_a_chain()
        outcome, entered = _drive(
            router, _StatusError(400, "Thinking mode does not support this tool_choice"), monkeypatch
        )
        assert isinstance(outcome, PermanentContractError), outcome
        assert entered == [], "a permanent contract error consumed a fallback"
        assert "Thinking mode does not support this tool_choice" in str(outcome)

    def test_a_503_from_the_champion_still_consumes_the_fallback(self, monkeypatch):
        """The positive control. The fix must narrow the fallback layer, not
        disable it — a change that made every failure fail fast would pass the
        test above and delete the fleet's whole degradation story."""
        router = _router_with_a_chain()
        outcome, entered = _drive(router, _StatusError(503, "upstream overloaded"), monkeypatch)
        assert outcome == "FELL-BACK", outcome
        assert entered == [["backup"]]

    def test_a_429_from_the_champion_still_consumes_the_fallback(self, monkeypatch):
        router = _router_with_a_chain()
        outcome, entered = _drive(router, _StatusError(429, "rate limited"), monkeypatch)
        assert outcome == "FELL-BACK", outcome
        assert entered == [["backup"]]
