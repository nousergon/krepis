"""Is this failure PERMANENT, or is the provider merely unwell?

Every degradation control in the fleet — the router's fallback chains, the
SDK's own retries, and every bounded retry loop a consumer writes around a
model call — is built for ONE failure class: the provider is down, throttled,
or slow. Aimed at that class they are correct. Aimed at a **contract** error
they are worse than useless: they burn the backup on a request that could
never have succeeded, and then report the backup's failure as the cause.

Measured, alpha-engine-config-I7904 (2026-08-21, `Judge Perturbation Smoke`)::

    litellm.BadRequestError: OpenAIException - Thinking mode does not support
      this tool_choice.  Received Model Group=low-deepseek-v4-flash-low
      Available Model Group Fallbacks=['glm-4.7-flash']
    Error doing the fallback: litellm.RateLimitError - Rate limit reached
      No fallback model group found for original model_group=glm-4.7-flash.

Three attempts, an identical 400 each time. The word a reader takes away from
that text is ``RateLimitError``, and it names the wrong model, the wrong
provider and the wrong system. The cause is the first line and nothing else.

Two rules follow, and this module exists so both are decided in ONE place
rather than re-derived at each retry loop:

1. **A permanent contract error is never retried and never failed over.** No
   number of attempts against any deployment changes a rejected request shape.
2. **The surfaced cause is always the CHAMPION's own message.** Whatever a
   fallback went on to say is demoted to a labelled aside that states, in
   words, that it is not the cause.

Calibration — which 4xx are availability after all
--------------------------------------------------
``408`` (request timeout), ``409`` (conflict), ``425`` (too early) and ``429``
(rate limited) are 4xx codes that describe a TRANSIENT server condition, not a
malformed request; they stay in the availability class, which is what keeps
ordinary rate-limit failover working. ``529`` is Anthropic's overloaded signal
and is likewise availability. Everything else in 400–499 is the caller's
request being refused on its merits, and refused identically forever.

Substitutability (principles.md §2.8): the classification is by HTTP status
and exception type only. No provider name, model id, base URL or vendor error
string appears here — a new provider behind the same router is classified by
the same rules on the day it is added.
"""

from __future__ import annotations

import re
from typing import Any, Optional

#: 4xx statuses that describe a transient server condition rather than a
#: rejected request shape. These stay in the AVAILABILITY class, so ordinary
#: rate-limit and timeout failover is untouched by this module.
TRANSIENT_4XX_STATUSES = frozenset({408, 409, 425, 429})

#: Non-4xx statuses that are unambiguously availability (kept explicit so a
#: reader does not have to infer the complement of the rule above).
OVERLOADED_STATUSES = frozenset({529})

#: Last-resort status extraction. Providers and gateways that wrap an upstream
#: error frequently keep the code only in the message text
#: (``Error code: 400``, ``OpenAIException - ... 400``). Anchored on the
#: literal phrase the OpenAI/litellm stack emits rather than any three digits,
#: so a model id or a token count is never read as a status.
_STATUS_IN_TEXT_RE = re.compile(r"[Ee]rror code:\s*(\d{3})\b")

#: litellm appends the fallback's own failure to the ORIGINAL exception's
#: message (``router.py::async_function_with_fallbacks_common_utils``). This
#: splits the two apart so the champion's message can be surfaced alone.
_FALLBACK_NOISE_RE = re.compile(
    r"\n?(?:Received Model Group=|Error doing the fallback:|"
    r"Available Model Group Fallbacks=).*",
    re.DOTALL,
)


class PermanentContractError(RuntimeError):
    """A request that was REFUSED on its merits and will be refused again.

    Raised in place of the provider/gateway exception so that no retry loop
    downstream has to re-classify it, and so the message a human reads leads
    with the champion's own words.
    """

    def __init__(
        self,
        cause_message: str,
        *,
        status_code: Optional[int],
        deployment: Optional[str] = None,
        model_group: Optional[str] = None,
        suppressed_fallback_error: str = "",
    ) -> None:
        self.cause_message = cause_message
        self.status_code = status_code
        self.deployment = deployment
        self.model_group = model_group
        self.suppressed_fallback_error = suppressed_fallback_error

        where = deployment or model_group or "the addressed deployment"
        message = (
            f"permanent_contract_error: {where} REFUSED this request "
            f"(HTTP {status_code if status_code is not None else '4xx'}). "
            f"Cause, verbatim from the model that rejected it: {cause_message}"
        )
        if model_group and deployment and model_group != deployment:
            message += f" [model_group={model_group}]"
        message += (
            ". This is a request-shape defect, not availability: retrying it "
            "and failing it over to another deployment cannot succeed, so "
            "neither was done."
        )
        if suppressed_fallback_error:
            message += (
                "\n\nNOT THE CAUSE — a fallback was attempted before this "
                "classification could stop it, and reported: "
                f"{suppressed_fallback_error}"
            )
        super().__init__(message)


def http_status_of(exc: BaseException) -> Optional[int]:
    """Best available HTTP status for *exc*, or None.

    Order: the exception's own ``status_code`` (litellm and the OpenAI SDK both
    set it), then a ``response.status_code``, then the literal ``Error code: N``
    the OpenAI SDK writes into the message. Returns None when no status can be
    established — and a None status is NEVER classified as permanent, because
    "we could not tell" must not be allowed to suppress a legitimate failover.
    """
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and 100 <= value < 600:
            return value
        if isinstance(value, str) and value.isdigit() and 100 <= int(value) < 600:
            return int(value)

    response: Any = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    if isinstance(value, int) and 100 <= value < 600:
        return value

    match = _STATUS_IN_TEXT_RE.search(str(exc))
    if match:
        return int(match.group(1))
    return None


def is_permanent_contract_error(exc: BaseException) -> bool:
    """True when *exc* is a 4xx refusal of the request itself.

    Fail-open by design: an exception whose status cannot be established is
    classified as AVAILABILITY, so an unrecognised failure keeps the existing
    retry-and-failover behaviour rather than silently losing it.
    """
    if isinstance(exc, PermanentContractError):
        return True
    status = http_status_of(exc)
    if status is None:
        return False
    if status in OVERLOADED_STATUSES or status in TRANSIENT_4XX_STATUSES:
        return False
    return 400 <= status < 500


def split_fallback_noise(message: str) -> tuple[str, str]:
    """Split a litellm-decorated message into ``(cause, fallback_aside)``.

    litellm mutates the ORIGINAL exception's message in place, appending the
    fallback chain it tried and whatever that chain raised. Both halves are
    worth keeping; only the first half is the cause.
    """
    match = _FALLBACK_NOISE_RE.search(message)
    if not match:
        return message.strip(), ""
    return message[: match.start()].strip(), message[match.start():].strip()


def as_permanent_contract_error(
    exc: BaseException,
    *,
    deployment: Optional[str] = None,
    model_group: Optional[str] = None,
) -> PermanentContractError:
    """Build the :class:`PermanentContractError` that replaces *exc*."""
    if isinstance(exc, PermanentContractError):
        return exc
    cause, aside = split_fallback_noise(str(exc))
    return PermanentContractError(
        cause or type(exc).__name__,
        status_code=http_status_of(exc),
        deployment=deployment,
        model_group=model_group,
        suppressed_fallback_error=aside,
    )


def raise_if_permanent_contract_error(
    exc: BaseException,
    *,
    deployment: Optional[str] = None,
    model_group: Optional[str] = None,
) -> None:
    """Re-raise *exc* as a :class:`PermanentContractError` when it is one.

    Returns normally — leaving the caller to handle *exc* as it always has —
    when the failure is availability. That asymmetry is the point: this
    function may only ever REMOVE a retry, never add one.
    """
    if is_permanent_contract_error(exc):
        raise as_permanent_contract_error(
            exc, deployment=deployment, model_group=model_group
        ) from exc
