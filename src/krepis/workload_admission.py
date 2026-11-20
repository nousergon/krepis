"""Opt-in admission for subscription-backed, content-only router execution.

This is a pure in-process contract, not an authenticator or dispatcher. Existing
router consumers are unchanged. The server must supply a trusted profile bound
to its authenticated identity and candidate metadata from the shared registry.
It must derive request size, classification and trigger from trusted intake,
not accept caller claims that downgrade them. Call before *every* attempt,
including fallbacks. Admission does not reserve quota, sandbox tools, authorize
account access, or prove qualification: those remain dispatch prerequisites.
"""
from dataclasses import dataclass
from typing import FrozenSet


class AdmissionRefused(ValueError):
    """A candidate or request does not satisfy the authenticated workload policy."""


def _text(value: object, field: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 512:
        raise ValueError("invalid " + field)


def _members(value: FrozenSet[str], field: str) -> None:
    if type(value) is not frozenset or not value:
        raise ValueError("nonempty immutable " + field + " required")
    for member in value:
        _text(member, field)


def _number(value: int, field: str, minimum: int = 1) -> None:
    if type(value) is not int or not minimum <= value <= 2**63 - 1:
        raise ValueError("invalid " + field)


@dataclass(frozen=True)
class ContentProfile:
    """Server-owned policy; revision must change whenever its contents change."""

    consumer_id: str
    revision: str
    group: str
    adapter_ids: FrozenSet[str]
    processors: FrozenSet[str]
    data_classes: FrozenSet[str]
    triggers: FrozenSet[str]
    max_input_bytes: int
    max_output_bytes: int
    timeout_seconds: int

    def __post_init__(self) -> None:
        for field in ("consumer_id", "revision", "group"):
            _text(getattr(self, field), field)
        for field in ("adapter_ids", "processors", "data_classes", "triggers"):
            _members(getattr(self, field), field)
        for field in ("max_input_bytes", "max_output_bytes", "timeout_seconds"):
            _number(getattr(self, field), field)


@dataclass(frozen=True)
class ContentRequest:
    """Normalized trusted intake; byte limits cover the complete rendered input."""

    request_id: str
    profile_revision: str
    group: str
    data_class: str
    trigger: str
    input_bytes: int
    output_bytes: int
    timeout_seconds: int
    execution_id: str = ""  # Minted durably by dispatch, never by caller intake.

    def __post_init__(self) -> None:
        if self.execution_id != "":
            _text(self.execution_id, "execution_id")
        if self.execution_id and (len(self.execution_id) != 32 or any(c not in "0123456789abcdef" for c in self.execution_id)):
            raise ValueError("invalid execution_id")
        for field in ("request_id", "profile_revision", "group", "data_class", "trigger"):
            _text(getattr(self, field), field)
        _number(self.input_bytes, "input_bytes", minimum=0)
        for field in ("output_bytes", "timeout_seconds"):
            _number(getattr(self, field), field)


@dataclass(frozen=True)
class ContentCandidate:
    """Trusted deployment metadata, including every processor/intermediary.

    qualification_ref is evidence provenance, not evidence validation. The
    dispatch service must establish its validity/freshness before constructing
    this candidate; a nonempty reference alone does not qualify a deployment.
    """

    adapter_id: str
    processors: FrozenSet[str]
    group: str
    execution_class: str
    billing: str
    qualification_ref: str
    registry_id: str = ""
    registry_revision: str = ""

    def __post_init__(self) -> None:
        if self.registry_id or self.registry_revision:
            _text(self.registry_id, "registry_id")
            _text(self.registry_revision, "registry_revision")
            if len(self.registry_revision) != 64 or any(c not in "0123456789abcdef" for c in self.registry_revision):
                raise ValueError("invalid registry revision digest")
        for field in ("adapter_id", "group", "execution_class", "billing", "qualification_ref"):
            _text(getattr(self, field), field)
        _members(self.processors, "processors")


def admit_content(
    profile: ContentProfile,
    request: ContentRequest,
    candidate: ContentCandidate,
    *,
    authenticated_consumer: str,
) -> None:
    """Refuse unsafe selection; does not select a model or perform any effects.

    A successful check is not a reusable permit. Dispatch must recheck current
    policy and deployment metadata after queueing and before each execution.
    """
    if authenticated_consumer != profile.consumer_id:
        raise AdmissionRefused("consumer/profile mismatch")
    if request.profile_revision != profile.revision:
        raise AdmissionRefused("stale profile revision")
    if request.group != profile.group or candidate.group != profile.group:
        raise AdmissionRefused("group mismatch")
    if request.data_class not in profile.data_classes or request.trigger not in profile.triggers:
        raise AdmissionRefused("data class or trigger not permitted")
    if (request.input_bytes > profile.max_input_bytes
            or request.output_bytes > profile.max_output_bytes
            or request.timeout_seconds > profile.timeout_seconds):
        raise AdmissionRefused("request exceeds profile bounds")
    if candidate.adapter_id not in profile.adapter_ids:
        raise AdmissionRefused("adapter not permitted")
    if not candidate.processors <= profile.processors:
        raise AdmissionRefused("processor chain not permitted")
    if candidate.execution_class != "content" or candidate.billing != "subscription":
        raise AdmissionRefused("content-only subscription execution required")
