"""Native content service binding core, with no listener or credential issuer.

Transport calls ``submit`` only through its authenticator. Authentication,
profile loading, qualification evidence and adapter execution are trusted
injected infrastructure, never request fields. The authenticator must reject
invalid/revoked credentials and return a stable consumer identity. Binding and
registry loaders must return current server-owned snapshots; qualifier must
validate freshness and the exact route/model/deployment/registry digest.

Execution remains isolated by the operated worker. Callbacks before execution
must be bounded and effect-free; execute must enforce the deadline, quota,
cancellation and tool isolation. This core never makes a fallback attempt once
an execution is claimed. It neither grants account access nor qualifies a host.
"""
from dataclasses import dataclass
import hashlib
import json
import time
from typing import Callable, Optional, cast

from .content_dispatch import ContentResult, DispatchJournal
from .model_registry import Registry
from .workload_admission import (
    AdmissionRefused, ContentCandidate, ContentProfile, ContentRequest, admit_content,
)


class ServiceRefused(ValueError):
    """Authenticated workload cannot be served under current trusted policy."""


def _text(value: object) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 512:
        raise ServiceRefused("bounded nonempty service identity required")
    return value


@dataclass(frozen=True)
class WorkloadBinding:
    profile: ContentProfile
    data_class: str
    trigger: str

    def __post_init__(self) -> None:
        if self.data_class not in self.profile.data_classes or self.trigger not in self.profile.triggers:
            raise ServiceRefused("binding classification/trigger not permitted by profile")


@dataclass(frozen=True)
class NativeRoute:
    candidate: ContentCandidate
    model: str
    deployment_ref: str
    served_models: frozenset[str]


def _registry_revision(registry: Registry) -> str:
    # Canonical data, not a file timestamp or caller-supplied revision. Any model
    # or group change invalidates the previous qualification and replay binding.
    raw = json.dumps(registry.doc, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _route(registry: Registry, group: str, mid: str, revision: str) -> NativeRoute:
    entry = registry.models[mid]
    value: object = entry.get("native")
    if not isinstance(value, dict):
        raise ServiceRefused("versioned native metadata required")
    native = cast(dict[str, object], value)
    fields = {"schema_version", "adapter_id", "processors", "execution_class", "billing",
              "qualification_ref", "deployment_ref", "served_models"}
    if set(native) != fields or type(native["schema_version"]) is not int or native["schema_version"] != 1:
        raise ServiceRefused("unsupported native metadata schema")
    processors = native["processors"]
    if not isinstance(processors, list) or not processors:
        raise ServiceRefused("nonempty native processor list required")
    members = [_text(item) for item in cast(list[object], processors)]
    if len(set(members)) != len(members):
        raise ServiceRefused("duplicate native processor")
    served = native["served_models"]
    if not isinstance(served, list) or not served:
        raise ServiceRefused("qualified served model identities required")
    served_members = [_text(item) for item in cast(list[object], served)]
    if len(set(served_members)) != len(served_members):
        raise ServiceRefused("duplicate served model identity")
    return NativeRoute(ContentCandidate(
        _text(native["adapter_id"]), frozenset(members), group,
        _text(native["execution_class"]), _text(native["billing"]),
        _text(native["qualification_ref"]), registry_id=mid, registry_revision=revision,
    ), _text(entry.get("model")), _text(native["deployment_ref"]), frozenset(served_members))


class ContentService:
    """Per-worker binding core sharing the durable dispatch journal's capacity.

    Only request identity and content are caller controlled. A binding fixes the
    profile, group, classification, trigger and all bounds. Separate authenticated
    bindings are needed for distinct trust levels or trigger classes.
    """

    def __init__(self, journal: DispatchJournal, *,
                 authenticate: Callable[[object], str],
                 load_binding: Callable[[str], WorkloadBinding],
                 load_registry: Callable[[], Registry],
                 qualified: Callable[[NativeRoute], bool],
                 execute: Callable[[NativeRoute, ContentRequest, str], ContentResult]) -> None:
        self.journal = journal
        journal.db.execute("""CREATE TABLE IF NOT EXISTS content_service_events (
            event_id INTEGER PRIMARY KEY, created_at REAL NOT NULL,
            consumer TEXT, request_id TEXT, registry_revision TEXT,
            outcome TEXT NOT NULL, stage TEXT NOT NULL, error_type TEXT, rejections TEXT)""")
        self.authenticate = authenticate
        self.load_binding = load_binding
        self.load_registry = load_registry
        self.qualified = qualified
        self.execute = execute

    def _identity(self, credentials: object) -> tuple[str, WorkloadBinding]:
        principal = _text(self.authenticate(credentials))
        binding = self.load_binding(principal)
        if binding.profile.consumer_id != principal:
            raise ServiceRefused("authenticated identity/profile mismatch")
        return principal, binding

    def _event(self, event_id: int, **values: str) -> None:
        # Keys are internal constants; never interpolate caller fields into SQL.
        permitted = {"consumer", "request_id", "registry_revision", "outcome", "stage", "error_type", "rejections"}
        if not values or not set(values) <= permitted:
            raise ValueError("invalid service event fields")
        self.journal.verify_durability()
        cursor = self.journal.db.execute(
            "UPDATE content_service_events SET " + ",".join(key + "=?" for key in values)
            + " WHERE event_id=?", (*values.values(), event_id))
        if cursor.rowcount != 1:
            raise ServiceRefused("service event missing")

    def submit(self, credentials: object, *, request_id: str, content: str) -> ContentResult:
        self.journal.verify_durability()
        cursor = self.journal.db.execute(
            "INSERT INTO content_service_events(created_at,outcome,stage) VALUES(?,'started','authentication')",
            (time.time(),))
        event_id = cursor.lastrowid
        if event_id is None:
            raise ServiceRefused("service event not recorded")
        try:
            result = self._submit(credentials, request_id=request_id, content=content, event_id=event_id)
        except BaseException as exc:
            # Never store exception text: authenticators and adapters may include
            # secrets. A durable stage + exception class describes the failure.
            row = self.journal.db.execute(
                "SELECT stage FROM content_service_events WHERE event_id=?", (event_id,)).fetchone()
            self._event(event_id, outcome="uncertain" if row and row[0] == "execution" else "refused",
                        error_type=type(exc).__name__[:128])
            raise
        self._event(event_id, outcome="completed", stage="completed")
        return result

    def _submit(self, credentials: object, *, request_id: str, content: str, event_id: int) -> ContentResult:
        principal, binding = self._identity(credentials)
        if type(content) is not str:
            raise ServiceRefused("content must be text")
        self._event(event_id, stage="selection", consumer=principal, request_id=_text(request_id))
        profile = binding.profile
        request = ContentRequest(request_id, profile.revision, profile.group,
                                 binding.data_class, binding.trigger, len(content.encode("utf-8")),
                                 profile.max_output_bytes, profile.timeout_seconds)
        registry = self.load_registry()
        revision = _registry_revision(registry)
        self._event(event_id, stage="selection", registry_revision=revision)
        rejected: list[tuple[str, str]] = []
        for mid in registry.live_group_ids(profile.group, transport="native"):
            route = _route(registry, profile.group, mid, revision)
            try:
                admit_content(profile, request, route.candidate, authenticated_consumer=principal)
            except AdmissionRefused as exc:
                # No external effect occurred. The final refusal below is explicit;
                # policy never widens to make a rejected member eligible.
                rejected.append((mid, str(exc)))
                self._event(event_id, rejections=json.dumps(rejected))
                continue
            self._event(event_id, stage="qualification")
            if self.qualified(route) is not True:
                rejected.append((mid, "qualification unavailable"))
                self._event(event_id, rejections=json.dumps(rejected))
                continue

            self._event(event_id, stage="preflight")

            def preflight() -> None:
                # Runs under the capacity lock, before the execution is claimed.
                # Local-only checks; refusal has no external effect to reconcile.
                current_principal, current_binding = self._identity(credentials)
                if (current_principal != principal or current_binding != binding
                        or _registry_revision(self.load_registry()) != revision):
                    raise ServiceRefused("service binding changed before execution")
                admit_content(current_binding.profile, request, route.candidate,
                              authenticated_consumer=current_principal)
                if self.qualified(route) is not True:
                    raise ServiceRefused("route qualification unavailable before execution")
            def invoke(claimed: ContentRequest, candidate: ContentCandidate, text: str) -> ContentResult:
                self._event(event_id, stage="execution")
                result = self.execute(route, claimed, text)
                if result.served_model not in route.served_models:
                    raise ServiceRefused("observed served model is outside qualified route")
                return result

            return self.journal.dispatch(profile, request, route.candidate,
                                         authenticated_consumer=principal, content=content, adapter=invoke, preflight=preflight)
        raise ServiceRefused("no permitted qualified native content route")

    def status(self, credentials: object, request_id: str) -> Optional[str]:
        principal, _binding = self._identity(credentials)
        return self.journal.status(principal, _text(request_id))

    def result(self, credentials: object, request_id: str) -> Optional[ContentResult]:
        principal, _binding = self._identity(credentials)
        return self.journal.result(principal, _text(request_id))
