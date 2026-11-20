"""A6 native harness adapter for an already resolved, qualified content route.

No selection, registry, credential issuance or fallback lives here. Operated
configuration supplies the binary and isolated worker homes, allowed egress,
model and qualification reference. The deployment must pin/verify binary and
model tool metadata, compel network egress and bound OS resources. CLI flags
are defense in depth, not host isolation. Re-qualify on any binary/model change.

The CLI does not expose actual served-model identity in JSON events. Success
therefore requires independent transport evidence matched to the request and
adapter; the requested model is never reported as an observed served model.
Wrap calls in content_dispatch to preserve no-replay semantics on all failures.
"""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import tempfile
import time
from typing import Callable, Optional, cast
from urllib.parse import urlsplit

from .content_dispatch import ContentResult
from .workload_admission import ContentCandidate, ContentRequest

_DISABLED_FEATURES = (
    "shell_tool", "apps", "plugins", "multi_agent", "view_image",
    "computer_use", "browser_use", "image_generation", "goals", "sleep_tool",
    "workspace_dependencies", "skill_search", "code_mode_host", "shell_snapshot",
    "hooks", "unbounded_connection_retries",
)
_MAX_CAPTURE_BYTES = 65_536


@dataclass(frozen=True)
class CodexSpec:
    binary: Path
    home: Path
    credential_home: Path
    workdir: Path
    model: str
    egress_url: str
    upstream_host: str
    adapter_id: str
    qualification_ref: str

    def __post_init__(self) -> None:
        for path in (self.binary, self.home, self.credential_home, self.workdir):
            if not path.is_absolute() or path.is_symlink():
                raise ValueError("absolute non-symlink worker paths required")
        if not self.binary.is_file() or not os.access(self.binary, os.X_OK):
            raise ValueError("native executable required")
        for path in (self.home, self.credential_home, self.workdir):
            if not path.is_dir() or path.stat().st_mode & 0o077:
                raise ValueError("private worker directories required")
        if len({p.resolve() for p in (self.home, self.credential_home, self.workdir)}) != 3:
            raise ValueError("separate worker, credential and work directories required")
        for value in (self.model, self.adapter_id, self.qualification_ref):
            if not value.strip() or len(value) > 512:
                raise ValueError("bounded deployment identity required")
        url = urlsplit(self.egress_url)
        if (url.scheme not in {"http", "https"} or not url.hostname
                or url.username or url.password or url.query or url.fragment):
            raise ValueError("invalid qualified egress URL")
        if not re.fullmatch(r"[a-zA-Z0-9.-]+", self.upstream_host):
            raise ValueError("invalid upstream host")


@dataclass(frozen=True)
class ServedEvidence:
    execution_id: str
    thread_id: str
    adapter_id: str
    qualification_ref: str
    served_model: str
    receipt_id: str


def _not_cancelled() -> bool:
    return False


class CodexAdapter:
    def __init__(self, spec: CodexSpec, *,
                 served_evidence: Callable[[str, str, float], Optional[ServedEvidence]],
                 cancelled: Callable[[], bool] = _not_cancelled) -> None:
        self.spec = spec
        self.served_evidence = served_evidence
        self.cancelled = cancelled

    def command(self, request: ContentRequest) -> list[str]:
        """Translate trusted resolution into flags; prompt travels only on stdin."""
        args = [str(self.spec.binary), "exec", "--ignore-user-config", "--strict-config",
                "--ephemeral", "--json", "--skip-git-repo-check", "--sandbox", "read-only",
                "-m", self.spec.model]
        config: dict[str, object] = {
            "model_provider": "controlled", "forced_login_method": "chatgpt",
            "cli_auth_credentials_store": "file", "approval_policy": "never",
            "model_providers.controlled.name": "Qualified native transport",
            "model_providers.controlled.base_url": self.spec.egress_url,
            "model_providers.controlled.wire_api": "responses",
            "model_providers.controlled.requires_openai_auth": True,
            "model_providers.controlled.request_max_retries": 0,
            "model_providers.controlled.stream_max_retries": 0,
            "web_search": "disabled", "analytics.enabled": False,
            "feedback.enabled": False, "check_for_update_on_startup": False,
            "memories.generate_memories": False, "memories.use_memories": False,
            "history.persistence": "none",
        }
        for key, value in config.items():
            args.extend(["-c", key + "=" + json.dumps(value)])
        headers = '{ "X-Upstream-Host" = ' + json.dumps(self.spec.upstream_host)
        headers += ', "X-Router-Execution-ID" = ' + json.dumps(request.execution_id) + ' }'
        args.extend(["-c", "model_providers.controlled.http_headers=" + headers])
        for name in _DISABLED_FEATURES:
            args.extend(["-c", "features." + name + "=false"])
        return args + ["-"]

    def __call__(self, request: ContentRequest, candidate: ContentCandidate,
                 content: str) -> ContentResult:
        deadline = time.monotonic() + request.timeout_seconds
        if not request.execution_id:
            raise ValueError("durably minted execution identity required")
        if os.name != "posix":
            raise ValueError("qualified POSIX worker required")
        if (candidate.adapter_id != self.spec.adapter_id
                or candidate.qualification_ref != self.spec.qualification_ref
                or candidate.execution_class != "content" or candidate.billing != "subscription"):
            raise ValueError("candidate does not match qualified native adapter")
        if len(content.encode("utf-8")) != request.input_bytes:
            raise ValueError("complete input size mismatch")
        if self.cancelled():
            raise InterruptedError("native execution cancelled")
        with tempfile.TemporaryDirectory(dir=self.spec.workdir) as directory:
            environment = {"PATH": "/usr/bin:/bin", "HOME": str(self.spec.home),
                           "CODEX_HOME": str(self.spec.credential_home), "TMPDIR": directory}
            output = self._run(request, content, Path(directory), environment, deadline)
        text, thread_id = _parse(output, request.output_bytes)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("native execution deadline exceeded before receipt")
        if self.cancelled():
            raise InterruptedError("native execution cancelled before receipt")
        # The injected receipt reader is trusted local transport infrastructure,
        # and MUST return within the supplied remaining-seconds budget. It must
        # never make an unbounded network call. Late replies are refused below.
        evidence = self.served_evidence(request.execution_id, thread_id, remaining)
        if time.monotonic() >= deadline:
            raise TimeoutError("served-route evidence deadline exceeded")
        if self.cancelled():
            raise InterruptedError("native execution cancelled during receipt")
        if (evidence is None or evidence.execution_id != request.execution_id
                or evidence.thread_id != thread_id
                or evidence.adapter_id != candidate.adapter_id
                or evidence.qualification_ref != candidate.qualification_ref):
            raise ValueError("matching served-route evidence required")
        return ContentResult(text, evidence.served_model, evidence.receipt_id)

    def _run(self, request: ContentRequest, content: str, directory: Path,
             environment: dict[str, str], deadline: float) -> bytes:
        # An unlinked private input file avoids stdin/stdout pipe deadlock and
        # keeps content out of argv, environment and persistent session files.
        with tempfile.TemporaryFile(dir=directory) as source:
            _ = source.write(content.encode("utf-8"))
            _ = source.seek(0)
            process = subprocess.Popen(self.command(request), cwd=directory, env=environment,
                                       stdin=source, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       start_new_session=True)
            assert process.stdout is not None and process.stderr is not None
            output = bytearray()
            captured = 0
            try:
                with selectors.DefaultSelector() as selector:
                    _ = selector.register(process.stdout, selectors.EVENT_READ, True)
                    _ = selector.register(process.stderr, selectors.EVENT_READ, False)
                    while selector.get_map() or process.poll() is None:
                        if self.cancelled():
                            raise InterruptedError("native execution cancelled")
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError("native execution deadline exceeded")
                        for key, _mask in selector.select(min(remaining, 0.1)):
                            chunk = os.read(key.fd, 4096)
                            if not chunk:
                                _ = selector.unregister(key.fileobj)
                                continue
                            captured += len(chunk)
                            if captured > _MAX_CAPTURE_BYTES:
                                raise ValueError("native output exceeds capture bound")
                            if key.data is True:
                                output.extend(chunk)
                try:
                    code = process.wait(timeout=max(0.001, deadline - time.monotonic()))
                except subprocess.TimeoutExpired as exc:
                    raise TimeoutError("native execution deadline exceeded") from exc
                if code != 0:
                    # Provider diagnostics can contain content or credentials.
                    # Caller records this bounded outcome, never raw stderr.
                    raise RuntimeError("native execution failed: exit " + str(code))
                return bytes(output)
            finally:
                # Kill the session even if the leader exited with children holding
                # inherited pipes. A qualified OS sandbox must prevent detachment.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass  # No process group remains; wait below verifies leader exit.
                _ = process.wait()
                process.stdout.close()
                process.stderr.close()


def _parse(output: bytes, limit: int) -> tuple[str, str]:
    thread_id: Optional[str] = None
    messages: list[str] = []
    completed = False
    for line in output.decode("utf-8", errors="strict").splitlines():
        value: object = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("invalid native event")
        event = cast(dict[str, object], value)
        kind = event.get("type")
        if completed:
            raise ValueError("event after terminal completion")
        if kind == "thread.started":
            identity = event.get("thread_id")
            if thread_id is not None or not isinstance(identity, str) or not identity.strip():
                raise ValueError("invalid native thread identity")
            thread_id = identity
        elif kind == "turn.started":
            continue
        elif kind in {"item.started", "item.updated", "item.completed"}:
            item = event.get("item")
            if not isinstance(item, dict):
                raise ValueError("invalid native item")
            fields = cast(dict[str, object], item)
            if fields.get("type") not in {"agent_message", "reasoning"}:
                raise ValueError("effectful or erroneous native item")
            if kind == "item.completed" and fields.get("type") == "agent_message":
                text = fields.get("text")
                if not isinstance(text, str):
                    raise ValueError("invalid native answer")
                messages.append(text)
        elif kind == "turn.completed":
            completed = True
        else:
            raise ValueError("unsupported or failed native event")
    text = "\n".join(messages)
    if not completed or thread_id is None or not messages or len(text.encode("utf-8")) > limit:
        raise ValueError("incomplete or oversized native result")
    return text, thread_id
