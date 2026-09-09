"""In-process DLP scan for request bodies — the Lambda-safe counterpart to
``llm_egress_proxy.py``.

The egress proxy is a **localhost** HTTP gateway (``127.0.0.1:8990``) that
scans outbound LLM request bodies before forwarding them upstream. It covers
the laptop and dashboard-box paths, but Lambda cannot reach ``127.0.0.1``,
and no Lambda is configured to reach a proxy anywhere else
(``alpha-engine-config-I4927``).

This module extracts the scan logic into a **caller-side hook** that runs
inside the same process — no network hop, no separate service, works on
Lambda without a VPC. It is the *in-process* tier of DLP enforcement
(``llm-egress-proxy-policy`` §2a: voluntary tier — the caller hooks itself
rather than being network-compelled) and pairs with the existing CI guard
tests that catch call-sites bypassing the shared client (config#4459).

**Coverage:** this module scans request bodies that pass through
:class:`krepis.llm.LLMClient`. It does NOT cover call-sites that construct
their own provider SDK clients directly — those are caught by CI guard
tests, not by runtime enforcement. Full network-layer interception (VPC /
route-table compulsion, the SOTA tier) remains the long-term posture;
this module closes the immediate gap at the chokepoint that fleet code
already uses.

**Dependency:** gitleaks must be on ``PATH`` at runtime. Its config
(``gitleaks-egress.toml`` + its ``[extend]`` chain) is **shipped as krepis
package data** (``krepis/dlp_config/``) and therefore present in every
context that can import krepis — Lambda, an ephemeral EC2 spot box, a CI
runner, a laptop, a container. An operator-managed directory still wins
when one exists: ``$KREPIS_GITLEAKS_DIR``, then
``/opt/{llm,groom,drain}-llm-routing``, then the packaged copy.

Until 2026-09-09 there was no packaged copy, so the resolver's last resort
was the string ``/opt/llm-routing`` whether or not anything was there, and
any context without that directory failed CLOSED on **every** outbound LLM
call. That was filed four times against four substrates
(``alpha-engine-config-I7913`` laptop, ``-I7719`` CI runner, ``-I9972``
crucible-v2 spot box, ``-I9407`` backtester tests), fixed on none, and ran
for 27 days undetected on the data-collector flow-doctor diagnosis path.
Provisioning a directory per substrate is a control that has to be
re-installed everywhere the code can run; packaging it is a control that
cannot be missing.

The gitleaks **binary** is still an environment dependency — a Go binary
krepis cannot ship in a wheel. Call :func:`preflight` at startup (or
``python -m krepis.session_dlp preflight``) to assert readiness where a
first outbound call would otherwise be the discovery mechanism.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time as _time
from collections import OrderedDict
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "DLPVerdict",
    "DLPBlockError",
    "check_request",
    "scan_request",
    "DLP_DISABLED",
    "DLP_BLOCKED",
    "DLP_SCAN_ERROR",
    "DLP_OK",
    "dlp_enabled",
    "packaged_gitleaks_dir",
    "preflight",
    "DLPPreflight",
]

# ── verdict constants ────────────────────────────────────────────────────

DLP_OK = "ok"
"""Scan completed; no secrets found."""

DLP_BLOCKED = "dlp_block"
"""A gitleaks finding was confirmed — the request should be blocked."""

DLP_SCAN_ERROR = "scan_error"
"""The scanner itself failed (config, binary, timeout) — fail-closed."""

DLP_DISABLED = "dlp_disabled"
"""DLP scanning is administratively disabled (env / feature flag)."""


def dlp_enabled() -> bool:
    """True unless DLP scanning is explicitly disabled via env."""
    return os.environ.get("KREPIS_DLP_DISABLED", "").lower() not in (
        "1", "true", "yes",
    )


# ── config resolution ────────────────────────────────────────────────────

# The gitleaks ruleset shipped inside the wheel. Both files live in the same
# package directory, which matters: ``gitleaks-egress.toml``'s ``[extend].path``
# is relative and gitleaks resolves it against the PROCESS CWD, so the scan
# below runs with ``cwd=GITLEAKS_DIR`` and the two files must be siblings.
_PACKAGED_CONFIG_DIRNAME = "dlp_config"
_ENTRY_CONFIG_FILENAME = "gitleaks-egress.toml"

# Operator-managed directories, in precedence order. A box that manages its own
# ruleset keeps it; the packaged copy is the floor, not an override.
_OPERATOR_CONFIG_DIRS = (
    "/opt/llm-routing",
    "/opt/groom-llm-routing",
    "/opt/drain-llm-routing",
)


def packaged_gitleaks_dir() -> Optional[str]:
    """Filesystem path of the gitleaks ruleset shipped with krepis, or None.

    gitleaks is a subprocess: it needs a real directory on disk, both for
    ``--config`` and as the cwd its relative ``[extend].path`` resolves
    against. ``importlib.resources.files()`` is the correct accessor, but it
    returns a ``Traversable`` that need not be a real path (a zipimported
    krepis has none). Rather than extract to a temp directory whose lifetime
    nothing owns, this returns ``None`` in that case and the caller fails loud
    with a message naming the missing config — the same fail-closed outcome as
    any other unusable ruleset, never a silent skip.
    """
    candidates = []
    try:
        from importlib.resources import files as _files

        candidates.append(str(_files("krepis") / _PACKAGED_CONFIG_DIRNAME))
    except Exception:  # noqa: BLE001 - see __file__ fallback immediately below
        # importlib.resources can raise for a namespace package or an exotic
        # loader. That is not a reason to give up: __file__ answers the same
        # question for every ordinary install, and a genuinely unresolvable
        # ruleset still returns None below and fails closed at scan time.
        pass
    candidates.append(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     _PACKAGED_CONFIG_DIRNAME)
    )
    for d in candidates:
        if os.path.isfile(os.path.join(d, _ENTRY_CONFIG_FILENAME)):
            return d
    return None


def _gitleaks_dir() -> str:
    """Resolve the gitleaks config directory.

    ``$KREPIS_GITLEAKS_DIR``, then the operator ``/opt`` directories, then the
    ruleset packaged inside krepis.

    Each candidate is tested for the ENTRY CONFIG FILE, not merely for the
    directory existing. A present-but-empty ``/opt/llm-routing`` — the shape a
    half-finished bootstrap leaves behind — used to satisfy ``os.path.isdir``
    and shadow every remaining candidate, turning a provisioning slip into a
    total egress outage with the packaged copy sitting unused on the same disk.
    """
    env_dir = os.environ.get("KREPIS_GITLEAKS_DIR")
    if env_dir and os.path.isfile(os.path.join(env_dir, _ENTRY_CONFIG_FILENAME)):
        return env_dir
    for candidate in _OPERATOR_CONFIG_DIRS:
        if os.path.isfile(os.path.join(candidate, _ENTRY_CONFIG_FILENAME)):
            return candidate
    packaged = packaged_gitleaks_dir()
    if packaged is not None:
        return packaged
    # Nothing resolved — including the packaged copy, which means krepis itself
    # is installed in a form whose package data is unreadable. Return the
    # conventional path so the scan-time error names something an operator can
    # act on; the scan fails closed either way.
    return _OPERATOR_CONFIG_DIRS[0]


GITLEAKS_DIR = _gitleaks_dir()
GITLEAKS_CONFIG = os.path.join(GITLEAKS_DIR, "gitleaks-egress.toml")
GITLEAKS_BIN = shutil.which("gitleaks") or "gitleaks"
GITLEAKS_TIMEOUT_S = 8

_EXTEND_PATH_RE = re.compile(
    r'^\s*\[extend\]\s*$.*?^\s*path\s*=\s*"([^"]+)"', re.M | re.S
)


def _verify_gitleaks_config_chain_at(config_dir: str) -> Optional[str]:
    """Verify the entry config in *config_dir* and its ``[extend]`` chain resolve.

    ``gitleaks-egress.toml`` extends its parent by a path that is **relative to
    the gitleaks process's CWD**, not to the referencing config file's
    directory (see the module docstring and the toml's own header). The caller
    invokes gitleaks with ``cwd=config_dir``, so the extend target must resolve
    relative to *config_dir*. Checking that here — before shelling out — turns
    a misconfigured chain into a message naming exactly the missing file,
    instead of a bare ``gitleaks exited 1`` that a caller has to re-derive from
    stderr.

    Returns ``None`` if the chain resolves; otherwise an error string naming
    the missing file. Parametrised by directory so :func:`preflight` can grade
    a candidate without mutating module state.
    """
    config = os.path.join(config_dir, _ENTRY_CONFIG_FILENAME)
    if not os.path.isfile(config):
        return f"gitleaks config not found: {config!r}"
    try:
        with open(config, encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError as exc:
        return f"gitleaks config unreadable: {config!r} ({exc})"
    m = _EXTEND_PATH_RE.search(text)
    if not m:
        # No [extend] stanza — nothing further to resolve.
        return None
    extend_path = m.group(1)
    resolved = (
        extend_path
        if os.path.isabs(extend_path)
        else os.path.join(config_dir, extend_path)
    )
    if not os.path.isfile(resolved):
        return (
            f"gitleaks [extend].path {extend_path!r} in {config!r} "
            f"resolves to {resolved!r} (against GITLEAKS_DIR={config_dir!r}), "
            "which does not exist"
        )
    return None


def _verify_gitleaks_config_chain() -> Optional[str]:
    """Verify the chain under the module-resolved :data:`GITLEAKS_DIR`."""
    return _verify_gitleaks_config_chain_at(GITLEAKS_DIR)


# ── content substitution (mirrors llm_egress_proxy.py) ────────────────────

# Large base64 blobs dominate scan time; replaced in the scan copy only.
_BASE64_BLOB_RE = re.compile(r"[A-Za-z0-9+/_-]{2000,}={0,2}")

# Canonical UUIDs trip stock generic-api-key on agent turns; they carry no
# secret material for any fleet provider.  Substituted in scan copy only.
_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)

# ── leaf cache ───────────────────────────────────────────────────────────


class _LeafScanCache:
    """LRU set of string-leaf digests already scanned clean.

    Keyed on SHA-256 of the leaf content.  Invalidated wholesale whenever
    the gitleaks config chain changes on disk — a rule tightening always
    re-scans from scratch.  Thread-safe.
    """

    def __init__(self, capacity: int = 100_000):
        self._capacity = capacity
        self._clean: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self._config_sig: Optional[tuple] = None
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _config_signature() -> tuple:
        sig = []
        paths = [GITLEAKS_CONFIG]
        try:
            with open(GITLEAKS_CONFIG, encoding="utf-8", errors="replace") as f:
                paths.extend(
                    re.findall(r'^\s*path\s*=\s*"([^"]+)"', f.read(), re.M)
                )
        except OSError:
            pass
        # Resolve every extend target against GITLEAKS_DIR before stat-ing it.
        #
        # THE SECOND INSTANCE OF I8267's CLASS, IN THE SAME FILE, AND SILENT.
        # `[extend].path` is relative BY CONTRACT (see the toml's header), and
        # `os.stat` resolves a relative path against the process cwd exactly
        # as gitleaks does. PR183 gave the SCAN its `cwd=GITLEAKS_DIR`; this
        # read never got one, so off the routing directory every extend target
        # fell into the `except OSError` below and was recorded as
        # `(path, None, None)`.
        #
        # Measured 2026-08-25 from `/tmp` at v0.59.35, with the scan itself
        # healthy: `('./gitleaks-custom.toml', None, None)`.
        #
        # It degrades in the one direction this cache must never degrade. The
        # signature then cannot change when `gitleaks-custom.toml` does — and
        # that is the file holding every fleet-specific secret shape — so
        # TIGHTENING A RULE would not invalidate the leaf cache, and content
        # already scanned clean under the looser ruleset would keep passing on
        # a cached verdict. A stale ALLOW, reported by nothing: the scan goes
        # on returning `ok`, quickly, which is exactly what a healthy cache
        # looks like.
        #
        # The loud half of this class was fixed and the quiet half was left,
        # which is the ordinary shape of a partial fix — the instance that
        # announces itself gets the patch.
        paths = [
            path if os.path.isabs(path)
            else os.path.normpath(os.path.join(GITLEAKS_DIR, path))
            for path in paths
        ]
        for p in paths:
            try:
                st = os.stat(p)
                sig.append((p, st.st_mtime_ns, st.st_size))
            except OSError:
                sig.append((p, None, None))
        return tuple(sig)

    @staticmethod
    def digest(leaf: str) -> bytes:
        return hashlib.sha256(leaf.encode("utf-8", errors="replace")).digest()

    def refresh_config(self) -> None:
        sig = self._config_signature()
        with self._lock:
            if sig != self._config_sig:
                if self._config_sig is not None:
                    logger.info(
                        "dlp scan cache cleared: gitleaks config chain changed on disk"
                    )
                self._clean.clear()
                self._config_sig = sig

    def is_clean(self, digest: bytes) -> bool:
        with self._lock:
            if digest in self._clean:
                self._clean.move_to_end(digest)
                self.hits += 1
                return True
            self.misses += 1
            return False

    def mark_clean(self, digests: list) -> None:
        with self._lock:
            for d in digests:
                self._clean[d] = True
                self._clean.move_to_end(d)
            while len(self._clean) > self._capacity:
                self._clean.popitem(last=False)

    def stats(self) -> dict:
        with self._lock:
            return {
                "size": len(self._clean),
                "capacity": self._capacity,
                "hits": self.hits,
                "misses": self.misses,
            }


_cache = _LeafScanCache()


# ── string leaf extraction ───────────────────────────────────────────────


def _collect_strings(value, out: list) -> None:
    """Recursively collect every string leaf in a decoded JSON value."""
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, dict):
        for v in value.values():
            _collect_strings(v, out)
    elif isinstance(value, list):
        for v in value:
            _collect_strings(v, out)


# ── scan ─────────────────────────────────────────────────────────────────


def scan_request(body: bytes) -> Tuple[str, str, float, float]:
    """Scan *body* (raw JSON bytes of an outbound LLM request) for secrets.

    Returns ``(verdict, reason, scan_ms, cache_ratio)``.

    *verdict* is one of :data:`DLP_OK`, :data:`DLP_BLOCKED`,
    :data:`DLP_SCAN_ERROR`, or :data:`DLP_DISABLED`.

    Incremental: only string leaves never previously scanned clean under
    the current gitleaks config chain are scanned.  Leaves overlapping a
    finding are never cached; clean leaves from a blocked request still
    are, so a caller's automatic retry of a near-identical body stays
    cheap.

    Fail-closed: a scan-infrastructure failure (missing binary, broken
    config, timeout) returns :data:`DLP_SCAN_ERROR`.  Callers should
    treat this as a block — it is NOT safe to forward unscanned.
    """
    if not dlp_enabled():
        return DLP_DISABLED, "DLP scanning administratively disabled", 0.0, 1.0

    t0 = _time.monotonic()

    try:
        parsed = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return DLP_BLOCKED, "request body is not valid JSON — failing closed", 0.0, 0.0

    leaves: list = []
    _collect_strings(parsed, leaves)
    _cache.refresh_config()

    new_leaves = []
    new_digests = []
    seen_this_request: set = set()
    for leaf in leaves:
        d = _LeafScanCache.digest(leaf)
        if d in seen_this_request:
            continue
        if _cache.is_clean(d):
            continue
        seen_this_request.add(d)
        new_leaves.append(leaf)
        new_digests.append(d)

    total = len(leaves)
    cache_ratio = 1.0 if not total else 1.0 - (len(new_leaves) / total)
    if not new_leaves:
        return DLP_OK, "", (_time.monotonic() - t0) * 1000.0, cache_ratio

    # Flatten only the NEW leaves, tracking each leaf's line range so a
    # finding's line numbers map back to the leaf that must not be cached.
    scan_lines: list = []
    leaf_line_ranges = []  # (first_line_1based, last_line_1based) per new leaf
    for leaf in new_leaves:
        substituted = _BASE64_BLOB_RE.sub(
            "[[large-blob-excluded-from-scan]]", leaf
        )
        substituted = _UUID_RE.sub("[[uuid-excluded-from-scan]]", substituted)
        lines = substituted.split("\n")
        first = len(scan_lines) + 1
        scan_lines.extend(lines)
        leaf_line_ranges.append((first, len(scan_lines)))
    scan_text = "\n".join(scan_lines).encode("utf-8", errors="replace")

    config_chain_error = _verify_gitleaks_config_chain()
    if config_chain_error:
        return DLP_SCAN_ERROR, (
            f"{config_chain_error} — failing closed"
        ), (_time.monotonic() - t0) * 1000.0, cache_ratio

    report_path = None
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp.write(scan_text)
            tmp_path = tmp.name
        report_path = tmp_path + ".report.json"
        result = subprocess.run(
            [
                GITLEAKS_BIN, "detect", "--no-git",
                "--source", tmp_path,
                "--config", GITLEAKS_CONFIG,
                "--no-banner", "--redact", "--exit-code", "1",
                "--report-format", "json", "--report-path", report_path,
            ],
            capture_output=True, text=True, timeout=GITLEAKS_TIMEOUT_S,
            # gitleaks-egress.toml's [extend] target is a RELATIVE path,
            # resolved against the process CWD — not against the config
            # file's own directory. Every consumer must run gitleaks from
            # the config's directory or the extend chain silently fails
            # to load (alpha-engine-config-I8267; the exact condition
            # that blocked 100% of outbound traffic on the dashboard box
            # in alpha-engine-config-I4451 / -I4511).
            cwd=GITLEAKS_DIR,
        )
    except subprocess.TimeoutExpired:
        return DLP_SCAN_ERROR, (
            "gitleaks scan timed out — failing closed"
        ), (_time.monotonic() - t0) * 1000.0, cache_ratio
    except FileNotFoundError:
        return DLP_SCAN_ERROR, (
            "gitleaks binary not found on PATH — failing closed"
        ), (_time.monotonic() - t0) * 1000.0, cache_ratio
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    scan_ms = (_time.monotonic() - t0) * 1000.0

    if result.returncode == 0:
        _cache.mark_clean(new_digests)
        try:
            os.unlink(report_path)
        except OSError:
            pass
        return DLP_OK, "", scan_ms, cache_ratio

    if result.returncode != 1:
        try:
            os.unlink(report_path)
        except OSError:
            pass
        return DLP_SCAN_ERROR, (
            f"gitleaks scan errored (exit {result.returncode}) — failing closed"
        ), scan_ms, cache_ratio

    # exit-code 1: findings OR config-load failure. Distinguish by the report.
    report_ok = True
    findings = []
    try:
        with open(report_path, encoding="utf-8") as fh:
            findings = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        report_ok = False
        report_err = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            os.unlink(report_path)
        except OSError:
            pass

    if not report_ok:
        stderr_tail = (result.stderr or "").strip().replace("\n", " ")[-400:]
        return DLP_SCAN_ERROR, (
            "gitleaks exited 1 but wrote no readable report — the scan did "
            "NOT run (this is a scanner/config failure, NOT a secret finding). "
            f"report_error={report_err} gitleaks_stderr={stderr_tail!r}"
        ), scan_ms, cache_ratio

    # Cache clean leaves even on a block — only leaves overlapping a
    # finding stay uncached and will be re-scanned next time.
    dirty_leaf_idx: set = set()
    for f in findings:
        f_start = int(f.get("StartLine", 1))
        f_end = int(f.get("EndLine", f.get("StartLine", 1)))
        for i, (first, last) in enumerate(leaf_line_ranges):
            if first <= f_end and last >= f_start:
                dirty_leaf_idx.add(i)
    if not findings:
        dirty_leaf_idx = set(range(len(new_leaves)))
    _cache.mark_clean(
        [d for i, d in enumerate(new_digests) if i not in dirty_leaf_idx]
    )

    rules = sorted({f.get("RuleID", "unknown") for f in findings}) or ["unknown"]
    reason = (
        f"gitleaks flagged outbound request body — rules {rules}"
    )
    return DLP_BLOCKED, reason, scan_ms, cache_ratio


class DLPVerdict:
    """A completed DLP scan result — callers branch on :attr:`verdict`."""

    __slots__ = ("verdict", "reason", "scan_ms", "cache_ratio")

    def __init__(self, verdict: str, reason: str, scan_ms: float, cache_ratio: float):
        self.verdict = verdict
        self.reason = reason
        self.scan_ms = scan_ms
        self.cache_ratio = cache_ratio

    @property
    def ok(self) -> bool:
        return self.verdict == DLP_OK

    @property
    def disabled(self) -> bool:
        return self.verdict == DLP_DISABLED

    @property
    def blocked(self) -> bool:
        return self.verdict == DLP_BLOCKED

    @property
    def scan_error(self) -> bool:
        return self.verdict == DLP_SCAN_ERROR

    @property
    def should_block(self) -> bool:
        """True when the caller must NOT forward the request.

        Blocks on a confirmed finding AND on scan-infrastructure failure
        (fail-closed). Only :data:`DLP_OK` and :data:`DLP_DISABLED` are
        safe to forward.
        """
        return self.verdict not in (DLP_OK, DLP_DISABLED)

    def __repr__(self) -> str:
        return (
            f"DLPVerdict({self.verdict!r}, reason={self.reason!r}, "
            f"scan={self.scan_ms:.0f}ms, cache={self.cache_ratio:.0%})"
        )


def check_request(body: bytes) -> DLPVerdict:
    """Scan *body* and return a :class:`DLPVerdict`.

    The caller checks ``.should_block`` and either proceeds or raises.
    """
    verdict, reason, scan_ms, cache_ratio = scan_request(body)
    return DLPVerdict(verdict, reason, scan_ms, cache_ratio)


class DLPBlockError(RuntimeError):
    """A DLP scan blocked this request — the request MUST NOT be forwarded."""

    def __init__(self, verdict: DLPVerdict):
        super().__init__(
            f"DLP scan blocked outbound LLM request: {verdict.reason}"
        )
        self.verdict = verdict


# ── preflight ────────────────────────────────────────────────────────────


class DLPPreflight:
    """The readiness of the in-process DLP control, as a value.

    Every field here was previously discoverable only by making a real
    outbound LLM call and reading the exception — which is why a missing
    ruleset ran for 27 days on the data-collector diagnosis path before a
    human happened to read the failure text inside an alert body
    (2026-08-13 last success .. 2026-09-09; the DLP cause from 2026-09-01).
    A control whose readiness can only be learned by tripping it is a
    control nothing can monitor.
    """

    __slots__ = (
        "enabled", "config_dir", "config_source", "config_error",
        "ruleset_sha256", "packaged_sha256", "binary_path", "binary_version",
        "binary_error",
    )

    def __init__(
        self,
        *,
        enabled: bool,
        config_dir: str,
        config_source: str,
        config_error: Optional[str],
        ruleset_sha256: Optional[str],
        packaged_sha256: Optional[str],
        binary_path: Optional[str],
        binary_version: Optional[str],
        binary_error: Optional[str],
    ) -> None:
        self.enabled = enabled
        self.config_dir = config_dir
        self.config_source = config_source
        self.config_error = config_error
        self.ruleset_sha256 = ruleset_sha256
        self.packaged_sha256 = packaged_sha256
        self.binary_path = binary_path
        self.binary_version = binary_version
        self.binary_error = binary_error

    @property
    def ready(self) -> bool:
        """True only if a real scan would run and could produce a verdict.

        Administratively disabled is NOT ready. ``KREPIS_DLP_DISABLED`` is one
        environment entry away from turning the fleet's only Lambda-path DLP
        control off, and a process running with it off is otherwise
        indistinguishable from one scanning cleanly (alpha-engine-config-I10001).
        Reporting that state as ready would preserve exactly that blindness.
        """
        return self.enabled and self.config_error is None and self.binary_error is None

    @property
    def ruleset_matches_packaged(self) -> Optional[bool]:
        """Whether the resolved ruleset is byte-identical to the packaged one.

        ``None`` when either side could not be hashed. A ``False`` here is the
        observable form of ruleset divergence between the operator directory
        and the shipped copy (alpha-engine-config-I9712), which until now
        changed detection strength with nothing reporting it.
        """
        if self.ruleset_sha256 is None or self.packaged_sha256 is None:
            return None
        return self.ruleset_sha256 == self.packaged_sha256

    def to_dict(self) -> dict:
        return {
            "ready": self.ready,
            "enabled": self.enabled,
            "config_dir": self.config_dir,
            "config_source": self.config_source,
            "config_error": self.config_error,
            "ruleset_sha256": self.ruleset_sha256,
            "packaged_sha256": self.packaged_sha256,
            "ruleset_matches_packaged": self.ruleset_matches_packaged,
            "binary_path": self.binary_path,
            "binary_version": self.binary_version,
            "binary_error": self.binary_error,
        }

    def __repr__(self) -> str:
        return f"DLPPreflight(ready={self.ready}, source={self.config_source!r})"


def _config_source_for(config_dir: str) -> str:
    env_dir = os.environ.get("KREPIS_GITLEAKS_DIR")
    if env_dir and os.path.abspath(env_dir) == os.path.abspath(config_dir):
        return "env"
    if config_dir in _OPERATOR_CONFIG_DIRS:
        return "operator"
    packaged = packaged_gitleaks_dir()
    if packaged is not None and os.path.abspath(packaged) == os.path.abspath(config_dir):
        return "packaged"
    return "unresolved"


def _hash_config_chain(config_dir: str) -> Optional[str]:
    """SHA-256 over the entry config and every file its ``[extend]`` chain names.

    Hashing the chain rather than the entry file alone is the point: the entry
    config is stable boilerplate and ``gitleaks-custom.toml`` is where every
    fleet-specific secret shape lives, so a hash of the entry file alone would
    read identical across two materially different rulesets.
    """
    entry = os.path.join(config_dir, _ENTRY_CONFIG_FILENAME)
    h = hashlib.sha256()
    seen = set()
    queue = [entry]
    while queue:
        path = queue.pop(0)
        real = os.path.abspath(path)
        if real in seen:
            continue
        seen.add(real)
        try:
            with open(real, "rb") as f:
                data = f.read()
        except OSError:
            return None
        h.update(os.path.basename(real).encode("utf-8"))
        h.update(data)
        text = data.decode("utf-8", errors="replace")
        for rel in re.findall(r'^\s*path\s*=\s*"([^"]+)"', text, re.M):
            queue.append(rel if os.path.isabs(rel) else os.path.join(config_dir, rel))
    return h.hexdigest()


def _binary_probe() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return ``(path, version, error)`` for the gitleaks binary."""
    path = shutil.which("gitleaks")
    if not path:
        return None, None, "gitleaks binary not found on PATH"
    try:
        proc = subprocess.run(
            [path, "version"], capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return path, None, f"gitleaks binary at {path!r} is not runnable ({exc})"
    if proc.returncode != 0:
        return path, None, (
            f"gitleaks binary at {path!r} exited {proc.returncode} on `version`"
        )
    return path, (proc.stdout or proc.stderr).strip() or None, None


def preflight() -> DLPPreflight:
    """Report whether an outbound DLP scan could run here, without running one.

    Call this at process start on any substrate whose first LLM call would
    otherwise be the readiness test. It performs no scan and makes no network
    call; it resolves the ruleset, hashes the chain, and probes the binary.
    """
    config_dir = _gitleaks_dir()
    config_error = _verify_gitleaks_config_chain_at(config_dir)
    ruleset = None if config_error else _hash_config_chain(config_dir)
    packaged_dir = packaged_gitleaks_dir()
    packaged = _hash_config_chain(packaged_dir) if packaged_dir else None
    binary_path, binary_version, binary_error = _binary_probe()
    return DLPPreflight(
        enabled=dlp_enabled(),
        config_dir=config_dir,
        config_source=_config_source_for(config_dir),
        config_error=config_error,
        ruleset_sha256=ruleset,
        packaged_sha256=packaged,
        binary_path=binary_path,
        binary_version=binary_version,
        binary_error=binary_error,
    )


def _main(argv: Optional[list] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m krepis.session_dlp",
        description="Report in-process DLP readiness on this substrate.",
    )
    parser.add_argument("command", choices=["preflight"])
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args(argv)

    pf = preflight()
    if args.json:
        print(json.dumps(pf.to_dict(), indent=2, sort_keys=True))
    else:
        for key, value in sorted(pf.to_dict().items()):
            print(f"{key}: {value}")
    return 0 if pf.ready else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(_main())
