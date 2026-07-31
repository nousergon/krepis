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

**Dependency:** gitleaks must be on ``PATH`` at runtime and its config
file (``gitleaks-egress.toml`` + its ``[extend]`` chain) must be
readable. The standard location is ``/opt/llm-routing/``; override with
``KREPIS_GITLEAKS_DIR``.  On Lambda this requires bundling the gitleaks
binary + config files in the deployment package or a Lambda layer.
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

def _gitleaks_dir() -> str:
    """Resolve the gitleaks config directory.

    Checks env override first, then the standard groom/dashboard-box
    paths, then a Lambda-layer path.
    """
    env_dir = os.environ.get("KREPIS_GITLEAKS_DIR")
    if env_dir and os.path.isdir(env_dir):
        return env_dir
    for candidate in (
        "/opt/llm-routing",
        "/opt/groom-llm-routing",
        "/opt/drain-llm-routing",
    ):
        if os.path.isdir(candidate):
            return candidate
    # Lambda fallback — bundled in the deployment package alongside krepis
    lambda_candidate = os.path.join(os.path.dirname(__file__), "_gitleaks_config")
    if os.path.isdir(lambda_candidate):
        return lambda_candidate
    return "/opt/llm-routing"  # default; will fail loudly at scan time if missing


GITLEAKS_DIR = _gitleaks_dir()
GITLEAKS_CONFIG = os.path.join(GITLEAKS_DIR, "gitleaks-egress.toml")
GITLEAKS_BIN = shutil.which("gitleaks") or "gitleaks"
GITLEAKS_TIMEOUT_S = 8

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
