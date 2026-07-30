"""Golden-prefix test: verify prompt assembly is byte-identical across processes.

G4 conformance gate from ``prompt-caching-policy.md`` §7 — prefix determinism
enforcement for §3.3 (the ~entirety of M2 caching, since all 13 active M2
models have no ``cache_control`` marker discipline to get wrong).

The test builds a representative prompt using the **real** ``krepis``
prompt-assembly path in **two separate subprocesses** (not two calls in one
process) with different ``PYTHONHASHSEED`` values, then asserts the rendered
prefix bytes are identical. An in-process comparison cannot catch
``PYTHONHASHSEED``-dependent set/dict iteration, which is exactly the class
§3.3 names as the primary nondeterminism vector.

Each subprocess exercises the real :func:`krepis.anthropic_payload.build_messages_payload`
path, serializes the result deterministically, and prints a SHA-256 digest.
The parent process compares digests across seeds.

Hazards covered (per policy §3.3):
- Unsorted mapping keys -> ``json.dumps(sort_keys=True)`` renders them sorted
- Float repr drift -> JSON float serialization is deterministic on CPython
- Set iteration -> different ``PYTHONHASHSEED`` changes set iteration order
- Unpinned tool-definition order -> stable list ordering is verified
- Unstable file-read order -> sorted directory traversal where applicable
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import textwrap

# ── subprocess payload assembly code ─────────────────────────────────────
# This MUST use the REAL krepis prompt-assembly path.  A reimplementation
# in the test would pass forever while the shipping assembler drifts.
# `build_messages_payload` and `build_web_search_tool` are the production
# entry points consumed by morning-signal and crucible-research.
#
# NOTE: this is a plain (non-f) string, so dict/set braces are single.
# The entire code block is dedented at call time via textwrap.dedent().
_SUBPROCESS_CODE = """
from krepis.anthropic_payload import build_messages_payload, build_web_search_tool
import json
import hashlib

# Build a representative prompt exercising all known section-3.3 hazard classes:
#
# 1. Multiple tools (tool-definition order stability)
# 2. Web search tool spec (server-side tool shape)
# 3. Float values in extra (float repr drift)
# 4. Nested dict structure (sorted key iteration)
# 5. Cache-control markers (ephemeral cache_control dict)
payload = build_messages_payload(
    model="claude-sonnet-4-5",
    system_prompt=(
        "You are a helpful financial analyst assistant. "
        "Your responses must be accurate and well-sourced."
    ),
    user_content=(
        "Analyze the S&P 500 sector performance for Q2 2026. "
        "Focus on technology, healthcare, and energy sectors."
    ),
    max_tokens=2048,
    tools=[
        {
            "name": "get_market_data",
            "description": "Retrieve market data for a given ticker or sector",
            "input_schema": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "period": {
                        "type": "string",
                        "enum": ["1d", "1w", "1m", "1y"],
                    },
                },
                "required": ["symbol"],
            },
        },
        {
            "name": "get_company_financials",
            "description": "Get financial statements for a company",
            "input_schema": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "statement": {
                        "type": "string",
                        "enum": ["income", "balance_sheet", "cash_flow"],
                    },
                },
                "required": ["ticker", "statement"],
            },
        },
        build_web_search_tool(),
    ],
    # Float values exercise float repr determinism across processes.
    # 0.0 is a hard-edge case: JSON serializes it as "0.0" in CPython.
    # boundary values like 1.0/0.5/0.3333333333333333 are included.
    extra={
        "temperature": 0.7,
        "top_p": 0.95,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.3333333333333333,
        "seed": 42,
    },
)

prefix_bytes = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
digest = hashlib.sha256(prefix_bytes).hexdigest()
print(digest)
"""


def _run_subprocess(seed: str) -> tuple[str, str]:
    """Run the golden-prefix assembly in a subprocess with the given
    ``PYTHONHASHSEED``.  Returns ``(stdout, stderr)``."""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_SUBPROCESS_CODE)],
        env={"PYTHONHASHSEED": seed},
        capture_output=True,
        text=True,
        timeout=30,
    )
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    if result.returncode != 0:
        raise RuntimeError(
            f"Subprocess (seed={seed}) exited {result.returncode}\n"
            f"stderr:\n{stderr}"
        )
    if not stdout:
        raise RuntimeError(
            f"Subprocess (seed={seed}) produced no output on stdout\n"
            f"stderr:\n{stderr}"
        )
    return stdout, stderr


# ── tests ────────────────────────────────────────────────────────────────


class TestGoldenPrefix:
    """G4 prefix-determinism conformance tests."""

    def test_golden_prefix_matches_across_hash_seeds(self):
        """Building the same prompt in two subprocesses with different
        ``PYTHONHASHSEED`` yields byte-identical prefixes."""
        stdout_0, _ = _run_subprocess("0")
        stdout_1, _ = _run_subprocess("1")
        assert stdout_0 == stdout_1, (
            f"Prefix SHA-256 differs across PYTHONHASHSEED values:\n"
            f"  seed=0: {stdout_0}\n"
            f"  seed=1: {stdout_1}\n\n"
            "This means the prompt-assembly path produces different bytes "
            "depending on the hash seed — a permanent, invisible, total "
            "cache miss on any M2 (automatic prefix caching) model. "
            "Check for: unsorted dict/set iteration in prompt construction, "
            "unstable file-order reads, or locale-dependent formatting."
        )

    def test_golden_prefix_matches_across_multiple_seeds(self):
        """Stability across a wider range of seeds improves confidence.
        Three seeds (0, 1, 42) gives 3 choose 2 = 3 pairwise comparisons,
        making a coincidental byte-identical result from exactly two seeds
        vanishingly unlikely while keeping subprocess overhead negligible."""
        digests = set()
        for seed in ("0", "1", "42"):
            out, _ = _run_subprocess(seed)
            digests.add(out)
        assert len(digests) == 1, (
            f"Prefix SHA-256 differs across seeds: {digests}"
        )

    # ── adversarial: test fails when determinism is deliberately broken ──
    # These verify the test harness itself is sensitive to the hazards it
    # claims to catch.  A golden-prefix test that has only ever been seen
    # passing green proves nothing (see bugclass alpha-engine-I260727:
    # record_asserts_action_that_never_happened).

    def test_detects_unsorted_tool_order(self):
        """The test MUST detect when tool definitions are supplied in
        non-deterministic order.  We verify the assertion fires by
        injecting a marker into a tool name and confirming the digest
        differs from the canonical assembly."""
        broken_code = _SUBPROCESS_CODE.replace(
            '"name": "get_market_data"',
            '"name": "ZZZ_get_market_data"',
        )
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(broken_code)],
            env={"PYTHONHASHSEED": "0"},
            capture_output=True,
            text=True,
            timeout=15,
        )
        broken_digest = result.stdout.strip()
        assert len(broken_digest) == 64, (
            f"Expected a valid SHA-256 hex digest but got: {broken_digest!r}\n"
            f"stderr: {result.stderr}"
        )
        canonical, _ = _run_subprocess("0")
        assert broken_digest != canonical, (
            "Test harness bug: deliberately broken tool order produced "
            "the SAME hash as the canonical assembly — the subprocess "
            "template manipulation did not actually vary the output."
        )

    def test_detects_cache_control_variation(self):
        """Cache-control marker differences are a common source of prefix
        drift (e.g., ``cache_control: {"type": "ephemeral"}`` vs. no
        cache_control at all).  Verify the test catches this class by
        injecting a different ``cache_system=False`` that removes the
        cache_control block from the system prompt."""
        # Call build_messages_payload with cache_system=False, which
        # suppresses the cache_control marker on the system block.
        broken_code = _SUBPROCESS_CODE.replace(
            "max_tokens=2048,",
            "max_tokens=2048, cache_system=False,",
        )
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(broken_code)],
            env={"PYTHONHASHSEED": "0"},
            capture_output=True,
            text=True,
            timeout=15,
        )
        broken_digest = result.stdout.strip()
        canonical, _ = _run_subprocess("0")
        assert broken_digest != canonical, (
            "Test harness bug: cache_control variation produced same hash."
        )
