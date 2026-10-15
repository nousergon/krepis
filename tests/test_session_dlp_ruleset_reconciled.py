"""The packaged DLP ruleset is the reconciled one, and it is the CANONICAL copy.

alpha-engine-config-I9712
-------------------------
``gitleaks-custom.toml`` -- the fleet's own secret-shape rules, chain-extended
by every DLP consumer -- existed in four copies (this package, alpha-engine-
config's groom routing dir, claude-code-config's laptop egress proxy,
nousergon-lib's egress package) that diverged in BOTH directions. Neither side
was a superset; each lacked a real fix the other had:

* laptop only: Telegram trailing ``\\b`` removed (a token whose 35th char is
  ``-`` was unmatched, ~1 in 64 real tokens), and ``nousergon-aws-access-key-id``
  (the stock rule fired on 5/20 and 1/20 randomised key ids -- its own entropy
  gate).
* fleet only (and so here): Anthropic rule requires ``{8,}`` body chars, and the
  IBKR rule excludes ``<...>`` placeholders -- both were false positives that
  blocked agents reading a diff of the rule file itself (I4946).

The reconciled file is the UNION of those fixes. It now lives here as the
canonical copy: each other repo's CI parses its copy and the one on krepis
``main`` and fails on any rule-level difference. That makes THIS file's content
load-bearing for three other repos, so these tests pin it -- a rule dropped or
loosened here would otherwise propagate as "the canonical ruleset".

Brian's condition on krepis-PR247 (full prompt text in LLM traces) was a CI
check that fails if krepis's and the proxy's configs diverge; the comparing
half lives in each copy's repo, and this is the half that keeps the canonical
side honest.
"""

from __future__ import annotations

import json
import os
import random
import re
import string

import pytest

import krepis.session_dlp as session_dlp

try:  # 3.11+
    import tomllib
except ImportError:  # pragma: no cover - 3.9/3.10 legs; tomli ships with pytest there
    import tomli as tomllib  # type: ignore[no-redef]

#: Exactly the reconciled rule set. Adding or removing a rule is a change to a
#: security control in four repos at once, so it has to be made here on purpose.
RECONCILED_RULE_IDS = (
    "nousergon-telegram-bot-token",
    "nousergon-openrouter-api-key",
    "nousergon-anthropic-api-key",
    "nousergon-deepseek-api-key",
    "nousergon-aws-session-token",
    "nousergon-ibkr-config-credential-line",
    "nousergon-ssm-secure-string-response",
    "nousergon-aws-access-key-id",
)


def _packaged(name: str) -> str:
    d = session_dlp.packaged_gitleaks_dir()
    assert d is not None, "packaged dlp_config/ not readable"
    return os.path.join(d, name)


@pytest.fixture(scope="module")
def custom() -> dict:
    with open(_packaged("gitleaks-custom.toml"), "rb") as fh:
        return tomllib.load(fh)


@pytest.fixture(scope="module")
def rules(custom) -> dict:
    return {r["id"]: r for r in custom["rules"]}


class TestReconciledRuleset:
    def test_both_files_parse(self):
        for name in ("gitleaks-custom.toml", "gitleaks-egress.toml"):
            with open(_packaged(name), "rb") as fh:
                tomllib.load(fh)

    def test_custom_extends_the_stock_ruleset(self, custom):
        assert custom["extend"] == {"useDefault": True}

    def test_rule_set_is_exactly_the_reconciled_one(self, custom):
        assert tuple(r["id"] for r in custom["rules"]) == RECONCILED_RULE_IDS

    def test_telegram_has_no_trailing_word_boundary(self, rules):
        """claude-code-config-I112: ``-`` is a non-word char, so a trailing
        ``\\b`` can never follow a token ending in it. RE2 has no lookahead;
        dropping the anchor is the fix."""
        regex = rules["nousergon-telegram-bot-token"]["regex"]
        assert not regex.endswith(r"\b"), regex
        assert re.fullmatch(regex, "123456789:" + "a" * 34 + "-")

    def test_anthropic_requires_a_short_open_ended_body(self, rules):
        """I4946 + the I3012 engine defect, together.

        ``{8,}`` keeps a bare prefix in prose from firing (I4946). It must stay
        OPEN-ENDED and SHORT: a long bounded quantifier (the 93-char form) is
        measured to silently not fire under ``useDefault = true`` on gitleaks
        8.30.1. A "cleanup" that tightens it disables the rule.
        """
        regex = rules["nousergon-anthropic-api-key"]["regex"]
        assert regex.endswith("{8,}"), regex
        assert not re.search(r"\{\d{2,}(,\d*)?\}", regex), (
            f"long quantifier in the Anthropic rule: {regex}"
        )

    def test_ibkr_excludes_placeholders(self, rules):
        regex = rules["nousergon-ibkr-config-credential-line"]["regex"]
        assert re.search(regex, "IbPassword = hunter2")
        assert not re.search(regex, "IbPassword = <value>")

    def test_aws_key_id_rule_is_ungated_with_a_match_target_allowlist(self, rules):
        """No entropy floor -- that floor is exactly what makes the stock rule
        fire on 5-25% of key ids. And ``regexTarget = "match"`` is the only
        allowlist form gitleaks 8.30.1 honours here (measured 2026-07-29)."""
        rule = rules["nousergon-aws-access-key-id"]
        assert "entropy" not in rule and "keywords" not in rule
        (allow,) = rule["allowlists"]
        assert allow["regexTarget"] == "match"
        assert any("IOSFODNN7EXAMPLE" in r for r in allow["regexes"])


# ── End to end on the real engine ───────────────────────────────────────────
#
# Randomised, like claude-code-config's test_gitleaks_class_coverage.py: the
# Telegram defect was found precisely because that test randomises, and one
# fixed string would have hidden it. Values are assembled from fragments so
# this file carries no secret-shaped literal for the repo's own gitleaks CI.

_B62 = string.ascii_letters + string.digits
_UPPER = string.ascii_uppercase + string.digits
_ANTHROPIC = "sk-" + "ant-" + "api03-"
_AKIA = "AK" + "IA"


def _r(n: int, alpha: str = _B62) -> str:
    return "".join(random.choice(alpha) for _ in range(n))


@pytest.fixture
def packaged_scan(monkeypatch):
    packaged = session_dlp.packaged_gitleaks_dir()
    monkeypatch.setattr(session_dlp, "GITLEAKS_DIR", packaged)
    monkeypatch.setattr(
        session_dlp, "GITLEAKS_CONFIG", os.path.join(packaged, "gitleaks-egress.toml")
    )
    monkeypatch.delenv("KREPIS_DLP_DISABLED", raising=False)
    session_dlp._cache._config_sig = None
    session_dlp._cache._clean.clear()

    def scan(text: str):
        body = json.dumps({"messages": [{"role": "user", "content": text}]}).encode()
        verdict, reason, _ms, _ratio = session_dlp.scan_request(body)
        return verdict, reason

    return scan


_POSITIVES = {
    "telegram token ending in '-'":
        lambda: f"{random.randint(10**8, 10**10)}:{_r(34, _B62 + '_-')}-",
    "aws key id": lambda: f"AWS_ACCESS_KEY_ID={_AKIA}{_r(16, _UPPER)}",
    "aws key id in a presigned url": lambda: (
        f"https://b.s3.amazonaws.com/k?X-Amz-Credential={_AKIA}{_r(16, _UPPER)}"
        "%2F20261015%2Fus-east-1%2Fs3%2Faws4_request"),
    "anthropic key at the minimum body": lambda: f"x {_ANTHROPIC}{_r(8)} y",
    "anthropic key at a real length": lambda: f"{_ANTHROPIC}{_r(93)}AA",
    "ibkr credential line": lambda: f"IbPassword = {random.choice(_B62)}{_r(12)}",
}

_NEGATIVES = {
    "anthropic prefix in prose": lambda: f'the prefix is "{_ANTHROPIC}" then a body',
    "anthropic prefix with a short body": lambda: f"see {_ANTHROPIC}{_r(5)} here",
    "ibkr placeholder": lambda: "FIXPassword = <your-password>",
    "aws published example key id": lambda: f"id {_AKIA}IOSFODNN7EXAMPLE in the docs",
}


@pytest.mark.skipif(
    not session_dlp.shutil.which("gitleaks"),
    reason="gitleaks binary not on PATH (test.yml installs 8.30.1)",
)
class TestReconciledRulesOnTheRealEngine:
    @pytest.mark.parametrize("label", sorted(_POSITIVES))
    def test_credential_is_blocked_every_time(self, packaged_scan, label):
        for _ in range(5):
            verdict, reason = packaged_scan(_POSITIVES[label]())
            assert verdict == session_dlp.DLP_BLOCKED, f"{label}: {verdict} {reason}"

    @pytest.mark.parametrize("label", sorted(_NEGATIVES))
    def test_description_of_a_secret_is_not_blocked(self, packaged_scan, label):
        for _ in range(3):
            verdict, reason = packaged_scan(_NEGATIVES[label]())
            assert verdict == session_dlp.DLP_OK, f"{label}: {verdict} {reason}"

    def test_the_rule_file_does_not_match_itself(self, packaged_scan):
        """I4946's sharpest form: an agent reading a diff of this file must not
        be blocked by it. Both of the fleet copy's false positives were
        self-matches."""
        with open(_packaged("gitleaks-custom.toml"), encoding="utf-8") as fh:
            verdict, reason = packaged_scan(fh.read())
        assert verdict == session_dlp.DLP_OK, reason
