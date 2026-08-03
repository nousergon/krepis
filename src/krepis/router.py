"""Model-group Router — single source of truth for LLM model routing.

Reads :file:`LLM_MODEL_REGISTRY.yaml` and builds a :class:`litellm.Router`
with fallback chains. Exposes a CLI for shell scripts and GHA workflows
to resolve model groups to the first healthy model.

Usage::

    # CLI — resolve a group to the best available model (bash/GHA)
    python3 -m krepis.router resolve low
    # → deepseek-v4-flash

    python3 -m krepis.router resolve ultra
    # → moonshotai/kimi-k3

    # List all groups
    python3 -m krepis.router groups

    # Python API
    from krepis.router import get_router, resolve_group
    router = get_router()
    model = resolve_group("high")  # → "deepseek-v4-pro"

Registry lookup
    By default reads ``$LLM_MODEL_REGISTRY_PATH`` (env var).  If unset,
    walks up from *cwd* looking for
    ``private-docs/LLM_MODEL_REGISTRY.yaml`` (alpha-engine-config path).
    Failing that, falls back to the hardcoded model list in
    :func:`_builtin_model_list` — the same list that was previously
    embedded in :func:`krepis.llm._get_router`.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_MAX_WALK_DEPTH = 8

# ── Resolution contract version ──────────────────────────────────────────
# The dict returned by resolve_group_structured() / `resolve <group> --json`
# is a CROSS-REPO CONTRACT with consumers in alpha-engine-config (groom
# driver, groom_run.sh, disposition audit, reviewed-merge sweep) and
# claude-code-config (the clauder wrapper).  It is versioned, schema'd
# (resolve_schema.json), and evolved ADDITIVELY.
#
# Field renames are the recurring failure mode here: `resolve-group` -> `resolve`
# broke the wrapper (2026-07-24), and `anthropic_base_url` -> `api_base_url`
# broke all four alpha-engine-config consumers (alpha-engine-config-I4453) --
# two of which degraded to an EMPTY base URL and targeted api.anthropic.com.
#
# Rule (model-router-policy R19): emit BOTH names for one release, migrate
# consumers, then remove.  Never a same-commit rename.
RESOLVE_SCHEMA_VERSION = 2

# Fields kept only to avoid breaking not-yet-migrated consumers.  Each entry
# is (deprecated_name, current_name, remove_after_version).
_DEPRECATED_RESOLVE_ALIASES = (
    ("anthropic_base_url", "api_base_url", 3),
    # supports_automatic_prefix_caching was the per-provider path's name for
    # the same field the LiteLLM path already emits as automatic_prefix_caching.
    # The wrapper reads the shorter name, so on the per-provider path it
    # silently defaulted to False (krepis-I100). Both names are now emitted in
    # both paths; remove the alias once every consumer reads the canonical name.
    ("supports_automatic_prefix_caching", "automatic_prefix_caching", 4),
)


def _with_compat_aliases(info: dict) -> dict:
    """Add deprecated field aliases to a resolve-contract dict.

    Keeps consumers that have not yet migrated working through one release,
    per the additive-then-remove rule.  Drop an alias by removing its row
    from :data:`_DEPRECATED_RESOLVE_ALIASES` once every consumer is migrated
    and its contract test asserts the new name.
    """
    for old, new, _remove_after in _DEPRECATED_RESOLVE_ALIASES:
        if new in info and old not in info:
            info[old] = info[new]
    return info

_egress_placeholder = "unused-placeholder-see-key-isolation-config3007"

# LiteLLM proxy — the central model router on the dashboard box.  When healthy,
# this is the PREFERRED endpoint for the Claude CLI because it:
#   1. Translates between wire formats (CLI format ↔ provider-native format),
#      making every provider CLI-compatible regardless of its native API format.
#   2. Handles fallback chains with cooldown via litellm.Router.
#   3. Passes through prompt-caching hints and translates them to
#      provider-specific caching mechanisms where applicable.
#   4. Routes all traffic through egress proxies for DLP scanning.
LITELLM_PROXY_URL = "http://127.0.0.1:8980"

# Env-var overrides for every CLI-compatible endpoint (config#4923).
# Each resolves at resolution time from (env var → default constant), so a
# box-level bootstrap can export the override alongside its --port without
# krepis needing a config file or code change.
_ENV_OVERRIDE_MAP: dict[tuple[str, str | None], str] = {
    ("egress_proxy", "deepseek"): "KREPIS_DEEPSEEK_EGRESS_URL",
    ("openrouter", None):         "KREPIS_OPENROUTER_API_URL",
    ("litellm_proxy", None):      "KREPIS_LITELLM_PROXY_URL",
}

# ── Execution context (model-router-policy R28/R29) ──────────────────────
# A DIRECT PROVIDER endpoint is not a global fact.  `http://127.0.0.1:8990` is
# true on the laptop and on the dashboard box and meaningless inside a Lambda
# container.  `reachable_from` scopes those, and only those.
#
# It does NOT scope the litellm_proxy route.  model-router-policy §3.4a R27a is
# categorical: the router is addressed by (url, credential) and reaching it may
# not depend on host, VPC, subnet, security group or private IP.  So the proxy
# path below is gated on its HEALTH PROBE and never on context — its
# unavailability is an outage, never a reachability fact about the caller.
#
# The names say WHERE CODE RUNS, never how it is attached.  An earlier draft of
# this constant read `lambda_vpc`, and a context name asserting a network
# attachment is an invitation to the R27a violation that cost a 2h20m
# fleet-wide SSM outage on 2026-08-03 (nous-ergon-ops-I417): the endpoint
# created to give one VPC-attached Lambda a private path to SSM carried a
# VPC-wide private-DNS override behind a security group that blocked the VPC.
#
# R29: the context is a DECLARED input, never inferred from hostname, the
# metadata service, or the presence of an env var.  Inference is what makes a
# mis-resolution look like a health failure.
EXEC_CONTEXT_LAPTOP = "laptop"
EXEC_CONTEXT_EC2 = "ec2"
EXEC_CONTEXT_LAMBDA = "lambda"
EXEC_CONTEXTS = (EXEC_CONTEXT_LAPTOP, EXEC_CONTEXT_EC2, EXEC_CONTEXT_LAMBDA)
DEFAULT_EXEC_CONTEXT = EXEC_CONTEXT_LAPTOP

# ── Wire formats ─────────────────────────────────────────────────────────
# An endpoint speaks one wire format.  The Claude CLI speaks the Anthropic
# Messages format; programmatic callers built on the openai transport speak
# the OpenAI format.  The same provider is frequently reachable at BOTH, on
# different ports (DeepSeek: 8971 Anthropic-wire, 8990 OpenAI-wire), which is
# the fact the old (route, provider) -> URL table was encoding without saying
# so — and the reason a caller could be handed an endpoint in the wrong format.
WIRE_ANTHROPIC = "anthropic"
WIRE_OPENAI = "openai"
WIRE_FORMATS = (WIRE_ANTHROPIC, WIRE_OPENAI)
DEFAULT_WIRE = WIRE_ANTHROPIC

# LEGACY — remove once every registry entry declares an `endpoints` block.
#
# This is the table model-router-policy R7 forbids ("no hardcoded hosts or
# ports anywhere in library code") and R29 names directly: a resolver that
# drops an entry because its own code lacks a row for that provider has
# invented a routing fact at layer 3.  It is retained ONLY as the migration
# shim for registry entries that predate `endpoints`, is consulted only when
# an entry carries no `endpoints` block, and warns on every use naming the
# entry that still needs migrating.
#
# Removal condition: `LLM_MODEL_REGISTRY.yaml` declares `endpoints` on every
# route-bearing row and the registry validator requires it.  Tracked as
# alpha-engine-config-I6186.
_LEGACY_CLI_ENDPOINTS: dict[tuple[str, str | None], str] = {
    ("egress_proxy", "deepseek"): "http://127.0.0.1:8971",
    ("openrouter", None):         "https://openrouter.ai/api",
    ("direct", "anthropic"):      "",
    ("litellm_proxy", None):      LITELLM_PROXY_URL,
}


def _resolve_litellm_proxy_url() -> str:
    """LiteLLM proxy URL, from ``KREPIS_LITELLM_PROXY_URL`` env var or the
    module-level :data:`LITELLM_PROXY_URL` constant.

    This is the URL the LiteLLM health probe targets and the ``api_base_url``
    returned on the LiteLLM route — keeping the two in sync with one source
    of truth (config#4923)."""
    return os.environ.get("KREPIS_LITELLM_PROXY_URL", LITELLM_PROXY_URL)


def _resolve_exec_context(exec_context: str | None = None) -> str:
    """Return the caller's declared execution context (R29).

    Resolution order: explicit argument → ``KREPIS_EXEC_CONTEXT`` env var →
    :data:`DEFAULT_EXEC_CONTEXT`.

    The value is DECLARED, never inferred.  krepis does not read the EC2
    metadata service, the Lambda runtime env vars, or the hostname to guess
    where it is running: a wrong guess produces a resolution that looks like a
    health failure, which is the exact confusion R29 exists to prevent.

    Raises :exc:`ValueError` on an unrecognised context rather than falling
    back to the default — an unknown context means the caller believes it is
    somewhere krepis has no reachability facts about, and resolving anyway
    would hand it an endpoint chosen on a vocabulary mismatch.
    """
    ctx = exec_context or os.environ.get("KREPIS_EXEC_CONTEXT") or DEFAULT_EXEC_CONTEXT
    ctx = ctx.strip()
    if ctx not in EXEC_CONTEXTS:
        raise ValueError(
            f"Unknown execution context {ctx!r}. Declared contexts are "
            f"{list(EXEC_CONTEXTS)} — they name where code runs, never how it "
            "is attached (model-router-policy R28). Set KREPIS_EXEC_CONTEXT to "
            "one of them, or add the new context to this constant AND to the "
            "registry's reachable_from vocabulary. A context name encoding a "
            "network posture (e.g. 'lambda_vpc') is a defect: reaching the "
            "router may not depend on network position (§3.4a R27a)."
        )
    return ctx


def _entry_reachable_from(entry: dict, exec_context: str) -> bool:
    """Whether *entry* declares itself reachable from *exec_context* (R28).

    An entry with no ``reachable_from`` key is **not** reachable from anywhere.

    This branch used to be permissive — an undeclared entry was treated as
    reachable from every context, with a warning — as the R19
    additive-then-remove migration position, to be removed "once the validator
    is enforcing."  The validator has been enforcing since #6203
    (``scripts/validate_llm_model_registry.py`` fails a pull request on a
    route-bearing row with no ``reachable_from``, asserted by
    ``test_missing_reachable_from_fails``), so this is that removal.

    It is not bookkeeping.  On 2026-08-03 the copy of the registry the Director
    Lambda actually reads — an S3 object published by hand, one day behind the
    repo — still had no ``reachable_from`` on the ``ultra`` chain.  The
    permissive branch read that silence as universal reachability and served
    ``glm-5.2`` at ``openrouter.ai`` from a Lambda, DLP-unscanned, while logging
    a healthy route (alpha-engine-config-I6183, model-router-policy R26).  A
    default that turns *absence of a declaration* into *permission* converts a
    stale artifact into a policy breach, silently, which is what R20 (fail
    closed) forbids.

    A skipped entry is recorded in ``skipped_entries`` with the reason naming
    the missing declaration, so this surfaces as a diagnosable resolution
    failure rather than an unexplained one — and, per R20, resolution raises
    only when nothing in the chain can serve.
    """
    declared = entry.get("reachable_from")
    if declared is None:
        logger.error(
            "Registry entry %r declares no reachable_from and is therefore "
            "UNREACHABLE from every context. model-router-policy R28 makes the "
            "field required and the registry validator enforces it, so this "
            "entry came from a registry copy that is stale or hand-written. "
            "Fix the registry — do not read the omission as permission.",
            entry.get("id", "<unknown>"),
        )
        return False
    return exec_context in declared


def _entry_endpoint(entry: dict, wire: str) -> str:
    """Return *entry*'s declared base URL for the *wire* format.

    The registry owns endpoints (model-router-policy §2 layer 1), so this
    reads them rather than deciding them:

    1. Per-route env-var override (``_ENV_OVERRIDE_MAP``) — a box-level
       bootstrap exporting its own port, which is a deployment fact rather
       than a routing one.
    2. ``entry["endpoints"][wire]`` — the declared endpoint for this format.
    3. ``entry["api_base"]`` when *wire* is the OpenAI format — the pre-
       ``endpoints`` spelling of the same fact.
    4. :data:`_LEGACY_CLI_ENDPOINTS` — the migration shim, warned on.

    Raises :exc:`ValueError` if the entry declares no endpoint for *wire*.
    The caller records that in ``skipped_entries`` and moves on, so "this
    provider does not speak your wire format" is a stated registry fact
    rather than a silent absence from a dict.
    """
    route = entry.get("route", "")
    provider = entry.get("provider", "")

    # 1. Env override — deployment fact, wins over the declared default.
    for key in ((route, provider), (route, None)):
        env_var = _ENV_OVERRIDE_MAP.get(key)
        if env_var:
            override = os.environ.get(env_var)
            if override:
                return override

    # 2. Declared per-wire endpoint.
    endpoints = entry.get("endpoints")
    if isinstance(endpoints, dict):
        if wire in endpoints:
            return endpoints[wire] or ""
        raise ValueError(
            f"registry entry {entry.get('id', '<unknown>')!r} declares no "
            f"{wire!r}-wire endpoint (declared: {sorted(endpoints)})"
        )

    # 3. Pre-`endpoints` spelling: bare `api_base` is the OpenAI-wire endpoint.
    if wire == WIRE_OPENAI and entry.get("api_base"):
        return entry["api_base"]

    # 4. Migration shim.
    for key in ((route, provider), (route, None)):
        if key in _LEGACY_CLI_ENDPOINTS:
            logger.warning(
                "Registry entry %r has no `endpoints` block; falling back to "
                "krepis' legacy hardcoded endpoint table, which "
                "model-router-policy R7 forbids. Declare "
                "endpoints.{anthropic,openai} on this row "
                "(alpha-engine-config-I6186).",
                entry.get("id", "<unknown>"),
            )
            return _LEGACY_CLI_ENDPOINTS[key]

    raise ValueError(
        f"registry entry {entry.get('id', '<unknown>')!r} "
        f"(route={route!r}, provider={provider!r}) declares no endpoint for "
        f"the {wire!r} wire format, and no legacy default applies"
    )


def _resolve_cli_endpoint(route: str, provider: str | None) -> str:
    """Resolve a CLI-compatible endpoint for *(route, provider)* at call time.

    DEPRECATED — superseded by :func:`_entry_endpoint`, which reads the
    endpoint the registry declares instead of choosing one from a table in
    library code.  Retained because the ``(route, provider)`` shape is what
    the legacy shim is keyed on and existing tests bind to this name.

    Resolution order:
    1. Per-route env var (``_ENV_OVERRIDE_MAP`` → ``os.environ``)
    2. Module-level :data:`_LEGACY_CLI_ENDPOINTS` default

    Returns the endpoint URL or ``""`` for provider-default routes (``direct``).
    Raises :exc:`ValueError` if the combination is not CLI-compatible.
    """
    # 1. Check env override by exact (route, provider) key
    precise_key = (route, provider)
    if provider is not None and precise_key in _ENV_OVERRIDE_MAP:
        override = os.environ.get(_ENV_OVERRIDE_MAP[precise_key])
        if override:
            return override

    # 2. Check env override by (route, None) key (catch-all for the route)
    catchall_key = (route, None)
    if catchall_key in _ENV_OVERRIDE_MAP:
        override = os.environ.get(_ENV_OVERRIDE_MAP[catchall_key])
        if override:
            return override

    # 3. Check exact match in the default map
    if precise_key in _LEGACY_CLI_ENDPOINTS:
        return _LEGACY_CLI_ENDPOINTS[precise_key]

    # 4. Check catch-all by route
    if catchall_key in _LEGACY_CLI_ENDPOINTS:
        return _LEGACY_CLI_ENDPOINTS[catchall_key]

    raise ValueError(
        f"({route!r}, {provider!r}) is not a CLI-compatible endpoint "
        "and cannot be resolved"
    )


def _probe_egress_proxy(url: str, timeout: int = 3) -> bool:
    """Probe an egress proxy's health endpoint.

    Returns ``True`` if the proxy responds with ``200`` at ``/__proxy_health__``.
    Used by :func:`_cli_endpoint_for` to fail at resolve time rather than
    returning a dead endpoint (config#4923).

    Pure / no side-effects beyond the network call.  Tested by mocking
    ``http.client``.
    """
    if not url:
        return False  # empty URL (direct route) is not an egress proxy
    try:
        from urllib.parse import urlparse

        parsed = urlparse(url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 8971

        import http.client as _http

        conn = _http.HTTPConnection(host, port, timeout=timeout)
        conn.request("GET", "/__proxy_health__")
        resp = conn.getresponse()
        resp.read()
        conn.close()
        return resp.status == 200
    except Exception:
        return False


# ── LiteLLM master key resolution ────────────────────────────────────────
# Same resolution order as the LiteLLM proxy shim and the clauder script:
#   1. LITELLM_MASTER_KEY env var
#   2. secrets.env file (LITELLM_MASTER_KEY=...)
#   3. AWS SSM /symposion/LITELLM_MASTER_KEY
# Returns the key string, or None if unresolvable.

def _resolve_litellm_master_key() -> Optional[str]:
    import os as _os

    # 1. Env var
    _key = _os.environ.get("LITELLM_MASTER_KEY", "").strip()
    if _key:
        return _key

    # 2. secrets.env (same path conventions as the shim)
    _secrets_paths = [
        _os.path.expanduser("~/Development/.llm-routing/secrets.env"),
        _os.path.expanduser("~/.llm-routing/secrets.env"),
    ]
    for _sp in _secrets_paths:
        try:
            with open(_sp) as _sf:
                for _line in _sf:
                    _line = _line.strip()
                    if _line.startswith("LITELLM_MASTER_KEY="):
                        _val = _line.split("=", 1)[1].strip().strip('"').strip("'")
                        if _val:
                            return _val
                        break
        except (OSError, IOError):
            continue

    # 3. AWS SSM
    return _litellm_master_key_from_ssm()


def _litellm_master_key_from_ssm() -> Optional[str]:
    """Last-resort lookup of the master key from SSM.

    Split out of ``_resolve_litellm_master_key`` so tests can neutralise the one
    leg that reaches outside the process. Until 2026-07-30 this ran inline and
    the laptop had no working AWS credentials, so six tests in test_router.py
    passed only because the call always failed — they were asserting the state
    of the machine, not the behaviour of the code. Giving the laptop a machine
    identity that actually works turned all six red at once.
    """
    try:
        import subprocess as _sp
        _result = _sp.run(
            ["aws", "ssm", "get-parameter",
             "--name", "/symposion/LITELLM_MASTER_KEY",
             "--with-decryption", "--region", "us-east-1",
             "--profile", "ne-laptop-daemon",
             "--query", "Parameter.Value", "--output", "text"],
            capture_output=True, text=True, timeout=5,
        )
        if _result.returncode == 0:
            _val = _result.stdout.strip()
            if _val and _val != "None":
                return _val
    except Exception:
        pass

    return None


# ── LiteLLM config staleness check (RETIRED) ────────────────────────────
# alpha-engine-config-I4452: this mtime-glob heuristic is RETIRED.  It
# inferred staleness from filesystem mtimes over a /tmp glob, ran only in
# a laptop CLI path, and nothing on any box called it.  The authoritative
# provenance-based drift checker lives in alpha-engine-config:
#
#   scripts/check_router_config_provenance.py
#
# which compares a CONTENT DIGEST (sha256) stamped into the generated config
# against a fresh hash of the live registry — a fact, not an inference.
#
# The auto-reconcile script (scripts/reconcile_litellm_config.py) runs on
# a 10-minute timer and restarts the router on digest mismatch, satisfying
# the 15-minute propagation SLO (R10) with no human step.
#
# The function has been removed; the caller (get_router) now skips this
# check.  See alpha-engine-config-PR4773 and I4452 for details.

# ── registry file discovery ─────────────────────────────────────────────

def _find_registry() -> Optional[Path]:
    """Resolve the registry file path.

    Lookup order:
    1. ``$LLM_MODEL_REGISTRY_PATH`` (explicit override)
    2. AWS AppConfig — when ``KREPIS_APPCONFIG_APPLICATION`` is set, polls
       AppConfig for the registry content and caches it to a local temp file
       (config-I5199 AppConfig resolution path per Brian's 2026-07-28 ruling).
       Cached with the AppConfig poll interval as TTL; falls through to disk
       on any error (AppConfig unreachable, no config deployed, etc.).
    3. Walk up from cwd for ``private-docs/LLM_MODEL_REGISTRY.yaml``
       (alpha-engine-config convention)

    Returns ``None`` if none is found — the caller (:func:`get_router`)
    raises :exc:`FileNotFoundError` rather than falling back to a stale
    duplicate.  There is exactly ONE source of truth for model groupings,
    and it lives in ``alpha-engine-config/private-docs/``.
    """
    env_path = os.environ.get("LLM_MODEL_REGISTRY_PATH")
    if env_path:
        p = Path(env_path)
        return p if p.exists() else None

    # ── Tier 2: AppConfig (config-I4799, Brian's 2026-07-28 Option-A ruling) ─
    # AppConfig distributes the class→model mapping to consumers that cannot
    # reach the private alpha-engine-config repo on disk (public repos, Lambda
    # runtimes).  Opt-in: only activates when KREPIS_APPCONFIG_APPLICATION is
    # set, so environments that have a local checkout are unaffected.
    appconfig_path = _find_registry_from_appconfig()
    if appconfig_path:
        return appconfig_path

    # Walk up from cwd looking for alpha-engine-config/private-docs/
    cwd = Path.cwd().resolve()
    for _ in range(_MAX_WALK_DEPTH):
        candidate = cwd / "private-docs" / "LLM_MODEL_REGISTRY.yaml"
        if candidate.exists():
            return candidate
        # Also check the nested alpha-engine-config path
        candidate2 = cwd / "alpha-engine-config" / "private-docs" / "LLM_MODEL_REGISTRY.yaml"
        if candidate2.exists():
            return candidate2
        parent = cwd.parent
        if parent == cwd:
            break
        cwd = parent

    return None


# ── AppConfig registry resolution (config-I4799) ──────────────────────────

# AppConfig env-var opt-in — these MUST all be set for the AppConfig path to
# activate.  Environments with a local alpha-engine-config checkout are
# unaffected (the filesystem walk wins for them; AppConfig is only tried
# when the walk-up fails).
_APPCONFIG_ENV_APPLICATION = "KREPIS_APPCONFIG_APPLICATION"
_APPCONFIG_ENV_CONFIG_PROFILE = "KREPIS_APPCONFIG_CONFIG_PROFILE"
_APPCONFIG_ENV_ENVIRONMENT = "KREPIS_APPCONFIG_ENVIRONMENT"
_APPCONFIG_DEFAULT_CLIENT_ID = "krepis"

# Cache: the temp file path and its next-poll time (monotonic seconds).
# Thread-safe behind _router_lock (reused from the Router singleton).
_appconfig_cached_path: Optional[Path] = None
_appconfig_next_poll_s: float = 0.0
# Minimum poll interval enforced by AppConfig (the service returns a
# RequiredMinimumPollIntervalInSeconds on start_configuration_session).
_APPCONFIG_MIN_POLL_SECONDS = 15
# Fallback poll interval when the service doesn't specify one.
_APPCONFIG_DEFAULT_POLL_SECONDS = 300  # 5 min


def _find_registry_from_appconfig() -> Optional[Path]:
    """Poll AWS AppConfig for the LLM model registry and cache it locally.

    Returns a :class:`Path` to the cached registry file on success, or
    ``None`` when AppConfig is not configured / not reachable / has no
    deployed configuration — the caller falls through to the filesystem walk.

    Thread-safe: guarded by the module-level ``_router_lock`` so concurrent
    callers don't race on the cache file.  Errors are logged and swallowed —
    the AppConfig path is a best-effort distribution mechanism, and a
    transient AppConfig outage must not prevent a run that has a local
    checkout from working (the filesystem walk is the fallback).
    """
    global _appconfig_cached_path, _appconfig_next_poll_s

    app_id = os.environ.get(_APPCONFIG_ENV_APPLICATION)
    if not app_id:
        return None  # Opt-in not set — skip AppConfig entirely

    config_profile = os.environ.get(
        _APPCONFIG_ENV_CONFIG_PROFILE, "llm-model-registry"
    )
    environment = os.environ.get(
        _APPCONFIG_ENV_ENVIRONMENT, "production"
    )

    # ── Cache hit — return cached file if still fresh ────────────────────
    import time as _time
    from threading import Lock as _Lock

    _lock = _router_lock or _Lock()
    with _lock:
        now = _time.monotonic()
        if (
            _appconfig_cached_path is not None
            and _appconfig_cached_path.exists()
            and now < _appconfig_next_poll_s
        ):
            return _appconfig_cached_path

        # ── Cache miss or expired — poll AppConfig ───────────────────────
        try:
            import boto3 as _boto3
            client = _boto3.client("appconfigdata", region_name=os.environ.get("AWS_REGION", "us-east-1"))

            # Start a configuration session.
            session = client.start_configuration_session(
                ApplicationIdentifier=app_id,
                ConfigurationProfileIdentifier=config_profile,
                EnvironmentIdentifier=environment,
                RequiredMinimumPollIntervalInSeconds=_APPCONFIG_MIN_POLL_SECONDS,
            )
            token = session["InitialConfigurationToken"]

            # Fetch the latest configuration.
            response = client.get_latest_configuration(
                ConfigurationToken=token,
            )
            content = response["Configuration"].read()
            if not content:
                logger.debug(
                    "AppConfig returned empty configuration for %s/%s/%s",
                    app_id, config_profile, environment,
                )
                return None

            poll_interval = response.get(
                "NextPollIntervalInSeconds", _APPCONFIG_DEFAULT_POLL_SECONDS
            )

            # Write to a stable temp file so callers only need a Path.
            import tempfile as _tempfile
            cache_dir = Path(_tempfile.gettempdir()) / "krepis-registry"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / "LLM_MODEL_REGISTRY.yaml"
            cache_file.write_bytes(content)

            _appconfig_cached_path = cache_file
            _appconfig_next_poll_s = now + max(
                poll_interval, _APPCONFIG_MIN_POLL_SECONDS
            )

            logger.info(
                "loaded LLM_MODEL_REGISTRY.yaml from AppConfig "
                "(%s/%s/%s), cached at %s, next poll in %ss",
                app_id, config_profile, environment,
                cache_file, poll_interval,
            )
            return cache_file

        except Exception:
            logger.debug(
                "AppConfig registry resolution failed — "
                "falling through to filesystem walk",
                exc_info=True,
            )
            # If we had a previously cached file that still exists, keep
            # using it past its TTL rather than returning None — a stale
            # registry beats no registry.
            if (
                _appconfig_cached_path is not None
                and _appconfig_cached_path.exists()
            ):
                logger.warning(
                    "AppConfig unreachable; serving stale cached registry "
                    "from %s (last poll was %ss ago)",
                    _appconfig_cached_path,
                    now - (_appconfig_next_poll_s - poll_interval)
                    if poll_interval
                    else "unknown",
                )
                return _appconfig_cached_path
            return None


# ── CLI endpoint helpers ──────────────────────────────────────────────────

def _caching_flags(entry: dict) -> tuple[bool, bool]:
    """Return ``(explicit_breakpoints, automatic_prefix)`` for one model entry.

    The two caching mechanisms are **mutually exclusive** and impose opposite
    client obligations:

    * ``explicit_breakpoints`` — the client must mark cacheable segments with
      ``cache_control: {"type": "ephemeral"}``. Anthropic-wire only.
    * ``automatic_prefix`` — the server caches repeated prefixes
      transparently. The client must send **no** markers; emitting them at a
      provider that does not accept the field is at best a wasted breakpoint
      and at worst a 400.

    This is the ONE place either flag is read, so a mis-declared entry
    produces one consistent answer everywhere rather than a different one per
    call site.

    Forward-compatible with the registry's move to a single
    ``capabilities.caching_mechanism`` enum
    (``explicit_breakpoint`` | ``automatic_prefix`` | ``none``), which makes
    the invalid both-true state unrepresentable rather than merely forbidden
    (alpha-engine-config-I4463). The enum wins when present; the two legacy
    booleans are the fallback until every registry entry carries it.

    An entry declaring BOTH legacy booleans is invalid. Rather than silently
    picking one, this resolves to ``automatic_prefix`` — the safe direction:
    a provider that caches transparently loses nothing by receiving no
    markers, whereas sending markers to one that rejects unknown fields is an
    outage. The registry validator rejects the state at PR time; this is the
    runtime backstop for a registry that predates it.
    """
    caps = entry.get("capabilities") or {}

    mechanism = caps.get("caching_mechanism")
    if mechanism:
        return mechanism == "explicit_breakpoint", mechanism == "automatic_prefix"

    explicit = bool(caps.get("prompt_caching", False))
    automatic = bool(caps.get("automatic_prefix_caching", False))
    if explicit and automatic:
        logger.warning(
            "model %r declares BOTH prompt_caching and automatic_prefix_caching "
            "— these are mutually exclusive; treating it as automatic_prefix "
            "(no client markers), the non-breaking direction",
            entry.get("id", "?"),
        )
        return False, True
    return explicit, automatic


def _cli_endpoint_for(entry: dict) -> str:
    """Return the CLI-compatible base URL for a registry model entry.

    Resolves the endpoint from env var overrides first (config#4923), then
    falls back to the legacy ``_LEGACY_CLI_ENDPOINTS`` literal.

    For ``egress_proxy`` routes: probes the resolved proxy at
    ``/__proxy_health__`` before returning.  An unreachable proxy raises
    :exc:`ValueError` with a message naming the URL and the env override var
    — so the caller falls through to the next model (same pattern as the
    LiteLLM health check) rather than returning a dead endpoint that would
    surface later as an opaque ConnectionRefused inside an LLM agent.

    Raises :exc:`ValueError` if this entry's route+provider cannot serve
    the wire format the Claude CLI expects (e.g. Gemini/xAI/Moonshot/Zhipu
    egress proxies are OpenAI-format only — use LiteLLM for format translation).
    """
    route = entry.get("route", "")
    provider = entry.get("provider", "")

    endpoint = _resolve_cli_endpoint(route, provider)

    # For egress_proxy routes: probe the proxy before returning.
    # A dead proxy means the route is unusable — fail at resolve time
    # rather than returning an endpoint nobody verified (config#4923).
    if route == "egress_proxy" and endpoint:
        env_var_key = _ENV_OVERRIDE_MAP.get((route, provider))
        env_var_hint = f" (override: ${env_var_key})" if env_var_key else ""
        if not _probe_egress_proxy(endpoint):
            raise ValueError(
                f"Egress proxy at {endpoint!r} is not reachable"
                f"{env_var_hint}. "
                "Set the override to a healthy proxy address, or ensure the "
                "local proxy is running on the configured port."
            )

    return endpoint


def _cli_deployment_id(entry: dict) -> str:
    """Return the model string to set as the CLI model identifier.

    * For *egress_proxy* routes: bare model name (e.g. ``deepseek-v4-flash``)
      — the proxy translates to the upstream model ID.
    * For *openrouter* routes: full OpenRouter slug (e.g.
      ``deepseek/deepseek-v4-flash``) — already stored in the registry's
      *model* field.
    * For *direct* routes: canonical provider model ID.
    """
    route = entry.get("route", "")
    model = entry.get("model", "")

    if route == "egress_proxy":
        return model
    elif route == "openrouter":
        return model
    elif route == "direct" and entry.get("provider") == "anthropic":
        return model
    else:
        return model


# ── registry → Router config ─────────────────────────────────────────────

def _parse_registry(path: Path, openrouter_key: str = "") -> tuple[list[dict], list[dict]]:
    """Parse LLM_MODEL_REGISTRY.yaml into litellm Router model_list + fallbacks.

    Returns (model_list, fallbacks) tuples ready for litellm.Router().
    """
    import yaml as _yaml

    with open(path) as f:
        doc = _yaml.safe_load(f)

    model_list: list[dict] = []
    fallbacks: list[dict] = []
    seen_models: set[str] = set()

    groups = doc.get("model_groups", {})
    models = {m["id"]: m for m in doc.get("models", [])}

    for group_name, group_ids in groups.items():
        fallback_chain: list[str] = []
        for i, mid in enumerate(group_ids):
            entry = models.get(mid)
            if entry is None:
                logger.warning("model %r referenced in group %r not found in models list", mid, group_name)
                continue

            # Primary is named after the GROUP ("low", "med", …) so
            # router.completion(model="low") resolves to the first entry.
            # Fallbacks get qualified: "low-gemini-2.5-flash", etc.
            model_name = group_name if i == 0 else f"{group_name}-{mid}"
            litellm_params = _model_to_litellm_params(entry, openrouter_key)

            if model_name not in seen_models:
                model_list.append({"model_name": model_name, "litellm_params": litellm_params})
                seen_models.add(model_name)

            if i > 0:
                fallback_chain.append(model_name)

        if fallback_chain:
            fallbacks.append({group_name: fallback_chain})

    return model_list, fallbacks


def _model_to_litellm_params(entry: dict, openrouter_key: str) -> dict:
    """Convert a registry model entry to litellm params."""
    provider = entry.get("provider", "")
    route = entry.get("route", "")
    model_id = entry.get("model", "")
    params = entry.get("params", {})

    litellm_params: dict = {}

    # Build the litellm model prefix
    if provider == "anthropic":
        litellm_params["model"] = f"anthropic/{model_id}"
        litellm_params["api_key"] = os.environ.get("ANTHROPIC_API_KEY", "")
    elif provider == "openrouter":
        litellm_params["model"] = f"openrouter/{model_id}"
        litellm_params["api_key"] = openrouter_key
    elif route == "egress_proxy":
        litellm_params["model"] = f"openai/{model_id}"
        litellm_params["api_key"] = _egress_placeholder
        # api_base is driven by the registry — no hardcoded provider→port map.
        # Every egress_proxy entry MUST carry an api_base field pointing at its
        # local proxy instance (e.g. "http://127.0.0.1:8972/v1").
        api_base = entry.get("api_base")
        if api_base:
            litellm_params["api_base"] = api_base
        else:
            entry_id = entry.get("id", model_id)
            logger.warning(
                "egress_proxy entry %r missing api_base — "
                "requests will fail without a base URL", entry_id
            )
    else:
        # Unknown route — treat as generic OpenAI-compatible, no proxy.
        litellm_params["model"] = f"openai/{model_id}"
        litellm_params["api_key"] = _egress_placeholder

    # Apply params from registry
    if "max_tokens" in params:
        litellm_params["max_tokens"] = params["max_tokens"]
    reasoning = params.get("reasoning")
    if reasoning:
        if "extra_body" not in litellm_params:
            litellm_params["extra_body"] = {}
        litellm_params["extra_body"]["reasoning"] = reasoning

    # Multi-tenant egress proxy: set X-Upstream-Host header so the single
    # proxy on port 8990 routes to the correct upstream provider.
    upstream_host = entry.get("upstream_host")
    if upstream_host:
        if "extra_headers" not in litellm_params:
            litellm_params["extra_headers"] = {}
        litellm_params["extra_headers"]["X-Upstream-Host"] = upstream_host

    # Apply RPM/TPM from registry if present
    if "rpm" in entry:
        litellm_params["rpm"] = entry["rpm"]
    if "tpm" in entry:
        litellm_params["tpm"] = entry["tpm"]

    return litellm_params


# ── Router singleton ─────────────────────────────────────────────────────

_router: Any = None
_router_lock: Any = None


def get_router() -> Any:
    """Return the module-level Router singleton, built from LLM_MODEL_REGISTRY.yaml.

    Raises :exc:`FileNotFoundError` if no registry file can be found —
    there is no hardcoded fallback.  The single source of truth lives in
    ``alpha-engine-config/private-docs/LLM_MODEL_REGISTRY.yaml``; set
    ``$LLM_MODEL_REGISTRY_PATH`` to point at it explicitly, or run from
    within a repo whose ``private-docs/`` directory contains the file.
    """
    global _router, _router_lock
    if _router is not None:
        return _router

    from threading import Lock as _Lock
    from litellm import Router as _Router

    _router_lock = _Lock()
    with _router_lock:
        if _router is not None:
            return _router

        openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")

        reg_path = _find_registry()
        if not reg_path:
            raise FileNotFoundError(
                "LLM_MODEL_REGISTRY.yaml not found — set "
                "LLM_MODEL_REGISTRY_PATH or run from within a repo "
                "whose private-docs/ directory contains the file.  "
                "The canonical copy lives in "
                "alpha-engine-config/private-docs/LLM_MODEL_REGISTRY.yaml."
            )

        logger.info("building Router from %s", reg_path)
        model_list, fallbacks = _parse_registry(reg_path, openrouter_key)

        _router = _Router(model_list=model_list, fallbacks=fallbacks)
        logger.info("Router initialized: %d models, %d fallback groups", len(model_list), len(fallbacks))
        return _router


# ── group resolution (for shell scripts / GHA) ───────────────────────────

def resolve_group(group: str) -> str:
    """Return the upstream model identifier for *group*'s primary.

    Checks the Router's cooldown state — if the primary model is in
    cooldown (recent failure), returns the first healthy fallback.
    """
    router = get_router()

    # Find the primary model's upstream identifier
    primary_model = _upstream_model_for(router, group)
    if not primary_model:
        # Group not found in model list — return the group name as-is
        return group

    # Check if primary is in cooldown
    deployments = getattr(router, "cooldown_deployments", {})
    primary_key = _deployment_key_for(router, group)
    if primary_key and primary_key in deployments:
        # Primary is in cooldown — try fallbacks
        for fb in router.fallbacks:
            if group in fb:
                for fb_name in fb[group]:
                    fb_key = _deployment_key_for(router, fb_name)
                    if fb_key and fb_key not in deployments:
                        fb_model = _upstream_model_for(router, fb_name)
                        if fb_model:
                            return fb_model
                break  # all fallbacks for this group exhausted

    return primary_model


def group_supports_explicit_cache_breakpoints(group: str) -> bool:
    """True when *group*'s primary honors explicit ``cache_control`` markers.

    The question a client actually needs answered before emitting Anthropic
    ephemeral cache breakpoints. Resolved from the PRIMARY — the entry that
    will serve the request — not from an ``any()`` over the fallback chain,
    which would claim support the served model may not have.

    Returns False when the registry cannot be read: refusing to claim a
    capability we could not verify is the safe direction, since the cost of a
    false negative is a missed cache hit and the cost of a false positive is
    markers sent to a provider that may reject them.
    """
    try:
        reg_path = _find_registry()
        if not reg_path:
            return False
        import yaml as _yaml
        with open(reg_path) as f:
            doc = _yaml.safe_load(f)
        group_ids = (doc.get("model_groups") or {}).get(group) or []
        if not group_ids:
            return False
        models_by_id = {m["id"]: m for m in doc.get("models", [])}
        explicit, _automatic = _caching_flags(models_by_id.get(group_ids[0], {}))
        return explicit
    except Exception:
        logger.warning(
            "could not resolve caching capability for group %r — "
            "assuming no explicit breakpoints", group, exc_info=True,
        )
        return False


def get_group_primary(group: str) -> Optional[str]:
    """Return the litellm model string for *group*'s primary model.

    This is the value that ``resp.model`` will carry when the primary
    (not a fallback) served the request.  Callers compare against it to
    detect whether a fallback was engaged::

        primary = get_group_primary("low")        # "openai/deepseek-v4-flash"
        fallback_used = (resp.model != primary)
    """
    router = get_router()
    for m in router.model_list:
        if m["model_name"] == group:
            return m["litellm_params"]["model"]
    return None


def _upstream_model_for(router: Any, model_name: str) -> Optional[str]:
    """Get the upstream model identifier for a Router model_name."""
    for m in router.model_list:
        if m["model_name"] == model_name:
            return _upstream_model(m["litellm_params"]["model"])
    return None


def _deployment_key_for(router: Any, model_name: str) -> Optional[str]:
    """Find the deployment key for a model in the Router's model list."""
    for m in router.model_list:
        if m["model_name"] == model_name:
            params = m.get("litellm_params", {})
            return params.get("model", "")
    return None


def _upstream_model(litellm_model: str) -> str:
    """Strip the litellm prefix: 'openai/deepseek-v4-flash' → 'deepseek-v4-flash'."""
    parts = litellm_model.split("/", 1)
    return parts[1] if len(parts) > 1 else litellm_model


# ── group resolution (structured, for shell scripts) ────────────────────

def _resolve_group_json(
    group: str,
    *,
    exec_context: str | None = None,
    wire: str = DEFAULT_WIRE,
) -> dict:
    """Return full routing info for *group* as a JSON-ready dict.

    Reads the registry directly (bypasses the Router's cooldown state).
    Prefers the LiteLLM proxy when healthy (port resolved from
    ``KREPIS_LITELLM_PROXY_URL`` env var or :data:`LITELLM_PROXY_URL`
    constant) — it handles format translation for all providers, making
    every registry model CLI-compatible.  Falls back to per-provider
    resolution when LiteLLM is unavailable.

    Parameters
    ----------
    group
        The model group name (``low``, ``med``, ``high``, ``ultra``).
    exec_context
        The caller's execution context — one of :data:`EXEC_CONTEXTS`.
        Declared, never inferred (R29). Defaults to ``KREPIS_EXEC_CONTEXT``
        and then to :data:`DEFAULT_EXEC_CONTEXT`. Entries the registry does
        not declare ``reachable_from`` this context are skipped, and each
        lands in ``skipped_entries`` naming the context — so "unreachable
        from here" is never confused with "unhealthy".
    wire
        The wire format the caller speaks: :data:`WIRE_ANTHROPIC` (the
        Claude CLI's Messages format, the default) or :data:`WIRE_OPENAI`.
        An entry that declares no endpoint for this format is skipped with
        that as the stated reason.

    Produces a dict with every field a shell script needs to configure
    the Claude CLI environment: endpoint URL, auth type, provider, route,
    deployment ID, registry ID, capabilities (prompt_caching, etc.), and
    cache pricing.

    The ``api_base_url`` is the CLI-compatible endpoint.  The LiteLLM proxy
    is the preferred path — it translates between CLI wire format and
    provider-native formats.  DeepSeek and OpenRouter are direct-fallback
    paths when LiteLLM is unavailable.  Every endpoint is resolvable via
    an env override (config#4923), so box-level bootstraps can export the
    override alongside their ``--port`` from a single configuration value.

    Raises :exc:`ValueError` if NO model in the group is CLI-compatible.
    """
    import yaml as _yaml

    # `exclude_route` used to be accepted here. It let a caller narrow the
    # fallback chain — which is holding a routing table at layer 5, whatever
    # the mechanism (model-router-policy §2). Both callers that passed it were
    # expressing "this route is not reachable from where I am running", which
    # is `exec_context` plus the registry's `reachable_from` (R28/R29). It was
    # deprecated in 0.27.0, both call sites migrated (crucible-evaluator #170;
    # the clauder wrapper's degraded path), and this is the R19 removal.
    #
    # It is worth remembering WHY it is gone rather than merely deprecated: it
    # was added to make the Director Lambda succeed while the path to the
    # LiteLLM proxy was down, and it worked — resolution fell through to
    # `glm-5.2` at openrouter.ai, DLP-unscanned, logging a healthy route for
    # weeks. Nothing failed when the proxy path was never restored. An argument
    # that lets a consumer route around an unreachable control is a control the
    # consumer can turn off.
    exec_context = _resolve_exec_context(exec_context)
    if wire not in WIRE_FORMATS:
        raise ValueError(
            f"Unknown wire format {wire!r}; declared formats are "
            f"{list(WIRE_FORMATS)}"
        )

    reg_path = _find_registry()
    if not reg_path:
        raise FileNotFoundError(
            "LLM_MODEL_REGISTRY.yaml not found — set "
            "LLM_MODEL_REGISTRY_PATH or run from within a repo "
            "whose private-docs/ directory contains the file."
        )

    with open(reg_path) as f:
        doc = _yaml.safe_load(f)

    models_by_id: dict[str, dict] = {m["id"]: m for m in doc.get("models", [])}
    group_ids: list[str] = doc.get("model_groups", {}).get(group, [])

    if not group_ids:
        raise ValueError(
            f"Model group {group!r} not found in registry. "
            f"Available groups: {list(doc.get('model_groups', {}).keys())}"
        )

    # ── Prefer LiteLLM proxy (format-translating central router) ──────────
    # When the LiteLLM proxy is healthy, it is ALWAYS the best endpoint for
    # the Claude CLI: it translates between CLI wire format and provider-native
    # formats, making every provider in the registry CLI-compatible regardless
    # of its native API format.  It also handles fallback chains with cooldown
    # and passes through caching hints.  All traffic still flows through egress
    # proxies for DLP scanning (placeholder key → real key injection).
    #
    # The probe URL comes from _resolve_litellm_proxy_url() so the port is
    # configurable via KREPIS_LITELLM_PROXY_URL env var (config#4923).
    #
    # Four SOTA checks gate the LiteLLM path:
    #   1. Proxy health (port reachable at the resolved URL)
    #   2. Master key resolvable (env → secrets.env → SSM)
    #   3. Running config not stale vs registry (LiteLLM boot-time config
    #      newer than or equal to registry mtime)
    #   4. Group exists in the registry (for the cache-capability lookup)
    # Any check failing → fall through to per-provider resolution.
    _litellm_url = _resolve_litellm_proxy_url()
    _litellm_reachable = False
    try:
        from urllib.parse import urlparse as _urlparse
        _parsed = _urlparse(_litellm_url)
        _host = _parsed.hostname or "127.0.0.1"
        _port = _parsed.port or 8980
        import http.client as _http
        conn = _http.HTTPConnection(_host, _port, timeout=2)
        conn.request("GET", "/health")
        resp = conn.getresponse()
        resp.read()
        conn.close()
        # 200 = no master key set; 401 = master key required but proxy alive.
        _litellm_reachable = resp.status in (200, 401)
    except Exception:
        _litellm_reachable = False

    _litellm_ok = False
    _litellm_skip_reasons: list[str] = []

    # There is deliberately no way for a caller to skip this path. The router
    # route is offered in every execution context and gated only on its health
    # probe (R27a.4) — a consumer that could exclude it would be making the
    # router's reachability a property of its own network, which is the exact
    # inversion §3.4a exists to forbid.
    if not _litellm_reachable:
        _litellm_skip_reasons.append(
            f"LiteLLM proxy at {_litellm_url} not reachable")
    else:
        # ── Check 2: master key resolvable ───────────────────────────────
        _master_key = _resolve_litellm_master_key()
        if _master_key is None:
            _litellm_skip_reasons.append(
                "LITELLM_MASTER_KEY not resolvable (env → secrets.env → SSM)")
        else:
            # ── Check 3: config staleness (RETIRED — externalized) ──────
            # The mtime-glob heuristic (_litellm_config_is_stale) has been
            # RETIRED.  It inferred staleness from filesystem mtimes over a
            # /tmp glob — an inference, not a fact, and nothing called it.
            #
            # The authoritative check lives in alpha-engine-config:
            #   scripts/check_router_config_provenance.py
            # which compares a content digest (sha256) of the generated config
            # against the live registry.  The auto-reconcile script
            # (scripts/reconcile_litellm_config.py) runs on a 10-minute timer
            # and restarts the router on digest mismatch (15-minute SLO, R10).
            #
            # This krepis path skips the staleness check: the external
            # reconcile is authoritative.  Proceed assuming the config is
            # current — the reconcile loop will fix it within minutes if not.
            _litellm_ok = True

    if _litellm_ok:
        # Caching capability comes from the PRIMARY entry — the model that
        # will actually serve the request — never from an any() over the
        # whole fallback chain.
        #
        # It used to be any(). That declares a capability the served model
        # may not have: a group whose primary does transparent prefix
        # caching but whose fourth fallback honours explicit breakpoints
        # reported "explicit breakpoints supported", so the CLI emitted
        # Anthropic cache_control markers at a provider that never asked for
        # them. Live on `ultra` until 2026-07-27
        # (alpha-engine-config-I4463).
        #
        # Cache pricing below was already primary-derived, so the two are
        # now consistent: everything describing "what serves this group"
        # comes from one entry.
        _primary_entry = models_by_id.get(group_ids[0], {}) if group_ids else {}
        _group_pc, _group_apc = _caching_flags(_primary_entry)
        _cache_pricing = {}
        for _key in ("cost_per_1m_input", "cost_per_1m_output",
                      "cost_per_1m_cache_read", "cost_per_1m_cache_write"):
            if _key in _primary_entry:
                _cache_pricing[_key] = _primary_entry[_key]

        # Derive max_tokens from the primary entry's params
        _params = dict(_primary_entry.get("params", {}))
        # LiteLLM doesn't set a default max_tokens — copy from the primary
        # entry so the Claude CLI has a sensible limit.
        if "max_tokens" not in _params:
            _params["max_tokens"] = 8192

        # The primary model from the registry — displayed to the user so they
        # know which concrete model serves this group.  LiteLLM may internally
        # fall back through the chain; the actual model appears in resp.model.
        _primary_model = _primary_entry.get("model", group)
        _primary_registry_id = _primary_entry.get("id", group)

        _display_name = f"{_primary_model} ({group})"

        return _with_compat_aliases({
            "schema_version": RESOLVE_SCHEMA_VERSION,
            "model": group,
            "display_name": _display_name,
            "provider": "litellm",
            "route": "litellm_proxy",
            "api_base_url": _litellm_url,
            "deployment_id": group,
            "auth_token_type": "litellm_master_key",
            "group": group,
            "registry_id": f"litellm:group:{group}",
            "primary_model": _primary_model,
            "primary_registry_id": _primary_registry_id,
            "capabilities": {
                "web_search": False,
                "tool_choice": False,
                "prompt_caching": _group_pc,
                "automatic_prefix_caching": _group_apc,
                "batches": False,
            },
            "params": _params,
            "cache_pricing": _cache_pricing,
            # supports_prompt_caching: only true for explicit cache_control
            # breakpoints.  automatic_prefix_caching works transparently on
            # the server side without client markers.
            "supports_prompt_caching": _group_pc,
            "automatic_prefix_caching": _group_apc,
            "exec_context": exec_context,
            "wire": wire,
            "skipped_entries": [],
        })

    # ── LiteLLM unavailable or gated — fall through to per-provider resolution ──
    # Log skip reasons to stderr so operators can diagnose routing decisions.

    if _litellm_skip_reasons:
        _msg = "; ".join(_litellm_skip_reasons)
        logger.warning("LiteLLM proxy skipped for group %r: %s", group, _msg)

    skips: list[dict] = []

    # Iterate the fallback chain — first CLI-compatible entry wins
    for mid in group_ids:
        entry = models_by_id.get(mid)
        if entry is None:
            continue

        _entry_route = entry.get("route", "")

        # R28/R29 — is this entry's endpoint reachable from where the caller
        # says it is running?  Recorded with the context named, so a caller
        # reading skipped_entries can tell "not reachable from here" apart
        # from "unhealthy", which is the distinction the old resolution path
        # collapsed.
        if not _entry_reachable_from(entry, exec_context):
            _declared = entry.get("reachable_from")
            if _declared is None:
                _reason = (
                    "Registry entry declares no reachable_from, so it is "
                    "unreachable from every context (model-router-policy R28). "
                    "This registry copy is stale or hand-written — the "
                    "validator rejects the field's absence."
                )
            else:
                _reason = (
                    f"Not reachable from execution context "
                    f"{exec_context!r} (registry reachable_from={_declared!r})"
                )
            skips.append({
                "registry_id": mid,
                "provider": entry.get("provider", ""),
                "reason": _reason,
            })
            continue

        # Skip entries that declare no endpoint for the caller's wire format.
        # This used to be decided by absence from a hardcoded (route, provider)
        # table in this module — so an entry was dropped because krepis had no
        # row for its provider, which is a routing fact invented at layer 3
        # (R7/R29).  It is now the registry's statement about itself.
        try:
            api_base_url = _entry_endpoint(entry, wire)
        except ValueError as exc:
            skips.append({
                "registry_id": mid,
                "provider": entry.get("provider", ""),
                "reason": f"No {wire!r}-wire endpoint declared: {exc}",
            })
            continue

        # An egress_proxy endpoint that does not answer its health probe is
        # unusable; fail at resolve time rather than handing back a dead
        # endpoint that surfaces later as an opaque ConnectionRefused
        # (config#4923).
        if _entry_route == "egress_proxy" and api_base_url:
            if not _probe_egress_proxy(api_base_url):
                skips.append({
                    "registry_id": mid,
                    "provider": entry.get("provider", ""),
                    "reason": f"Egress proxy at {api_base_url!r} is not reachable",
                })
                continue

        model_str = entry.get("model", "")
        route = entry.get("route", "")
        provider = entry.get("provider", "")
        capabilities = entry.get("capabilities", {})
        params = entry.get("params", {})

        # Determine auth token type
        if route == "egress_proxy":
            auth_token_type = "placeholder"
        elif route == "openrouter":
            auth_token_type = "openrouter_key"
        elif provider == "anthropic":
            auth_token_type = "direct_api_key"
        else:
            auth_token_type = "placeholder"

        # Extract cache pricing for the clauder script to decide caching config
        cache_pricing = {}
        for key in ("cost_per_1m_input", "cost_per_1m_output",
                     "cost_per_1m_cache_read", "cost_per_1m_cache_write"):
            if key in entry:
                cache_pricing[key] = entry[key]

        _entry_pc, _entry_apc = _caching_flags(entry)

        _display_name = f"{model_str} ({group})"

        return _with_compat_aliases({
            "schema_version": RESOLVE_SCHEMA_VERSION,
            "model": model_str,
            "display_name": _display_name,
            "provider": provider,
            "route": route,
            "api_base_url": api_base_url,
            "deployment_id": _cli_deployment_id(entry),
            "auth_token_type": auth_token_type,
            "group": group,
            "registry_id": mid,
            "capabilities": capabilities,
            # Same single reader as the LiteLLM branch, so the two paths
            # cannot disagree about a model's caching mechanism.
            "supports_automatic_prefix_caching": _entry_apc,
            "automatic_prefix_caching": _entry_apc,
            "params": params,
            "cache_pricing": cache_pricing,
            "supports_prompt_caching": _entry_pc,
            "exec_context": exec_context,
            "wire": wire,
            "skipped_entries": skips if skips else [],
        })

    # R29 — fail closed. No entry in the group is reachable from this
    # context in this wire format, so there is nothing legitimate left to
    # reach for. The skip reasons are carried into the message because this
    # exception is frequently the only artifact a weekly unattended caller
    # leaves behind, and "no model was CLI-compatible" without them sent
    # operators looking at provider health for a reachability fault.
    _detail = "; ".join(
        f"{s_['registry_id']}: {s_['reason']}" for s_ in skips
    ) or "no entries examined"
    raise ValueError(
        f"No model in group {group!r} is reachable from execution context "
        f"{exec_context!r} in the {wire!r} wire format. "
        f"Chain: {group_ids}. Skipped — {_detail}. "
        f"LiteLLM proxy skipped: {'; '.join(_litellm_skip_reasons) or 'no'}."
    )


def resolve_group_structured(
    group: str,
    *,
    exec_context: str | None = None,
    wire: str = DEFAULT_WIRE,
) -> dict:
    """Resolve *group* to a full routing decision — the PUBLIC contract.

    This is the supported entry point for programmatic callers.  It returns
    the same dict the ``resolve <group> --json`` CLI prints, carrying every
    field a caller needs to configure an LLM client: endpoint, model,
    auth-token type, capabilities, params, and cache pricing.

    The returned dict conforms to ``resolve_schema.json`` (validated in CI)
    and carries ``schema_version`` (:data:`RESOLVE_SCHEMA_VERSION`).  Callers
    MUST branch on ``schema_version`` rather than probing for fields.

    Raises
    ------
    FileNotFoundError
        No registry file could be located.
    ValueError
        *group* is not in the registry, or no model in it is reachable.

    Notes
    -----
    ``alpha-engine-config``'s groom driver has imported this name since it was
    written, while the module only ever exposed the private
    ``_resolve_group_json`` — so the import failed on every run, the router
    path was dead code, and routing silently fell back to an endpoint exported
    by a bootstrap shell script (alpha-engine-config-I4454).  Consumers must
    bind to a supported public surface; a leading underscore is not one.

    Callers declare *where they are running* (``exec_context``) and *what wire
    format they speak* (``wire``), and nothing else about routing.  Which
    entries those two facts admit is a registry decision, resolved above the
    consumer (model-router-policy §2 layer 5, R28/R29).  ``exclude_route`` was
    removed in 0.29.0 after both call sites migrated; see
    :func:`_resolve_group_json`.
    """
    return _resolve_group_json(
        group,
        exec_context=exec_context,
        wire=wire,
    )


# ── CLI ──────────────────────────────────────────────────────────────────

def _cli() -> None:
    """Entry point for ``python3 -m krepis.router``."""
    if len(sys.argv) < 2:
        print("Usage: python3 -m krepis.router <command> [args]", file=sys.stderr)
        print("  resolve <group> [--json]  — print first healthy model for group", file=sys.stderr)
        print("  groups                    — list all model groups", file=sys.stderr)
        print("  models                    — list all models in the Router", file=sys.stderr)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "resolve":
        if len(sys.argv) < 3:
            print("Usage: python3 -m krepis.router resolve <low|med|high|ultra> [--json] [--exec-context <ctx>] [--wire <fmt>]", file=sys.stderr)
            print(f"  --exec-context  one of {list(EXEC_CONTEXTS)} (default: $KREPIS_EXEC_CONTEXT, else {DEFAULT_EXEC_CONTEXT})", file=sys.stderr)
            print(f"  --wire          one of {list(WIRE_FORMATS)} (default: {DEFAULT_WIRE})", file=sys.stderr)
            sys.exit(1)
        group = sys.argv[2]
        want_json = "--json" in sys.argv
        exec_context: str | None = None
        wire = DEFAULT_WIRE
        for i, arg in enumerate(sys.argv):
            if arg == "--exclude-route":
                # Removed in 0.29.0. Exiting beats silently ignoring it: a
                # caller passing it believes it is narrowing the chain, and a
                # flag that is accepted-and-dropped resolves to something the
                # caller did not ask for.
                print("--exclude-route was removed in krepis 0.29.0. Declare --exec-context instead and let the registry's reachable_from decide (model-router-policy R28/R29).", file=sys.stderr)
                sys.exit(2)
            elif arg == "--exec-context" and i + 1 < len(sys.argv):
                exec_context = sys.argv[i + 1]
            elif arg == "--wire" and i + 1 < len(sys.argv):
                wire = sys.argv[i + 1]
        if want_json:
            info = _resolve_group_json(
                group,
                exec_context=exec_context,
                wire=wire,
            )
            print(json.dumps(info))
        else:
            model = resolve_group(group)
            print(model)

    elif cmd == "groups":
        router = get_router()
        for fb in router.fallbacks:
            group_name = list(fb.keys())[0]
            print(group_name)

    elif cmd == "models":
        router = get_router()
        for m in router.model_list:
            name = m["model_name"]
            upstream = _upstream_model(m["litellm_params"]["model"])
            print(f"{name} → {upstream}")

    else:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    _cli()
