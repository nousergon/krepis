"""
Per-key SSM-backed secret fetcher for Alpha Engine modules.

This module is the consolidation point for secret resolution across all six
alpha-engine repos. Before this, each repo duplicated some variant of the
``ssm_secrets.py`` bulk-load-into-os.environ pattern (alpha-engine-data has
the canonical one). Going forward, callers do::

    from krepis.secrets import get_secret

    api_key = get_secret("ANTHROPIC_API_KEY")              # required → raises if absent
    opt_key = get_secret("DEBUG_TRACE_KEY", required=False)  # optional → returns None

**Resolution order** (first hit wins):

1. Per-process cache (populated on first read; thread-safe).
2. ``ALPHA_ENGINE_SECRETS_SOURCE`` env-var toggle:

   - ``env`` → ``os.environ[name]`` only (local-dev escape hatch — never hit SSM)
   - ``ssm`` → SSM only (production strictness — no silent env fallback)
   - unset / ``auto`` / anything else → SSM first, ``os.environ`` fallback

3. SSM at ``/alpha-engine/{name}`` (under :data:`SSM_PREFIX`).
4. ``os.environ[name]`` (the fallback in ``auto`` mode).
5. ``default`` arg if provided.
6. :exc:`SecretNotFoundError` if ``required=True``; else ``None``.

**Caching design.** Per-process dict keyed on secret name. The SSM round-trip
happens at most once per name per process; subsequent calls hit the dict.
Lambda cold-starts pay the round-trip once; warm invocations reuse the cache.
:func:`clear_cache` is exposed for tests.

**SSM-unavailable latch.** If the first SSM call fails (no boto3, no creds,
network error), latch ``_ssm_unavailable = True`` and skip SSM for the rest of
the process. Avoids repeated multi-second timeouts in local dev. Reset via
:func:`clear_cache` if a test needs to re-probe SSM.

**Migration arc**: ``alpha-engine-config/private-docs/ROADMAP.md`` line ~2780
(Deprecate ``.env`` entirely). Plan doc:
``alpha-engine-docs/private/env-to-ssm-260512.md``.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Final

logger = logging.getLogger(__name__)

SSM_PREFIX: Final[str] = "/alpha-engine/"
SOURCE_TOGGLE_ENV: Final[str] = "ALPHA_ENGINE_SECRETS_SOURCE"

_cache: dict[str, str] = {}
_cache_lock = threading.Lock()
_ssm_unavailable = False


class SecretNotFoundError(LookupError):
    """Raised when a required secret is missing from both SSM and the environment."""


def get_secret(
    name: str,
    *,
    required: bool = True,
    default: str | None = None,
) -> str | None:
    """Fetch a secret by ``name`` from SSM with environment fallback.

    See module docstring for the full resolution order. The lookup is
    per-process cached; the first call per process pays the SSM round-trip,
    subsequent calls hit the in-memory dict.

    :param name: Secret name (no prefix). E.g. ``"POLYGON_API_KEY"``.
    :param required: If ``True`` (default), raise :exc:`SecretNotFoundError`
        when the secret is absent. If ``False``, return ``default`` (or
        ``None`` if ``default`` is unset).
    :param default: Value to return if the secret is absent and
        ``required=False``. Ignored when ``required=True``.
    :raises SecretNotFoundError: When ``required=True`` and the secret is
        absent from cache, SSM, and ``os.environ``.
    :raises ValueError: When ``name`` is empty or contains a forward slash.
    """
    if not name:
        raise ValueError("secret name must be non-empty")
    if "/" in name:
        raise ValueError(
            f"secret name must not contain '/': got {name!r} "
            f"(the SSM_PREFIX is added automatically)"
        )

    with _cache_lock:
        cached = _cache.get(name)
    if cached is not None:
        return cached

    source = os.environ.get(SOURCE_TOGGLE_ENV, "auto").lower()
    if source not in ("auto", "env", "ssm"):
        logger.warning(
            "unknown %s=%r — falling back to 'auto'", SOURCE_TOGGLE_ENV, source
        )
        source = "auto"

    value: str | None = None

    if source in ("auto", "ssm"):
        value = _fetch_from_ssm(name)

    if value is None and source in ("auto", "env"):
        value = os.environ.get(name)

    if value is None:
        if default is not None:
            return default
        if required:
            raise SecretNotFoundError(
                f"secret {name!r} not found in cache, SSM ({SSM_PREFIX}{name}), "
                f"or environment (source={source!r})"
            )
        return None

    with _cache_lock:
        _cache[name] = value
    return value


def clear_cache() -> None:
    """Clear the per-process cache and re-arm SSM probing.

    Mostly for tests — production code should not need to call this. Resets
    both the secret cache and the ``_ssm_unavailable`` latch.
    """
    global _ssm_unavailable
    with _cache_lock:
        _cache.clear()
        _ssm_unavailable = False


def _fetch_from_ssm(name: str) -> str | None:
    """Single-key SSM read. Returns ``None`` on miss or unavailability."""
    global _ssm_unavailable
    if _ssm_unavailable:
        return None

    try:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError:
        logger.debug("boto3 not installed — skipping SSM for %s", name)
        _ssm_unavailable = True
        return None

    region = os.environ.get("AWS_REGION") or os.environ.get(
        "AWS_DEFAULT_REGION", "us-east-1"
    )
    try:
        client = boto3.client("ssm", region_name=region)
        resp = client.get_parameter(
            Name=f"{SSM_PREFIX}{name}",
            WithDecryption=True,
        )
        return resp["Parameter"]["Value"]
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code == "ParameterNotFound":
            # Genuine miss — not an SSM-availability problem. Fall through
            # to env without latching, since other secrets may resolve fine.
            logger.debug("SSM miss for %s (ParameterNotFound)", name)
            return None
        logger.warning(
            "SSM read for %s failed (%s) — latching unavailable for this process",
            name,
            code or "unknown",
        )
        _ssm_unavailable = True
        return None
    except BotoCoreError as e:
        logger.warning(
            "SSM read for %s failed (%s) — latching unavailable for this process",
            name,
            type(e).__name__,
        )
        _ssm_unavailable = True
        return None
