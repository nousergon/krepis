"""Is this process part of a pipeline REHEARSAL, and where do its records go?

A rehearsal is a full run of a scheduled pipeline launched by hand, at any
time, to prove the pipeline works before the scheduled run needs it. It runs
the same code as the scheduled run and must never write the scheduled run's
records: a rehearsal that overwrites a live verdict, health row or console row
makes the live surface describe a run that did not happen in production.

**The signal.** The orchestrator names a rehearsal execution
``rehearsal-<date>-<n>``, and :mod:`krepis.ssm_log_capture` exports that name to
its child process as ``RUN_TOKEN``. So a process is a rehearsal when the
execution name it was given, or ``RUN_TOKEN``, starts with
:data:`REHEARSAL_PREFIX`. ``AE_PIPELINE_ROLE=rehearsal`` is the explicit
override for a hand-launched process that has no execution name. The same
three inputs are what the pipeline repos already key their own rehearsal
guards on, so a process is never a rehearsal to one layer and live to another.

**Where its records go.** :func:`namespaced_key` puts a rehearsal's record
under a ``.rehearsal/<token>/`` segment directly below the record's own
top-level prefix (``_stage_coverage/x/y.json`` becomes
``_stage_coverage/.rehearsal/<token>/x/y.json``). Readers of the live record
address it by its exact key or a date partition, so they never see the
rehearsal's copy. Staying below the same top-level prefix also means a role
that may write the live record may write the rehearsal's copy, so isolating a
rehearsal never needs a wider grant.

Everything here is pure: no I/O and no AWS client.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import Final, Optional

#: Execution-name prefix of a pipeline rehearsal.
REHEARSAL_PREFIX: Final[str] = "rehearsal-"
#: Env var :mod:`krepis.ssm_log_capture` sets to the execution name.
RUN_TOKEN_ENV: Final[str] = "RUN_TOKEN"  # noqa: S105 — an env var NAME, not a secret
#: Explicit override for a hand-launched rehearsal process.
PIPELINE_ROLE_ENV: Final[str] = "AE_PIPELINE_ROLE"
#: Token used when ``AE_PIPELINE_ROLE=rehearsal`` is set without an execution name.
UNNAMED_TOKEN: Final[str] = "rehearsal-unnamed"
#: Path segment a rehearsal's records sit under, below their own prefix.
NAMESPACE_SEGMENT: Final[str] = ".rehearsal"

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


def _name_from(value: Optional[str]) -> str:
    """The execution name inside ``value``: a bare name or an execution ARN."""
    text = (value or "").strip()
    if text.startswith("arn:"):
        text = text.rsplit(":", 1)[-1]
    return text


def rehearsal_token(
    execution_name: Optional[str] = None,
    *,
    env: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    """The rehearsal's execution name, or ``None`` when this is not a rehearsal.

    ``execution_name`` (a bare name or an execution ARN) wins when given. A
    production execution name that is passed explicitly is authoritative: it
    is not overridden by a stray ``RUN_TOKEN`` in the environment. When it is
    absent, ``RUN_TOKEN`` decides, then ``AE_PIPELINE_ROLE=rehearsal``.

    The returned token is safe to use as one S3 path segment.
    """
    env = os.environ if env is None else env
    name = _name_from(execution_name)
    if not name:
        name = _name_from(env.get(RUN_TOKEN_ENV))
        if not name and (env.get(PIPELINE_ROLE_ENV) or "").strip().lower() == "rehearsal":
            return UNNAMED_TOKEN
    if not name.startswith(REHEARSAL_PREFIX):
        return None
    return _UNSAFE.sub("", name) or UNNAMED_TOKEN


def is_rehearsal(
    execution_name: Optional[str] = None,
    *,
    env: Optional[Mapping[str, str]] = None,
) -> bool:
    """True when :func:`rehearsal_token` names a rehearsal."""
    return rehearsal_token(execution_name, env=env) is not None


def namespaced_key(key: str, token: Optional[str]) -> str:
    """``key`` as a rehearsal's copy of it, or ``key`` itself when ``token`` is None.

    ``_stage_coverage/2026-10-15/Backtester.json`` with token
    ``rehearsal-2026-10-16-2`` becomes
    ``_stage_coverage/.rehearsal/rehearsal-2026-10-16-2/2026-10-15/Backtester.json``.
    A key with no ``/`` gets the segment in front of it.
    """
    if not token:
        return key
    head, sep, rest = key.partition("/")
    if not sep:
        return f"{NAMESPACE_SEGMENT}/{token}/{key}"
    return f"{head}/{NAMESPACE_SEGMENT}/{token}/{rest}"


__all__ = [
    "NAMESPACE_SEGMENT",
    "PIPELINE_ROLE_ENV",
    "REHEARSAL_PREFIX",
    "RUN_TOKEN_ENV",
    "UNNAMED_TOKEN",
    "is_rehearsal",
    "namespaced_key",
    "rehearsal_token",
]
