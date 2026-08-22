"""The writer declares its S3 surface; the consumer's IAM contract consumes it.

**The bug class this closes** (``alpha-engine-config-I8156``). A consumer repo's
IAM S3-prefix contract test reads *its own* AST: it enumerates the boto3
``Key=`` / ``Prefix=`` call sites in its own production packages and asserts
each resolvable prefix is declared in a contract file the ops repo grants
against. A prefix written by an **imported library** has no call site in that
scan, so it is invisible to the guard while being entirely real at runtime.

Measured 2026-08-22: :mod:`krepis.stage_coverage` writes ``_stage_coverage/``
and :mod:`krepis.cost_sink` writes ``decision_artifacts/``. Neither was
declared in ``crucible-evaluator/grading/iam_s3_contract.json``,
``alpha-engine-evaluator-role`` never granted either, and the result stood
undetected for as long as the prefixes existed — four weekly stages with zero
coverage verdicts since 2026-08-14, and the Director's LLM spend never once
attributed. The failure mode produces no PR signal, no CI signal and no page:
only a fail-soft ``ERROR`` log nobody reads.

**Why the declaration lives with the WRITER.** The consumer cannot derive it —
that is the whole defect. Every repo adopting :mod:`krepis.stage_coverage` or
:mod:`krepis.cost_sink` inherits the same hole, so a per-consumer fix scales
with the number of consumers and is wrong the moment a new one appears. The
writing module knows its own namespace; it says so once, here, and every
consumer's contract test reads the same declaration.

**Three declaration kinds, because krepis writes three ways:**

- :data:`KIND_LITERAL` — a fixed top-level prefix baked into the module
  (``_stage_coverage``, ``_ssm_logs``). Resolves unconditionally.
- :data:`KIND_ENV` — the target is an environment fact, not a code fact
  (:mod:`krepis.cost_sink`'s ``KREPIS_COST_SINK_PREFIX``). The declaration
  names the **variable**; :func:`prefixes_for` resolves it against a
  **caller-supplied** environment mapping so the consumer checks the value its
  own deploy config actually sets, not this laptop's. An unset variable
  resolves to nothing — the module writes nowhere, so nothing needs granting.
- :data:`KIND_CALLER_SUPPLIED` — the bucket and prefix are arguments
  (:mod:`krepis.locks`, :mod:`krepis._dedup`, :mod:`krepis.ssm_dispatcher`).
  krepis cannot name a prefix it never chooses. This kind contributes **no**
  prefix, and exists so that a module which genuinely has nothing to declare
  has still *said so*. "Declared caller-supplied" and "nobody ever considered
  this module" are different claims and only the first is a pass — the same
  distinction :mod:`krepis.stage_coverage` draws between ``output: none`` and
  an absent registry row.

**Modes are ``read`` / ``readwrite``**, matching the consumer contract's own
vocabulary, so a declaration drops into an IAM contract without translation.
Unioning takes the **wider** mode: a prefix declared ``read`` by one module and
``readwrite`` by another needs ``readwrite``.

**Fail loud.** A module with no declaration is not silently skipped — it simply
contributes nothing, and ``tests/test_s3_surface.py`` in this repo fails any
module that performs an S3 write without a declaration, so the declaration
cannot rot the way the contract it feeds did. An unimportable module named to
:func:`prefixes_for` raises: a consumer asking about a module that does not
exist has a broken scan, and answering "no prefixes" would render that as a
clean bill of health.

Public surface:

- :class:`SurfaceEntry` — one declared prefix (or variable) and its mode.
- :func:`literal` / :func:`from_env_var` / :func:`caller_supplied` —
  constructors.
- :func:`prefixes_for` — ``modules -> {prefix: mode}``, the consumer's door.
- :func:`declarations_for` — the raw entries, for a caller that wants to
  report *why* a prefix is required (or why one did not resolve).
- :func:`module_declares_surface` — whether one module has declared anything.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Dict, Final, Iterable, List, Mapping, Optional, Tuple

__all__ = [
    "ATTRIBUTE",
    "KIND_CALLER_SUPPLIED",
    "KIND_ENV",
    "KIND_LITERAL",
    "MODE_READ",
    "MODE_READWRITE",
    "SurfaceEntry",
    "caller_supplied",
    "declarations_for",
    "from_env_var",
    "literal",
    "module_declares_surface",
    "prefixes_for",
    "widest_mode",
]

#: The module-level attribute a writing module declares its surface in.
ATTRIBUTE: Final[str] = "S3_SURFACE"

KIND_LITERAL: Final[str] = "literal"
KIND_ENV: Final[str] = "env"
KIND_CALLER_SUPPLIED: Final[str] = "caller_supplied"

MODE_READ: Final[str] = "read"
MODE_READWRITE: Final[str] = "readwrite"

_KINDS: Final[frozenset] = frozenset({KIND_LITERAL, KIND_ENV, KIND_CALLER_SUPPLIED})
_MODES: Final[frozenset] = frozenset({MODE_READ, MODE_READWRITE})


@dataclass(frozen=True)
class SurfaceEntry:
    """One declared element of a module's S3 surface.

    Attributes:
        kind: One of :data:`KIND_LITERAL`, :data:`KIND_ENV`,
            :data:`KIND_CALLER_SUPPLIED`.
        mode: :data:`MODE_READ` or :data:`MODE_READWRITE` — the access the
            module needs, in the consumer contract's own vocabulary.
        prefix: The **top-level** prefix, for :data:`KIND_LITERAL`. A key
            like ``overseer/intake-fallback/...`` declares ``overseer``:
            IAM grants are written per top-level namespace, and declaring a
            deeper path would not match the contract it feeds.
        variable: The environment variable naming the prefix, for
            :data:`KIND_ENV`.
        reason: Why this module has no nameable prefix, for
            :data:`KIND_CALLER_SUPPLIED`. Required for that kind: a bare
            "nothing to declare" is indistinguishable from an oversight.
    """

    kind: str
    mode: str = MODE_READWRITE
    prefix: str = ""
    variable: str = ""
    reason: str = ""

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(
                "SurfaceEntry.kind must be one of "
                + ", ".join(sorted(_KINDS))
                + f"; got {self.kind!r}"
            )
        if self.mode not in _MODES:
            raise ValueError(
                "SurfaceEntry.mode must be one of "
                + ", ".join(sorted(_MODES))
                + f"; got {self.mode!r}"
            )
        if self.kind == KIND_LITERAL:
            if not self.prefix:
                raise ValueError("a literal SurfaceEntry requires a prefix")
            if "/" in self.prefix:
                raise ValueError(
                    f"declare the TOP-LEVEL prefix only; got {self.prefix!r}. "
                    "IAM grants are written per top-level namespace and a "
                    "deeper path cannot match the consumer contract."
                )
        if self.kind == KIND_ENV and not self.variable:
            raise ValueError("an env SurfaceEntry requires a variable name")
        if self.kind == KIND_CALLER_SUPPLIED and not self.reason:
            raise ValueError(
                "a caller_supplied SurfaceEntry requires a reason — "
                "'nothing to declare' and 'nobody considered it' must not "
                "share a shape"
            )


def literal(prefix: str, mode: str = MODE_READWRITE) -> SurfaceEntry:
    """Declare a fixed top-level prefix this module reads or writes."""
    return SurfaceEntry(kind=KIND_LITERAL, prefix=prefix, mode=mode)


def from_env_var(name: str, mode: str = MODE_READWRITE) -> SurfaceEntry:
    """Declare that the prefix is named by the environment variable ``name``."""
    return SurfaceEntry(kind=KIND_ENV, variable=name, mode=mode)


def caller_supplied(reason: str, mode: str = MODE_READWRITE) -> SurfaceEntry:
    """Declare that the caller chooses the bucket and prefix, not this module."""
    return SurfaceEntry(kind=KIND_CALLER_SUPPLIED, reason=reason, mode=mode)


def widest_mode(left: str, right: str) -> str:
    """The mode satisfying both. ``readwrite`` strictly contains ``read``."""
    return MODE_READWRITE if MODE_READWRITE in (left, right) else MODE_READ


def _top_segment(value: str) -> str:
    """Top-level segment of a prefix value (``'a/b' -> 'a'``)."""
    return value.strip().strip("/").split("/", 1)[0]


def _import(module: str) -> Any:
    dotted = module if module.startswith("krepis") else f"krepis.{module}"
    try:
        return importlib.import_module(dotted)
    except ImportError as exc:  # fail loud: see the module docstring
        raise ImportError(
            f"krepis.s3_surface: cannot import {dotted!r} to read its declared "
            "S3 surface. Answering 'no prefixes' for an unimportable module "
            "would render a broken scan as a clean bill of health."
        ) from exc


def declarations_for(modules: Iterable[str]) -> "Dict[str, Tuple[SurfaceEntry, ...]]":
    """Return ``{dotted_module_name: entries}`` for each named module.

    A bare name (``"stage_coverage"``) is resolved under ``krepis.``. A module
    with no declaration maps to an empty tuple — that is a real answer, and the
    krepis anti-rot test is what keeps it from being the wrong one.
    """
    out: "Dict[str, Tuple[SurfaceEntry, ...]]" = {}
    for name in modules:
        mod = _import(name)
        out[mod.__name__] = tuple(getattr(mod, ATTRIBUTE, ()) or ())
    return out


def prefixes_for(
    modules: Iterable[str],
    *,
    environment: Optional[Mapping[str, str]] = None,
) -> "Dict[str, str]":
    """Return ``{top_level_prefix: mode}`` for everything ``modules`` declare.

    Args:
        modules: krepis module names, bare (``"cost_sink"``) or dotted
            (``"krepis.cost_sink"``).
        environment: The mapping :data:`KIND_ENV` declarations resolve against.
            **Caller-supplied on purpose** — the consumer passes the deploy
            configuration its own Lambda or box actually sets, so the contract
            it checks is the one that will run. ``None`` means "nothing
            configured", which resolves every env-kind declaration to nothing
            rather than silently reading this process's own environment: an
            implicit fall-back to the auditing machine's configuration is
            exactly the answer-about-a-different-world this module exists to
            prevent.

    Returns:
        Mapping of top-level prefix to the widest mode any declaring module
        needs. Entries of kind :data:`KIND_CALLER_SUPPLIED`, and env-kind
        entries whose variable is unset or blank, contribute nothing.
    """
    configured: Mapping[str, str] = environment if environment is not None else {}
    resolved: "Dict[str, str]" = {}
    for entries in declarations_for(modules).values():
        for entry in entries:
            if entry.kind == KIND_LITERAL:
                prefix = entry.prefix
            elif entry.kind == KIND_ENV:
                prefix = _top_segment(configured.get(entry.variable, "") or "")
            else:
                continue
            if not prefix:
                continue
            resolved[prefix] = (
                widest_mode(resolved[prefix], entry.mode)
                if prefix in resolved
                else entry.mode
            )
    return resolved


def module_declares_surface(module: str) -> bool:
    """Whether ``module`` carries a non-empty :data:`ATTRIBUTE` declaration."""
    entries: "List[SurfaceEntry]" = list(getattr(_import(module), ATTRIBUTE, ()) or [])
    return bool(entries)
