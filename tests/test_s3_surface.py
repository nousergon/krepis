"""Tests for :mod:`krepis.s3_surface` — the writer's own S3 declaration.

The last test in this file is the one that matters most: it is the anti-rot
guard. ``alpha-engine-config-I8156`` exists because a consumer's IAM contract
claimed to cover "every top-level prefix this code reads or writes" while
structurally being unable to see a library-mediated one. A declaration that
only humans remember to update fails exactly the same way, one release later.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from krepis import cost_sink, s3_surface, stage_coverage

SRC = Path(s3_surface.__file__).resolve().parent

#: S3 methods that WRITE. A module containing one of these must declare a
#: surface. Reads are deliberately not enforced: a read of a caller-supplied
#: key is the overwhelming majority case and enforcing it would train people
#: to add empty declarations, which is worse than none.
_WRITE_METHODS = frozenset(
    {"put_object", "copy_object", "upload_file", "upload_fileobj"}
)


# ── SurfaceEntry construction ────────────────────────────────────────────────


def test_a_literal_entry_carries_its_prefix_and_mode() -> None:
    entry = s3_surface.literal("_stage_coverage")
    assert entry.kind == s3_surface.KIND_LITERAL
    assert entry.prefix == "_stage_coverage"
    assert entry.mode == s3_surface.MODE_READWRITE


def test_a_literal_entry_refuses_a_nested_path() -> None:
    """IAM grants are per top-level namespace; a nested path cannot match."""
    with pytest.raises(ValueError, match="TOP-LEVEL"):
        s3_surface.literal("overseer/intake-fallback")


def test_a_literal_entry_refuses_an_empty_prefix() -> None:
    with pytest.raises(ValueError, match="requires a prefix"):
        s3_surface.literal("")


def test_an_env_entry_requires_a_variable_name() -> None:
    with pytest.raises(ValueError, match="requires a variable"):
        s3_surface.SurfaceEntry(kind=s3_surface.KIND_ENV)


def test_a_caller_supplied_entry_requires_a_reason() -> None:
    """'Nothing to declare' and 'nobody considered it' must not share a shape."""
    with pytest.raises(ValueError, match="requires a reason"):
        s3_surface.SurfaceEntry(kind=s3_surface.KIND_CALLER_SUPPLIED)


def test_an_unknown_kind_or_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="kind must be one of"):
        s3_surface.SurfaceEntry(kind="whatever")
    with pytest.raises(ValueError, match="mode must be one of"):
        s3_surface.literal("x", mode="write")


# ── prefixes_for ─────────────────────────────────────────────────────────────


def test_a_literal_declaration_resolves_unconditionally() -> None:
    assert s3_surface.prefixes_for(["stage_coverage"])["_stage_coverage"] == (
        s3_surface.MODE_READWRITE
    )


def test_a_bare_module_name_resolves_under_krepis() -> None:
    assert s3_surface.prefixes_for(["stage_coverage"]) == s3_surface.prefixes_for(
        ["krepis.stage_coverage"]
    )


def test_an_env_declaration_resolves_against_the_supplied_configuration() -> None:
    resolved = s3_surface.prefixes_for(
        ["cost_sink"],
        environment={cost_sink.PREFIX_ENV_VAR: "decision_artifacts"},
    )
    assert resolved == {"decision_artifacts": s3_surface.MODE_READWRITE}


def test_an_env_declaration_resolves_to_the_top_level_segment() -> None:
    resolved = s3_surface.prefixes_for(
        ["cost_sink"],
        environment={cost_sink.PREFIX_ENV_VAR: "decision_artifacts/director/"},
    )
    assert resolved == {"decision_artifacts": s3_surface.MODE_READWRITE}


def test_an_unset_env_declaration_contributes_nothing() -> None:
    """The sink writes nowhere, so nothing needs granting. Not an error."""
    assert s3_surface.prefixes_for(["cost_sink"], environment={}) == {}


def test_no_environment_never_falls_back_to_the_process_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An implicit read of os.environ would answer about a different world."""
    monkeypatch.setenv(cost_sink.PREFIX_ENV_VAR, "decision_artifacts")
    assert s3_surface.prefixes_for(["cost_sink"]) == {}


def test_a_caller_supplied_declaration_contributes_no_prefix() -> None:
    assert s3_surface.prefixes_for(["ssm_dispatcher"]) == {}
    assert s3_surface.module_declares_surface("ssm_dispatcher") is True


def test_the_widest_mode_wins_when_two_modules_share_a_prefix() -> None:
    assert (
        s3_surface.widest_mode(s3_surface.MODE_READ, s3_surface.MODE_READWRITE)
        == s3_surface.MODE_READWRITE
    )
    assert (
        s3_surface.widest_mode(s3_surface.MODE_READ, s3_surface.MODE_READ)
        == s3_surface.MODE_READ
    )


def test_a_module_with_no_declaration_resolves_to_nothing_not_an_error() -> None:
    assert s3_surface.prefixes_for(["dates"]) == {}
    assert s3_surface.module_declares_surface("dates") is False


def test_an_unimportable_module_raises_rather_than_reporting_clean() -> None:
    """A broken scan must not be indistinguishable from a clean bill of health."""
    with pytest.raises(ImportError, match="clean bill of health"):
        s3_surface.prefixes_for(["no_such_module_at_all"])


def test_declarations_for_returns_the_raw_entries() -> None:
    decl = s3_surface.declarations_for(["stage_coverage"])
    assert set(decl) == {"krepis.stage_coverage"}
    assert decl["krepis.stage_coverage"] == stage_coverage.S3_SURFACE


# ── The two prefixes I8156 was opened for ────────────────────────────────────


def test_stage_coverage_declares_the_prefix_it_actually_writes() -> None:
    resolved = s3_surface.prefixes_for(["stage_coverage"])
    assert resolved[stage_coverage.VERDICT_PREFIX] == s3_surface.MODE_READWRITE
    assert resolved["_freshness_monitor"] == s3_surface.MODE_READ


def test_cost_sink_declares_the_variable_not_a_baked_in_prefix() -> None:
    kinds = {entry.kind for entry in cost_sink.S3_SURFACE}
    assert kinds == {s3_surface.KIND_ENV}
    assert cost_sink.S3_SURFACE[0].variable == cost_sink.PREFIX_ENV_VAR


# ── Anti-rot: a module that WRITES must have declared ────────────────────────


def _modules_with_write_sites() -> "list[str]":
    """Module basenames under ``src/krepis`` containing an S3 write call."""
    found = []
    for path in sorted(SRC.glob("*.py")):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in _WRITE_METHODS
                and any(kw.arg in ("Bucket", "Key") for kw in node.keywords)
            ):
                found.append(path.stem)
                break
            # upload_file(path, bucket, key) is positional, not keyword.
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("upload_file", "upload_fileobj")
            ):
                found.append(path.stem)
                break
    return found


def test_the_scan_finds_the_modules_we_know_write() -> None:
    """Guard the guard: a detector that finds nothing passes vacuously."""
    found = set(_modules_with_write_sites())
    assert {"stage_coverage", "cost_sink", "locks", "fleet_events"} <= found


def test_every_module_that_writes_to_s3_declares_a_surface() -> None:
    undeclared = sorted(
        name
        for name in _modules_with_write_sites()
        if not s3_surface.module_declares_surface(name)
    )
    assert not undeclared, (
        "krepis module(s) perform an S3 write without declaring an "
        "S3_SURFACE:\n"
        + "\n".join(f"  - krepis/{n}.py" for n in undeclared)
        + "\n\nEvery consumer's IAM contract reads these declarations to "
        "learn which prefixes a krepis import obliges its role to grant "
        "(alpha-engine-config-I8156). An undeclared writer is invisible to "
        "every one of them, and the failure mode is a fail-soft ERROR log "
        "nobody reads.\n\nResolution: add a module-level\n"
        "    S3_SURFACE = (s3_surface.literal('<top-level-prefix>'),)\n"
        "or, where the caller chooses the namespace,\n"
        "    S3_SURFACE = (s3_surface.caller_supplied('<why>'),)"
    )
