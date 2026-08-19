"""``krepis[flow_doctor]`` must never pull the ``anthropic`` distribution.

alpha-engine-config-I7722: the extra previously read
``flow-doctor[diagnosis,s3]>=0.8.1,<0.14``. ``flow-doctor[diagnosis]``
resolved to ``anthropic>=0.40``, so every ``krepis[flow_doctor]`` consumer
installed the Anthropic SDK transitively — including ``crucible-executor``,
which had deliberately removed direct LLM exposure as a guardrail
("executor should not have any llm exposure") five months earlier. Nothing
detected the reintroduction for ~3 months.

``krepis.logging`` (the only consumer of this extra inside krepis) touches
exactly two names off the bare ``flow_doctor`` package root —
``flow_doctor.FlowDoctor.from_config()`` and
``flow_doctor.FlowDoctorHandler(...)`` — both base-package exports needing
no extra. Both tests below read the *live* ``pyproject.toml`` declaration
rather than a hardcoded copy of it, so a future hand-edit that reintroduces
a forcing extra is caught here without this file also needing an edit.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import pytest

_PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _flow_doctor_extra_requirement() -> str:
    """Return the literal requirement string(s) declared for the
    ``flow_doctor`` optional-dependency extra in ``pyproject.toml``.

    Regex, not tomllib — matches the stdlib-free convention already used by
    ``tests/test_version_pin.py`` for reading this same file, and keeps the
    3.9 floor honest without a tomllib backport dependency.
    """
    text = _PYPROJECT.read_text()
    match = re.search(
        r'^flow_doctor\s*=\s*\[\s*"([^"]+)"\s*\]', text, re.MULTILINE
    )
    assert match is not None, (
        "flow_doctor extra not found in pyproject.toml under "
        "[project.optional-dependencies] — has it been renamed or reshaped?"
    )
    return match.group(1)


def test_flow_doctor_extra_declaration_names_no_forcing_sub_extra():
    """Static guard, no network: the declared requirement string must not
    name a flow-doctor sub-extra (``[diagnosis,...]`` etc.) that would pull
    in a vendor SDK or storage backend krepis's own glue does not use.

    krepis.logging imports only ``flow_doctor.FlowDoctor`` and
    ``flow_doctor.FlowDoctorHandler`` (both base-package exports) — any
    bracketed sub-extra on this line is by definition forcing something
    onto a consumer that krepis itself does not need.
    """
    requirement = _flow_doctor_extra_requirement()
    assert "flow-doctor[" not in requirement.replace(" ", ""), (
        f"flow_doctor extra = {requirement!r} names a flow-doctor sub-extra "
        f"(e.g. flow-doctor[diagnosis,s3]) — that forces the sub-extra's "
        f"deps (anthropic, boto3, ...) onto every krepis[flow_doctor] "
        f"consumer. krepis.logging only needs the bare flow_doctor package "
        f"root; a diagnosis/storage transport is a consumer's own choice "
        f"via their own flow-doctor.yaml, declared in the consumer's own "
        f"requirements, not here (alpha-engine-config-I7722)."
    )


def _network_reachable() -> bool:
    try:
        urllib.request.urlopen("https://pypi.org/simple/flow-doctor/", timeout=5)
        return True
    except OSError:
        return False


def test_flow_doctor_extra_resolves_without_anthropic():
    """End-to-end guard: actually resolve ``krepis[flow_doctor]`` and assert
    no ``anthropic`` distribution appears anywhere in the resolved set.

    Uses ``pip install --dry-run --ignore-installed --report`` so the
    resolution is real (reads flow-doctor's own declared deps off whatever
    version the floor/cap in pyproject.toml admits) without installing
    anything. ``--ignore-installed`` is load-bearing: without it pip skips
    resolving any distribution already present in the running environment
    ("Requirement already satisfied") and silently omits it from the
    report, which would make this test blind on a machine that already has
    flow-doctor (or anthropic) installed globally — exactly the kind of
    environment krepis's own CI and every developer laptop already is.
    """
    if not _network_reachable():
        pytest.skip("PyPI unreachable from this environment")

    with tempfile.NamedTemporaryFile(
        suffix=".json", delete=False
    ) as report_file:
        report_path = Path(report_file.name)
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--dry-run",
                "--ignore-installed",
                "--quiet",
                "--report",
                str(report_path),
                f"{_PYPROJECT.parent}[flow_doctor]",
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert proc.returncode == 0, (
            f"pip resolution of krepis[flow_doctor] failed "
            f"(rc={proc.returncode}):\nstdout={proc.stdout}\nstderr={proc.stderr}"
        )
        report = json.loads(report_path.read_text())
        resolved_names = sorted(
            item["metadata"]["name"].lower() for item in report["install"]
        )
        assert "anthropic" not in resolved_names, (
            f"krepis[flow_doctor] resolved to include the anthropic "
            f"distribution — full resolved set: {resolved_names}. This is "
            f"exactly the alpha-engine-config-I7722 regression: a "
            f"transitive extra silently forcing the Anthropic SDK onto "
            f"every krepis[flow_doctor] consumer, including repos "
            f"(crucible-executor) that deliberately carry zero LLM "
            f"exposure as a guardrail."
        )
    finally:
        report_path.unlink(missing_ok=True)
