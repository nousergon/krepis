"""The cost record's declared column contract, pinned on the producer side.

**The defect** (`alpha-engine-config-I7393`). Three places downstream of this
library declare the cost artifact's schema, and all three agree:

* `nousergon-lib/src/nousergon_lib/transparency_inventory.yaml`, row
  ``cost_telemetry`` — ``assert_columns_present: [schema_version, run_id,
  agent_id, model_name]``;
* `crucible-research/scripts/aggregate_costs.py` — its module docstring lists
  ``schema_version``, ``run_id``, ``agent_id`` and ``model_name`` as the
  schema, and it builds the parquet with ``pd.DataFrame(rows)`` straight from
  these records;
* `crucible-backtester/analysis/cost_report.py` — groups by ``model_name``
  (line 143) and ``agent_id`` (line 145).

`krepis.cost.record_llm_call` emitted none of the four. Measured on the live
artifact ``s3://alpha-engine-research/decision_artifacts/_cost/2026-08-10/
cost.parquet`` (8 rows), whose columns were::

    ts, provider, model, input_tokens, output_tokens, cache_read_tokens,
    cache_create_tokens, cache_create_1h_tokens, prompt_cache_miss_tokens,
    web_search_requests, web_fetch_requests, cost_usd, cost_source, callsite_id

Two of the four were naming drift (``model``/``model_name``,
``callsite_id``/``agent_id``) and two were genuinely absent.

**Why it stayed invisible for so long.** Every consumer degrades silently.
``cost_report._group_sum`` returns ``{}`` for a missing column, so the "By
model" and "By agent_id" sections of the weekly cost report rendered EMPTY
rather than erroring — a section with no rows looks like a week with no spend.
The only thing that ever complained was the substrate health check's
``assert_columns_present``, which is how it finally surfaced: as the single
``[FAIL]`` in an otherwise-green weekly health report.

**What this module holds.** The producer emits every column the contract
names. Additive only — ``model`` and ``callsite_id`` are still emitted, so
consumers reading history keep working; a rename would have fixed the
contract readers by breaking everyone else.
"""

from __future__ import annotations

import json

import pytest

from krepis.cost import COST_RECORD_SCHEMA_VERSION, record_llm_call
from krepis.cost_sink import S3JsonlCostSink

#: The exact set `nousergon-lib`'s transparency_inventory.yaml asserts on the
#: `cost_telemetry` row. Duplicated here deliberately: krepis cannot import
#: that repo, and a contract test that derives the expectation from the code
#: under test asserts nothing. If this list and the inventory ever disagree,
#: the substrate health check fails and names the column — which is exactly
#: how this defect was found.
DECLARED_CONTRACT_COLUMNS = ("schema_version", "run_id", "agent_id", "model_name")

#: Emitted before the contract columns existed. Still emitted, because every
#: row written to date carries them and consumers read history.
LEGACY_COLUMNS_KEPT = ("model", "callsite_id")


class _FakeUsage:
    input_tokens = 10
    output_tokens = 5
    cache_read_input_tokens = 0
    cache_creation_input_tokens = 0


class _FakeMessage:
    """Minimal Anthropic-shaped message `record_llm_call` can price."""

    model = "claude-haiku-4-5"   # priced in the default card; see tests/test_cost_llm.py
    usage = _FakeUsage()


def _record(**extra):
    return record_llm_call(
        _FakeMessage(), extra_fields={"callsite_id": "unit-test-callsite", **extra}
    )


def test_schema_version_is_emitted_and_is_the_declared_constant():
    rec = _record()
    assert rec["schema_version"] == COST_RECORD_SCHEMA_VERSION


def test_model_name_is_emitted_alongside_model():
    """Both names, same value. `model_name` is what the contract and
    cost_report read; `model` is what every historical row carries."""
    rec = _record()
    assert rec["model_name"] == rec["model"]
    assert rec["model_name"]


def test_agent_id_mirrors_callsite_id():
    rec = _record()
    assert rec["agent_id"] == "unit-test-callsite"
    assert rec["callsite_id"] == "unit-test-callsite"


def test_an_explicit_agent_id_is_not_overwritten():
    """The mirror fills a gap; it never overrides a caller that knows better."""
    rec = _record(agent_id="explicitly-set")
    assert rec["agent_id"] == "explicitly-set"
    assert rec["callsite_id"] == "unit-test-callsite"


def test_legacy_columns_are_still_emitted():
    """Additive, not a rename. Dropping these would fix the contract readers
    by breaking every consumer of the existing artifacts."""
    rec = _record()
    for col in LEGACY_COLUMNS_KEPT:
        assert col in rec, f"{col} was dropped — this change must stay additive"


@pytest.mark.parametrize("column", DECLARED_CONTRACT_COLUMNS)
def test_every_declared_contract_column_reaches_the_written_row(column, monkeypatch):
    """The load-bearing test: assert on what the SINK WRITES, not on what the
    builder returns.

    `run_id` is added by the sink, not by `record_llm_call` — it is the only
    layer that knows it, because it partitions by it. A test that checked only
    the builder's dict would pass while `run_id` never reached the parquet,
    which is the precise shape of the original defect: a value present in the
    S3 KEY and absent from the ROW.
    """
    written: list[dict] = []

    class _FakeS3:
        def put_object(self, **kwargs):
            for line in kwargs["Body"].decode().splitlines():
                written.append(json.loads(line))
            return {}

    sink = S3JsonlCostSink(
        bucket="test-bucket",
        prefix="decision_artifacts/_cost_raw",
        run_id="2026-08-15T09:00Z-abc123",
        s3_client=_FakeS3(),
    )
    sink(_record())   # the sink IS the callable — see S3JsonlCostSink.__call__
    sink.flush()

    assert written, "the sink wrote no rows — this test proves nothing as written"
    for row in written:
        assert column in row, (
            f"{column!r} is declared by nousergon-lib's transparency_inventory.yaml "
            "cost_telemetry row but never reaches the written record. The weekly "
            "substrate health check asserts these columns present and will FAIL "
            "(alpha-engine-config-I7393)."
        )
        assert row[column] not in (None, ""), (
            f"{column!r} is present but empty — assert_columns_present would pass "
            "while the column carries nothing a consumer can group by."
        )
