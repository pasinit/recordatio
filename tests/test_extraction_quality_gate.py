"""Regression: the extraction quality gate must not silently drop every unit.

Production evidence (Aug 3-10) showed active agents persisting zero knowledge
units for days: the persist-side quality gate (``_has_target_evidence``) was
discarding every extracted unit on every turn while the extraction LLM call
still logged ``ok=1`` with cost. The drop was logged only at DEBUG, so the
systematic failure was invisible. This test pins the observability contract:
when a non-empty extraction has *all* of its units dropped by the gate, a
WARNING must be emitted.
"""

import logging
from datetime import UTC, datetime
from typing import Any

from tpasini.recordatio.extraction.contracts import ExtractionSide
from tpasini.recordatio.extraction.extractor import _coerce_extraction_result
from tpasini.recordatio.proxy.extraction_pipeline import _filter_extraction_result


def _coerce(payload: dict[str, Any]):
    return _coerce_extraction_result(
        payload,
        now=datetime(2026, 8, 8, tzinfo=UTC),
        relation_types=[],
        topics=[],
        input_tokens=11,
        output_tokens=7,
        model="test-model",
        side=ExtractionSide.OUTPUT,
    )


def _unit_with_evidence_source(source: str) -> dict[str, Any]:
    return {
        "units": [
            {
                "id": "unit:1",
                "type": "semantic",
                "content": "Still tracking these three properties.",
                "provenance": "extraction",
                "confidence": "0.91",
                "evidence": [{"source": source, "quote": "still tracking"}],
            }
        ]
    }


def test_all_units_dropped_emits_warning(caplog) -> None:
    # A unit whose only evidence carries a non-"target" source is dropped by
    # _has_target_evidence — exactly the silent-drop shape from production.
    result = _coerce(_unit_with_evidence_source("assistant"))
    assert result.units, "fixture must start with at least one extracted unit"

    with caplog.at_level(logging.WARNING, logger="tpasini.recordatio.proxy.extraction_pipeline"):
        filtered = _filter_extraction_result(result)

    assert filtered.units == [], "gate should drop the non-target-evidence unit"
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, "dropping every extracted unit must emit a WARNING, not just DEBUG"
    assert "dropped all" in warnings[0].getMessage()


def test_kept_units_do_not_warn(caplog) -> None:
    # A well-formed target-sourced unit survives the gate and emits no warning.
    result = _coerce(_unit_with_evidence_source("target"))

    with caplog.at_level(logging.WARNING, logger="tpasini.recordatio.proxy.extraction_pipeline"):
        filtered = _filter_extraction_result(result)

    assert len(filtered.units) == 1
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
