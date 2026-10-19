"""Persistence entry points for structured extraction results."""

from tpasini.recordatio.extraction.extractor import ExtractionResult
from tpasini.recordatio.proxy import extraction_pipeline
from tpasini.recordatio.proxy.extraction_pipeline import ExtractionPipelineContext


async def persist_side(
    ctx: ExtractionPipelineContext,
    result: ExtractionResult,
    agent_id: str,
    source_agent_id: str | None = None,
) -> None:
    await extraction_pipeline._persist_side(ctx, result, agent_id, source_agent_id=source_agent_id)


_persist_side = persist_side
