"""Entity model for the memory layer."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Entity:
    label: str
    asserted_at: datetime
    wikidata_id: str | None = None
    description: str | None = None
    wikipedia_url: str | None = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str = "concept"
    subtype: str | None = None
    canonical_label: str | None = None
    canonical_key: str | None = None
    extraction_count: int = 1  # times asserted/re-asserted by extraction (write-side reinforcement)
    retrieval_count: int = 0  # times surfaced in a briefing (read-side usage)
