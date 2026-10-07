from dataclasses import dataclass
from enum import Enum
from typing import Any


class BeliefProvenance(str, Enum):
    OBSERVED = "observed"
    INFERRED = "inferred"


@dataclass(frozen=True)
class ObjectStateHypothesis:
    subject_object_id: str
    predicate: str
    value: Any
    confidence: float
    provenance: BeliefProvenance
    updated_frame_index: int
    updated_timestamp_ms: int
    supporting_event_types: tuple[str, ...]
    evidence_track_ids: tuple[str, ...]
