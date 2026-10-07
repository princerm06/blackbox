from dataclasses import dataclass
from enum import Enum


class RelationType(str, Enum):
    SPATIALLY_ASSOCIATED = "SPATIALLY_ASSOCIATED"


@dataclass(frozen=True)
class ObjectRelation:
    first_object_id: str
    second_object_id: str
    relation_type: RelationType
    confidence: float
    frame_index: int
    timestamp_ms: int
    evidence_track_ids: tuple[str, str]
