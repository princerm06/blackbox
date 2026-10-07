from dataclasses import dataclass
from enum import Enum


class OpportunityType(str, Enum):
    ELAPSED_UNSEEN = "ELAPSED_UNSEEN"
    OTHER_OBJECT_MOVED = "OTHER_OBJECT_MOVED"
    OTHER_OBJECT_APPEARED = "OTHER_OBJECT_APPEARED"
    OTHER_OBJECT_DISAPPEARED = "OTHER_OBJECT_DISAPPEARED"
    OTHER_OBJECT_REAPPEARED = "OTHER_OBJECT_REAPPEARED"


@dataclass(frozen=True)
class OpportunitySignal:
    target_object_id: str
    signal_type: OpportunityType
    strength: float
    frame_index: int
    timestamp_ms: int
    source_object_id: str | None
    source_track_id: str | None
    rationale: str
