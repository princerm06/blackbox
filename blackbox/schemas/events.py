from dataclasses import dataclass
from enum import Enum
from typing import Any


class EventType(str, Enum):
    OBJECT_APPEARED = "OBJECT_APPEARED"
    OBJECT_MOVED = "OBJECT_MOVED"
    OBJECT_DISAPPEARED = "OBJECT_DISAPPEARED"
    OBJECT_REAPPEARED = "OBJECT_REAPPEARED"


class EventProvenance(str, Enum):
    OBSERVED = "observed"
    INFERRED = "inferred"


@dataclass(frozen=True)
class WorldEvent:
    event_type: EventType
    frame_index: int
    timestamp_ms: int
    subject_track_id: str
    class_name: str
    confidence: float
    provenance: EventProvenance
    metadata: dict[str, Any]
