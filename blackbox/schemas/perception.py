from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    bbox: BoundingBox


@dataclass(frozen=True)
class TrackObservation:
    track_id: str
    class_name: str
    confidence: float
    bbox: BoundingBox


@dataclass(frozen=True)
class Observation:
    frame_index: int
    timestamp_ms: int

    track_id: str
    object_id: Optional[str]

    class_name: str
    confidence: float

    bbox: BoundingBox
    visible: bool = True