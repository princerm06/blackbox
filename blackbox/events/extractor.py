from dataclasses import dataclass

from blackbox.schemas.events import (
    EventProvenance,
    EventType,
    WorldEvent,
)
from blackbox.schemas.perception import BoundingBox, TrackObservation


def _center(bbox: BoundingBox) -> tuple[float, float]:
    return (
        (bbox.x1 + bbox.x2) / 2.0,
        (bbox.y1 + bbox.y2) / 2.0,
    )


def _center_distance(
    first: BoundingBox,
    second: BoundingBox,
) -> float:
    first_x, first_y = _center(first)
    second_x, second_y = _center(second)

    dx = second_x - first_x
    dy = second_y - first_y

    return (dx * dx + dy * dy) ** 0.5


@dataclass
class _TrackEventState:
    class_name: str
    bbox: BoundingBox
    visible: bool


class EventExtractor:
    """Converts tracked observations into simple, evidence-grounded events.

    A disappearance event means a track stopped being observed in the current
    frame. It does not claim the object left the room or changed physical state.
    """

    def __init__(
        self,
        movement_threshold_px: float = 20.0,
    ) -> None:
        if movement_threshold_px < 0.0:
            raise ValueError(
                "movement_threshold_px must be non-negative"
            )

        self.movement_threshold_px = movement_threshold_px
        self._states: dict[str, _TrackEventState] = {}

    def update(
        self,
        frame_index: int,
        timestamp_ms: int,
        observations: list[TrackObservation],
    ) -> list[WorldEvent]:
        current_by_track = {
            observation.track_id: observation
            for observation in observations
        }

        events: list[WorldEvent] = []

        for track_id, state in self._states.items():
            if state.visible and track_id not in current_by_track:
                state.visible = False

                events.append(
                    WorldEvent(
                        event_type=EventType.OBJECT_DISAPPEARED,
                        frame_index=frame_index,
                        timestamp_ms=timestamp_ms,
                        subject_track_id=track_id,
                        class_name=state.class_name,
                        confidence=1.0,
                        provenance=EventProvenance.OBSERVED,
                        metadata={
                            "meaning": (
                                "track not observed in current frame"
                            ),
                        },
                    )
                )

        for observation in observations:
            previous = self._states.get(observation.track_id)

            if previous is None:
                self._states[observation.track_id] = _TrackEventState(
                    class_name=observation.class_name,
                    bbox=observation.bbox,
                    visible=True,
                )

                events.append(
                    WorldEvent(
                        event_type=EventType.OBJECT_APPEARED,
                        frame_index=frame_index,
                        timestamp_ms=timestamp_ms,
                        subject_track_id=observation.track_id,
                        class_name=observation.class_name,
                        confidence=observation.confidence,
                        provenance=EventProvenance.OBSERVED,
                        metadata={},
                    )
                )
                continue

            if not previous.visible:
                previous.visible = True
                previous.class_name = observation.class_name
                previous.bbox = observation.bbox

                events.append(
                    WorldEvent(
                        event_type=EventType.OBJECT_REAPPEARED,
                        frame_index=frame_index,
                        timestamp_ms=timestamp_ms,
                        subject_track_id=observation.track_id,
                        class_name=observation.class_name,
                        confidence=observation.confidence,
                        provenance=EventProvenance.OBSERVED,
                        metadata={},
                    )
                )
                continue

            distance_px = _center_distance(
                previous.bbox,
                observation.bbox,
            )

            if distance_px >= self.movement_threshold_px:
                events.append(
                    WorldEvent(
                        event_type=EventType.OBJECT_MOVED,
                        frame_index=frame_index,
                        timestamp_ms=timestamp_ms,
                        subject_track_id=observation.track_id,
                        class_name=observation.class_name,
                        confidence=observation.confidence,
                        provenance=EventProvenance.OBSERVED,
                        metadata={
                            "center_distance_px": distance_px,
                        },
                    )
                )

            previous.class_name = observation.class_name
            previous.bbox = observation.bbox
            previous.visible = True

        return events
