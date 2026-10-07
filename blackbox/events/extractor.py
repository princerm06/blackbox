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


def _overlap_fraction_of_smaller(
    first: BoundingBox,
    second: BoundingBox,
) -> float:
    x1 = max(first.x1, second.x1)
    y1 = max(first.y1, second.y1)
    x2 = min(first.x2, second.x2)
    y2 = min(first.y2, second.y2)

    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    intersection = width * height
    smaller_area = min(first.area, second.area)

    if smaller_area <= 0.0:
        return 0.0

    return intersection / smaller_area


@dataclass
class _TrackEventState:
    class_name: str
    bbox: BoundingBox
    visible: bool


class EventExtractor:
    """Converts tracked observations into evidence-grounded events.

    Person-object interaction is an inferred visual relation based on spatial
    overlap. It is not treated as proof that a transfer or containment occurred.
    """

    def __init__(
        self,
        movement_threshold_px: float = 20.0,
        interaction_overlap_threshold: float = 0.15,
    ) -> None:
        if movement_threshold_px < 0.0:
            raise ValueError(
                "movement_threshold_px must be non-negative"
            )
        if not 0.0 <= interaction_overlap_threshold <= 1.0:
            raise ValueError(
                "interaction_overlap_threshold must be between 0 and 1"
            )

        self.movement_threshold_px = movement_threshold_px
        self.interaction_overlap_threshold = interaction_overlap_threshold
        self._states: dict[str, _TrackEventState] = {}
        self._active_interactions: set[tuple[str, str]] = set()

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
                        provenance=EventProvenance.INFERRED,
                        metadata={
                            "meaning": (
                                "track not observed in current frame; "
                                "this does not establish physical departure"
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

        events.extend(
            self._interaction_events(
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                observations=observations,
            )
        )

        return events

    def _interaction_events(
        self,
        frame_index: int,
        timestamp_ms: int,
        observations: list[TrackObservation],
    ) -> list[WorldEvent]:
        people = [
            observation
            for observation in observations
            if observation.class_name == "person"
        ]
        objects = [
            observation
            for observation in observations
            if observation.class_name != "person"
        ]

        current_interactions: set[tuple[str, str]] = set()
        events: list[WorldEvent] = []

        for person in people:
            for obj in objects:
                overlap = _overlap_fraction_of_smaller(
                    person.bbox,
                    obj.bbox,
                )

                if overlap < self.interaction_overlap_threshold:
                    continue

                pair = (person.track_id, obj.track_id)
                current_interactions.add(pair)

                if pair in self._active_interactions:
                    continue

                confidence = min(
                    person.confidence,
                    obj.confidence,
                    max(overlap, 0.01),
                )

                events.append(
                    WorldEvent(
                        event_type=EventType.PERSON_OBJECT_INTERACTION,
                        frame_index=frame_index,
                        timestamp_ms=timestamp_ms,
                        subject_track_id=obj.track_id,
                        class_name=obj.class_name,
                        confidence=confidence,
                        provenance=EventProvenance.INFERRED,
                        metadata={
                            "person_track_id": person.track_id,
                            "overlap_fraction_of_smaller": overlap,
                            "meaning": (
                                "person and object overlap spatially; "
                                "interaction is inferred, not directly proven"
                            ),
                        },
                        related_track_ids=(person.track_id,),
                    )
                )

        self._active_interactions = current_interactions
        return events
