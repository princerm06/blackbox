from dataclasses import dataclass, field

from blackbox.schemas.beliefs import (
    BeliefProvenance,
    ObjectStateHypothesis,
)
from blackbox.schemas.events import EventType, WorldEvent
from blackbox.schemas.perception import TrackObservation


@dataclass
class _ObjectHistory:
    events: list[WorldEvent] = field(default_factory=list)
    last_observation: TrackObservation | None = None
    last_frame_index: int | None = None
    last_timestamp_ms: int | None = None


class WorldStateProjector:
    """Minimal evidence-grounded projection over track histories."""

    def __init__(self) -> None:
        self._histories: dict[str, _ObjectHistory] = {}

    def update(
        self,
        frame_index: int,
        timestamp_ms: int,
        observations: list[TrackObservation],
        events: list[WorldEvent],
    ) -> None:
        for event in events:
            history = self._histories.setdefault(
                event.subject_track_id,
                _ObjectHistory(),
            )
            history.events.append(event)

        for observation in observations:
            history = self._histories.setdefault(
                observation.track_id,
                _ObjectHistory(),
            )
            history.last_observation = observation
            history.last_frame_index = frame_index
            history.last_timestamp_ms = timestamp_ms

    def history_for(
        self,
        track_id: str,
    ) -> tuple[WorldEvent, ...]:
        history = self._histories.get(track_id)
        if history is None:
            return ()
        return tuple(history.events)

    def beliefs_for(
        self,
        track_id: str,
    ) -> tuple[ObjectStateHypothesis, ...]:
        history = self._histories.get(track_id)
        if history is None:
            return ()

        beliefs: list[ObjectStateHypothesis] = []

        if history.last_observation is not None:
            beliefs.append(
                ObjectStateHypothesis(
                    subject_track_id=track_id,
                    predicate="LAST_OBSERVED_BBOX",
                    value={
                        "x1": history.last_observation.bbox.x1,
                        "y1": history.last_observation.bbox.y1,
                        "x2": history.last_observation.bbox.x2,
                        "y2": history.last_observation.bbox.y2,
                    },
                    confidence=history.last_observation.confidence,
                    provenance=BeliefProvenance.OBSERVED,
                    updated_frame_index=history.last_frame_index or 0,
                    updated_timestamp_ms=history.last_timestamp_ms or 0,
                    supporting_event_types=tuple(
                        event.event_type.value
                        for event in history.events
                        if event.event_type
                        in {
                            EventType.OBJECT_APPEARED,
                            EventType.OBJECT_MOVED,
                            EventType.OBJECT_REAPPEARED,
                        }
                    ),
                )
            )

        if history.events:
            latest_event = history.events[-1]

            if latest_event.event_type == EventType.OBJECT_DISAPPEARED:
                beliefs.append(
                    ObjectStateHypothesis(
                        subject_track_id=track_id,
                        predicate="CURRENT_VISIBILITY",
                        value="not_observed",
                        confidence=latest_event.confidence,
                        provenance=BeliefProvenance.INFERRED,
                        updated_frame_index=latest_event.frame_index,
                        updated_timestamp_ms=latest_event.timestamp_ms,
                        supporting_event_types=(
                            latest_event.event_type.value,
                        ),
                    )
                )
            else:
                beliefs.append(
                    ObjectStateHypothesis(
                        subject_track_id=track_id,
                        predicate="CURRENT_VISIBILITY",
                        value="observed",
                        confidence=1.0,
                        provenance=BeliefProvenance.OBSERVED,
                        updated_frame_index=latest_event.frame_index,
                        updated_timestamp_ms=latest_event.timestamp_ms,
                        supporting_event_types=(
                            latest_event.event_type.value,
                        ),
                    )
                )

        return tuple(beliefs)

    def all_beliefs(
        self,
    ) -> dict[str, tuple[ObjectStateHypothesis, ...]]:
        return {
            track_id: self.beliefs_for(track_id)
            for track_id in self._histories
        }
