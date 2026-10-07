from dataclasses import dataclass, field

from blackbox.schemas.beliefs import (
    BeliefProvenance,
    ObjectStateHypothesis,
)
from blackbox.schemas.events import EventType, WorldEvent
from blackbox.schemas.identity import IdentityAssignment
from blackbox.schemas.perception import TrackObservation
from blackbox.schemas.uncertainty import (
    CompetingHypothesis,
    HypothesisDistribution,
)


@dataclass
class _ObjectHistory:
    events: list[WorldEvent] = field(default_factory=list)
    track_ids: set[str] = field(default_factory=set)
    last_observation: TrackObservation | None = None
    last_frame_index: int | None = None
    last_timestamp_ms: int | None = None


class WorldStateProjector:
    """Projects event history into beliefs keyed by persistent object ID."""

    def __init__(self) -> None:
        self._histories: dict[str, _ObjectHistory] = {}
        self._track_to_object: dict[str, str] = {}

    def update(
        self,
        frame_index: int,
        timestamp_ms: int,
        identity_assignments: list[IdentityAssignment],
        events: list[WorldEvent],
    ) -> None:
        for assignment in identity_assignments:
            self._track_to_object[assignment.track_id] = assignment.object_id

            history = self._histories.setdefault(
                assignment.object_id,
                _ObjectHistory(),
            )
            history.track_ids.add(assignment.track_id)
            history.last_observation = assignment.observation
            history.last_frame_index = frame_index
            history.last_timestamp_ms = timestamp_ms

        for event in events:
            object_id = self._track_to_object.get(
                event.subject_track_id
            )

            if object_id is None:
                continue

            history = self._histories.setdefault(
                object_id,
                _ObjectHistory(),
            )
            history.track_ids.add(event.subject_track_id)
            history.events.append(event)

    def history_for(
        self,
        object_id: str,
    ) -> tuple[WorldEvent, ...]:
        history = self._histories.get(object_id)
        if history is None:
            return ()
        return tuple(history.events)

    def tracks_for(
        self,
        object_id: str,
    ) -> tuple[str, ...]:
        history = self._histories.get(object_id)
        if history is None:
            return ()
        return tuple(sorted(history.track_ids))

    def beliefs_for(
        self,
        object_id: str,
    ) -> tuple[ObjectStateHypothesis, ...]:
        history = self._histories.get(object_id)
        if history is None:
            return ()

        beliefs: list[ObjectStateHypothesis] = []
        evidence_track_ids = tuple(sorted(history.track_ids))

        if history.last_observation is not None:
            beliefs.append(
                ObjectStateHypothesis(
                    subject_object_id=object_id,
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
                    evidence_track_ids=evidence_track_ids,
                )
            )

        if history.events:
            latest_event = history.events[-1]

            if latest_event.event_type == EventType.OBJECT_DISAPPEARED:
                visibility_value = "not_observed"
                visibility_provenance = BeliefProvenance.INFERRED
                visibility_confidence = latest_event.confidence
            else:
                visibility_value = "observed"
                visibility_provenance = BeliefProvenance.OBSERVED
                visibility_confidence = 1.0

            beliefs.append(
                ObjectStateHypothesis(
                    subject_object_id=object_id,
                    predicate="CURRENT_VISIBILITY",
                    value=visibility_value,
                    confidence=visibility_confidence,
                    provenance=visibility_provenance,
                    updated_frame_index=latest_event.frame_index,
                    updated_timestamp_ms=latest_event.timestamp_ms,
                    supporting_event_types=(
                        latest_event.event_type.value,
                    ),
                    evidence_track_ids=evidence_track_ids,
                )
            )

        return tuple(beliefs)

    def uncertainty_for(
        self,
        object_id: str,
    ) -> tuple[HypothesisDistribution, ...]:
        history = self._histories.get(object_id)
        if history is None or not history.events:
            return ()

        latest_event = history.events[-1]

        if latest_event.event_type != EventType.OBJECT_DISAPPEARED:
            return ()

        distribution = HypothesisDistribution(
            subject_object_id=object_id,
            predicate="UNOBSERVED_CHANGE_STATE",
            hypotheses=(
                CompetingHypothesis(
                    value="unchanged_since_last_observation",
                    weight=0.5,
                    provenance="baseline_prior",
                    rationale=(
                        "No post-disappearance evidence currently "
                        "distinguishes unchanged from changed."
                    ),
                ),
                CompetingHypothesis(
                    value="changed_while_unobserved",
                    weight=0.5,
                    provenance="baseline_prior",
                    rationale=(
                        "No post-disappearance evidence currently "
                        "distinguishes unchanged from changed."
                    ),
                ),
            ),
            updated_frame_index=latest_event.frame_index,
            updated_timestamp_ms=latest_event.timestamp_ms,
        )

        return (distribution,)

    def all_beliefs(
        self,
    ) -> dict[str, tuple[ObjectStateHypothesis, ...]]:
        return {
            object_id: self.beliefs_for(object_id)
            for object_id in self._histories
        }

    def all_uncertainty(
        self,
    ) -> dict[str, tuple[HypothesisDistribution, ...]]:
        return {
            object_id: self.uncertainty_for(object_id)
            for object_id in self._histories
            if self.uncertainty_for(object_id)
        }
