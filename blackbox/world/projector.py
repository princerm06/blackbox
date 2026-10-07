from dataclasses import dataclass, field

from blackbox.schemas.beliefs import (
    BeliefProvenance,
    ObjectStateHypothesis,
)
from blackbox.schemas.events import EventType, WorldEvent
from blackbox.schemas.identity import IdentityAssignment
from blackbox.schemas.opportunities import (
    OpportunitySignal,
    OpportunityType,
)
from blackbox.schemas.perception import TrackObservation
from blackbox.schemas.uncertainty import (
    CompetingHypothesis,
    HypothesisDistribution,
)


_EVENT_OPPORTUNITY_STRENGTHS = {
    EventType.OBJECT_MOVED: (
        OpportunityType.OTHER_OBJECT_MOVED,
        0.08,
    ),
    EventType.OBJECT_APPEARED: (
        OpportunityType.OTHER_OBJECT_APPEARED,
        0.03,
    ),
    EventType.OBJECT_DISAPPEARED: (
        OpportunityType.OTHER_OBJECT_DISAPPEARED,
        0.05,
    ),
    EventType.OBJECT_REAPPEARED: (
        OpportunityType.OTHER_OBJECT_REAPPEARED,
        0.04,
    ),
}


@dataclass
class _ObjectHistory:
    events: list[WorldEvent] = field(default_factory=list)
    opportunities: list[OpportunitySignal] = field(default_factory=list)
    track_ids: set[str] = field(default_factory=set)
    last_observation: TrackObservation | None = None
    last_frame_index: int | None = None
    last_timestamp_ms: int | None = None


class WorldStateProjector:
    """Projects event history into object-level beliefs and uncertainty.

    The uncertainty updater is a deliberately simple heuristic baseline, not a
    calibrated probability model. It exists so later experiments can measure
    whether learned or calibrated belief evolution improves on it.
    """

    def __init__(self) -> None:
        self._histories: dict[str, _ObjectHistory] = {}
        self._track_to_object: dict[str, str] = {}
        self._current_frame_index = 0
        self._current_timestamp_ms = 0

    def update(
        self,
        frame_index: int,
        timestamp_ms: int,
        identity_assignments: list[IdentityAssignment],
        events: list[WorldEvent],
    ) -> None:
        self._current_frame_index = frame_index
        self._current_timestamp_ms = timestamp_ms

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

        mapped_events: list[tuple[str, WorldEvent]] = []

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
            mapped_events.append((object_id, event))

        hidden_object_ids = {
            object_id
            for object_id, history in self._histories.items()
            if self._is_currently_unobserved(history)
        }

        for target_object_id in hidden_object_ids:
            target_history = self._histories[target_object_id]

            for source_object_id, event in mapped_events:
                if source_object_id == target_object_id:
                    continue

                signal_spec = _EVENT_OPPORTUNITY_STRENGTHS.get(
                    event.event_type
                )
                if signal_spec is None:
                    continue

                signal_type, strength = signal_spec

                target_history.opportunities.append(
                    OpportunitySignal(
                        target_object_id=target_object_id,
                        signal_type=signal_type,
                        strength=strength,
                        frame_index=frame_index,
                        timestamp_ms=timestamp_ms,
                        source_object_id=source_object_id,
                        source_track_id=event.subject_track_id,
                        rationale=(
                            f"{event.event_type.value} occurred for "
                            f"{source_object_id} while "
                            f"{target_object_id} was unobserved."
                        ),
                    )
                )

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

    def opportunities_for(
        self,
        object_id: str,
    ) -> tuple[OpportunitySignal, ...]:
        history = self._histories.get(object_id)
        if history is None or not self._is_currently_unobserved(history):
            return ()

        disappearance = history.events[-1]
        unseen_frames = max(
            0,
            self._current_frame_index - disappearance.frame_index,
        )
        elapsed_strength = min(0.15, unseen_frames * 0.005)

        signals = list(history.opportunities)

        if unseen_frames > 0:
            signals.append(
                OpportunitySignal(
                    target_object_id=object_id,
                    signal_type=OpportunityType.ELAPSED_UNSEEN,
                    strength=elapsed_strength,
                    frame_index=self._current_frame_index,
                    timestamp_ms=self._current_timestamp_ms,
                    source_object_id=None,
                    source_track_id=None,
                    rationale=(
                        f"{object_id} has been unobserved for "
                        f"{unseen_frames} frame(s)."
                    ),
                )
            )

        return tuple(signals)

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
        if history is None or not self._is_currently_unobserved(history):
            return ()

        signals = self.opportunities_for(object_id)
        opportunity_strength = min(
            0.45,
            sum(signal.strength for signal in signals),
        )

        changed_weight = 0.5 + opportunity_strength
        unchanged_weight = 1.0 - changed_weight

        if signals:
            rationale = (
                f"{len(signals)} opportunity signal(s) increased the "
                "heuristic baseline probability of unobserved change."
            )
        else:
            rationale = (
                "No opportunity-for-change signals have accumulated since "
                "the object disappeared."
            )

        distribution = HypothesisDistribution(
            subject_object_id=object_id,
            predicate="UNOBSERVED_CHANGE_STATE",
            hypotheses=(
                CompetingHypothesis(
                    value="unchanged_since_last_observation",
                    weight=unchanged_weight,
                    provenance="heuristic_baseline",
                    rationale=rationale,
                ),
                CompetingHypothesis(
                    value="changed_while_unobserved",
                    weight=changed_weight,
                    provenance="heuristic_baseline",
                    rationale=rationale,
                ),
            ),
            updated_frame_index=self._current_frame_index,
            updated_timestamp_ms=self._current_timestamp_ms,
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

    @staticmethod
    def _is_currently_unobserved(
        history: _ObjectHistory,
    ) -> bool:
        return bool(
            history.events
            and history.events[-1].event_type
            == EventType.OBJECT_DISAPPEARED
        )
