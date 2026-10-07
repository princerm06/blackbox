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
from blackbox.schemas.relations import ObjectRelation
from blackbox.schemas.uncertainty import (
    CompetingHypothesis,
    HypothesisDistribution,
)
from blackbox.world.change_model import (
    HeuristicUnobservedChangeModel,
    UnobservedChangeModel,
)
from blackbox.world.relations import infer_spatial_relations


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
    EventType.PERSON_OBJECT_INTERACTION: (
        OpportunityType.PERSON_OBJECT_INTERACTION,
        0.18,
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

    Relation-aware weighting is intentionally heuristic. Spatial association
    only changes the strength of a change opportunity; it does not establish
    containment, transfer, or causal interaction.
    """

    def __init__(
        self,
        max_relation_age_frames: int = 30,
        change_model: UnobservedChangeModel | None = None,
    ) -> None:
        if max_relation_age_frames < 0:
            raise ValueError(
                "max_relation_age_frames must be non-negative"
            )

        self.max_relation_age_frames = max_relation_age_frames
        self.change_model = (
            change_model
            if change_model is not None
            else HeuristicUnobservedChangeModel()
        )
        self._histories: dict[str, _ObjectHistory] = {}
        self._track_to_object: dict[str, str] = {}
        self._latest_relations: dict[
            tuple[str, str],
            ObjectRelation,
        ] = {}
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

        for relation in infer_spatial_relations(
            assignments=identity_assignments,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
        ):
            self._latest_relations[
                self._relation_key(
                    relation.first_object_id,
                    relation.second_object_id,
                )
            ] = relation

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

            if event.event_type == EventType.OBJECT_DISAPPEARED:
                # A disappearance starts a new unseen episode. Evidence from
                # earlier unseen episodes must not leak into this one.
                history.opportunities.clear()
            elif event.event_type == EventType.OBJECT_REAPPEARED:
                # Reappearance closes the unseen episode. The historical
                # events remain, but episode-scoped opportunities are stale.
                history.opportunities.clear()

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

                signal_type, base_strength = signal_spec
                relation = self._relevant_relation(
                    target_object_id,
                    source_object_id,
                    target_history,
                )

                strength = base_strength
                relation_type = None
                relation_confidence = None

                if relation is not None:
                    relation_type = relation.relation_type.value
                    relation_confidence = relation.confidence
                    strength = min(
                        0.45,
                        base_strength * (1.0 + relation.confidence),
                    )

                if relation is None:
                    rationale = (
                        f"{event.event_type.value} occurred for "
                        f"{source_object_id} while "
                        f"{target_object_id} was unobserved."
                    )
                else:
                    rationale = (
                        f"{event.event_type.value} occurred for "
                        f"{source_object_id} while "
                        f"{target_object_id} was unobserved; the two "
                        "objects had a recent spatial association, so "
                        "this opportunity is weighted more strongly."
                    )

                target_history.opportunities.append(
                    OpportunitySignal(
                        target_object_id=target_object_id,
                        signal_type=signal_type,
                        strength=strength,
                        frame_index=frame_index,
                        timestamp_ms=timestamp_ms,
                        source_object_id=source_object_id,
                        source_track_id=event.subject_track_id,
                        rationale=rationale,
                        relation_type=relation_type,
                        relation_confidence=relation_confidence,
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

    def relations_for(
        self,
        object_id: str,
    ) -> tuple[ObjectRelation, ...]:
        relations = [
            relation
            for relation in self._latest_relations.values()
            if object_id in {
                relation.first_object_id,
                relation.second_object_id,
            }
        ]

        return tuple(
            sorted(
                relations,
                key=lambda relation: (
                    relation.frame_index,
                    relation.first_object_id,
                    relation.second_object_id,
                ),
            )
        )

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
        state_distribution = self.change_model.distribution(
            object_id=object_id,
            signals=signals,
            frame_index=self._current_frame_index,
            timestamp_ms=self._current_timestamp_ms,
        )
        location_distribution = self._location_distribution_for(
            object_id,
            history,
            signals,
        )

        return (
            state_distribution,
            location_distribution,
        )

    def _location_distribution_for(
        self,
        object_id: str,
        history: _ObjectHistory,
        signals: tuple[OpportunitySignal, ...],
    ) -> HypothesisDistribution:
        candidate_scores: dict[str, float] = {
            "last_seen_area": 1.0,
            "outside_observed_scene": 0.35,
        }
        candidate_rationales: dict[str, list[str]] = {
            "last_seen_area": [
                "The object was directly observed at its last-seen area."
            ],
            "outside_observed_scene": [
                "The object is currently unobserved, so departure from the "
                "observed scene remains possible."
            ],
        }

        for relation in self.relations_for(object_id):
            other_object_id = (
                relation.second_object_id
                if relation.first_object_id == object_id
                else relation.first_object_id
            )
            value = f"associated_with:{other_object_id}"
            score = 0.45 + (0.55 * relation.confidence)

            candidate_scores[value] = max(
                candidate_scores.get(value, 0.0),
                score,
            )
            candidate_rationales.setdefault(value, []).append(
                "The candidate object had a recent spatial association "
                "with the hidden object."
            )

        for signal in signals:
            if signal.source_object_id is None:
                continue

            value = f"associated_with:{signal.source_object_id}"

            if value not in candidate_scores:
                continue

            if signal.relation_type is None:
                continue

            candidate_scores[value] += signal.strength
            candidate_rationales[value].append(
                f"{signal.signal_type.value} affected this related "
                "candidate while the object was unseen."
            )

        elapsed_strength = sum(
            signal.strength
            for signal in signals
            if signal.signal_type == OpportunityType.ELAPSED_UNSEEN
        )
        candidate_scores["outside_observed_scene"] += elapsed_strength
        if elapsed_strength > 0.0:
            candidate_rationales["outside_observed_scene"].append(
                "Longer unseen duration increases the baseline possibility "
                "that the object is outside the observed scene."
            )

        total_score = sum(candidate_scores.values())

        hypotheses = tuple(
            CompetingHypothesis(
                value=value,
                weight=score / total_score,
                provenance="heuristic_location_baseline",
                rationale=" ".join(candidate_rationales[value]),
            )
            for value, score in sorted(
                candidate_scores.items(),
                key=lambda item: (-item[1], item[0]),
            )
        )

        return HypothesisDistribution(
            subject_object_id=object_id,
            predicate="CURRENT_LOCATION_CANDIDATE",
            hypotheses=hypotheses,
            updated_frame_index=self._current_frame_index,
            updated_timestamp_ms=self._current_timestamp_ms,
        )

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

    def _relevant_relation(
        self,
        target_object_id: str,
        source_object_id: str,
        target_history: _ObjectHistory,
    ) -> ObjectRelation | None:
        relation = self._latest_relations.get(
            self._relation_key(
                target_object_id,
                source_object_id,
            )
        )

        if relation is None:
            return None

        if not target_history.events:
            return None

        disappearance = target_history.events[-1]

        if relation.frame_index > disappearance.frame_index:
            return None

        age = disappearance.frame_index - relation.frame_index

        if age > self.max_relation_age_frames:
            return None

        return relation

    @staticmethod
    def _relation_key(
        first_object_id: str,
        second_object_id: str,
    ) -> tuple[str, str]:
        return tuple(
            sorted(
                (first_object_id, second_object_id)
            )
        )

    @staticmethod
    def _is_currently_unobserved(
        history: _ObjectHistory,
    ) -> bool:
        return bool(
            history.events
            and history.events[-1].event_type
            == EventType.OBJECT_DISAPPEARED
        )
