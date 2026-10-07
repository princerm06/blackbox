from typing import Protocol

from blackbox.schemas.opportunities import OpportunitySignal
from blackbox.schemas.uncertainty import (
    CompetingHypothesis,
    HypothesisDistribution,
)


class UnobservedChangeModel(Protocol):
    """Strategy interface for belief over change while an object is unseen."""

    def distribution(
        self,
        object_id: str,
        signals: tuple[OpportunitySignal, ...],
        frame_index: int,
        timestamp_ms: int,
    ) -> HypothesisDistribution:
        ...


class HeuristicUnobservedChangeModel:
    """Transparent baseline; weights are hypotheses, not calibrated claims."""

    def __init__(
        self,
        prior_changed: float = 0.5,
        max_opportunity_shift: float = 0.45,
    ) -> None:
        if not 0.0 <= prior_changed <= 1.0:
            raise ValueError("prior_changed must be between 0 and 1")
        if max_opportunity_shift < 0.0:
            raise ValueError(
                "max_opportunity_shift must be non-negative"
            )
        if prior_changed + max_opportunity_shift > 1.0:
            raise ValueError(
                "prior_changed + max_opportunity_shift must not exceed 1"
            )

        self.prior_changed = prior_changed
        self.max_opportunity_shift = max_opportunity_shift

    def distribution(
        self,
        object_id: str,
        signals: tuple[OpportunitySignal, ...],
        frame_index: int,
        timestamp_ms: int,
    ) -> HypothesisDistribution:
        opportunity_shift = min(
            self.max_opportunity_shift,
            sum(signal.strength for signal in signals),
        )
        changed_weight = self.prior_changed + opportunity_shift
        unchanged_weight = 1.0 - changed_weight

        if signals:
            rationale = (
                f"{len(signals)} opportunity signal(s) shifted the "
                "heuristic probability of unobserved change from prior "
                f"{self.prior_changed:.3f}."
            )
        else:
            rationale = (
                "No opportunity-for-change signals have accumulated since "
                "the object disappeared; the heuristic prior is retained."
            )

        return HypothesisDistribution(
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
            updated_frame_index=frame_index,
            updated_timestamp_ms=timestamp_ms,
        )
