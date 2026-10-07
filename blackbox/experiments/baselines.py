from typing import Protocol

from blackbox.experiments.schema import UnseenEpisodeCase
from blackbox.schemas.opportunities import OpportunitySignal, OpportunityType
from blackbox.world.change_model import HeuristicUnobservedChangeModel


class ChangeBaseline(Protocol):
    name: str

    def predict(self, case: UnseenEpisodeCase) -> float:
        ...


class LastObservationBaseline:
    """Assumes the last observed state persists while unseen."""

    name = "last_observation"

    def predict(self, case: UnseenEpisodeCase) -> float:
        return 0.0


class ConstantPriorBaseline:
    name = "constant_prior"

    def __init__(self, probability_changed: float = 0.5) -> None:
        if not 0.0 <= probability_changed <= 1.0:
            raise ValueError(
                "probability_changed must be between 0 and 1"
            )
        self.probability_changed = probability_changed

    def predict(self, case: UnseenEpisodeCase) -> float:
        return self.probability_changed


class DeterministicOpportunityBaseline:
    """Binary state-update baseline triggered by accumulated opportunity."""

    name = "deterministic_opportunity"

    def __init__(self, threshold: float = 0.15) -> None:
        if threshold < 0.0:
            raise ValueError("threshold must be non-negative")
        self.threshold = threshold

    def predict(self, case: UnseenEpisodeCase) -> float:
        return float(
            sum(case.opportunity_strengths) >= self.threshold
        )


class HeuristicModelBaseline:
    """Adapter around BlackBox's current swappable change model."""

    name = "heuristic_change_model"

    def __init__(
        self,
        model: HeuristicUnobservedChangeModel | None = None,
    ) -> None:
        self.model = model or HeuristicUnobservedChangeModel()

    def predict(self, case: UnseenEpisodeCase) -> float:
        signals = tuple(
            OpportunitySignal(
                target_object_id=case.case_id,
                signal_type=OpportunityType.OTHER_OBJECT_MOVED,
                strength=strength,
                frame_index=index,
                timestamp_ms=index,
                source_object_id=f"source-{index}",
                source_track_id=f"track-{index}",
                rationale="Controlled synthetic opportunity signal.",
            )
            for index, strength in enumerate(
                case.opportunity_strengths
            )
        )
        distribution = self.model.distribution(
            object_id=case.case_id,
            signals=signals,
            frame_index=len(signals),
            timestamp_ms=len(signals),
        )

        for hypothesis in distribution.hypotheses:
            if hypothesis.value == "changed_while_unobserved":
                return hypothesis.weight

        raise ValueError(
            "change model did not return changed_while_unobserved"
        )
