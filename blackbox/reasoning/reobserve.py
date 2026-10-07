import math

from blackbox.schemas.reobservation import (
    ReobservationPlan,
    ReobservationTarget,
)
from blackbox.world.projector import WorldStateProjector


def normalized_entropy(weights: list[float]) -> float:
    positive = [weight for weight in weights if weight > 0.0]

    if len(positive) <= 1:
        return 0.0

    entropy = -sum(
        weight * math.log(weight)
        for weight in positive
    )
    return entropy / math.log(len(positive))


class ReobservationPlanner:
    """Ranks evidence-gathering targets from current location uncertainty.

    This is a transparent baseline policy, not learned active perception.
    Candidate probability and ambiguity drive the score; no hypothesis is
    silently converted into a fact.
    """

    def __init__(
        self,
        projector: WorldStateProjector,
        uncertainty_threshold: float = 0.55,
    ) -> None:
        if not 0.0 <= uncertainty_threshold <= 1.0:
            raise ValueError(
                "uncertainty_threshold must be between 0 and 1"
            )

        self.projector = projector
        self.uncertainty_threshold = uncertainty_threshold

    def plan_for(self, object_id: str) -> ReobservationPlan:
        distributions = self.projector.uncertainty_for(object_id)
        location = next(
            (
                distribution
                for distribution in distributions
                if distribution.predicate
                == "CURRENT_LOCATION_CANDIDATE"
            ),
            None,
        )

        if location is None:
            return ReobservationPlan(
                subject_object_id=object_id,
                should_reobserve=False,
                uncertainty=0.0,
                targets=(),
                rationale=(
                    "No unresolved location distribution requires "
                    "active re-observation."
                ),
            )

        uncertainty = normalized_entropy(
            [
                hypothesis.weight
                for hypothesis in location.hypotheses
            ]
        )
        should_reobserve = (
            uncertainty >= self.uncertainty_threshold
        )

        targets = tuple(
            sorted(
                (
                    ReobservationTarget(
                        target=self._inspection_target(
                            str(hypothesis.value)
                        ),
                        score=hypothesis.weight * uncertainty,
                        rationale=self._target_rationale(
                            str(hypothesis.value),
                            hypothesis.weight,
                            uncertainty,
                        ),
                    )
                    for hypothesis in location.hypotheses
                ),
                key=lambda target: target.score,
                reverse=True,
            )
        )

        if should_reobserve:
            rationale = (
                "Location hypotheses remain ambiguous enough to justify "
                "another observation; inspect the highest-ranked target "
                "first."
            )
        else:
            rationale = (
                "The current location distribution is concentrated enough "
                "that this baseline policy does not request re-observation."
            )

        return ReobservationPlan(
            subject_object_id=object_id,
            should_reobserve=should_reobserve,
            uncertainty=uncertainty,
            targets=targets if should_reobserve else (),
            rationale=rationale,
        )

    @staticmethod
    def _inspection_target(value: str) -> str:
        if value == "last_seen_area":
            return "inspect:last_seen_area"
        if value == "outside_observed_scene":
            return "inspect:scene_exit_or_boundary"
        if value.startswith("associated_with:"):
            object_id = value.split(":", 1)[1]
            return f"inspect:object:{object_id}"
        return f"inspect:hypothesis:{value}"

    @staticmethod
    def _target_rationale(
        value: str,
        weight: float,
        uncertainty: float,
    ) -> str:
        return (
            f"Candidate {value} has heuristic weight {weight:.3f}; "
            f"current normalized location entropy is {uncertainty:.3f}."
        )
