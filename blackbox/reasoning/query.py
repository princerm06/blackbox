from blackbox.schemas.answers import EvidenceGroundedAnswer
from blackbox.world.projector import WorldStateProjector


from collections.abc import Callable


class BeliefQueryEngine:
    """Turns projected beliefs into evidence-grounded user answers.

    This layer summarizes existing state only. It must not invent containment,
    transfers, or observations that are absent from the projector.
    """

    def __init__(
        self,
        projector: WorldStateProjector,
        object_label: Callable[[str], str] | None = None,
    ) -> None:
        self.projector = projector
        self.object_label = object_label or (lambda object_id: object_id)

    def where_is(
        self,
        object_id: str,
    ) -> EvidenceGroundedAnswer:
        beliefs = self.projector.beliefs_for(object_id)
        distributions = self.projector.uncertainty_for(object_id)

        if not beliefs:
            return EvidenceGroundedAnswer(
                subject_object_id=object_id,
                answer=f"I do not have evidence for {self.object_label(object_id)}.",
                confidence=0.0,
                provenance="no_evidence",
                primary_hypothesis="unknown",
                alternatives=(),
                evidence=(),
                caveats=("The object has not been observed.",),
            )

        visibility = next(
            (
                belief
                for belief in beliefs
                if belief.predicate == "CURRENT_VISIBILITY"
            ),
            None,
        )

        if visibility is not None and visibility.value == "observed":
            return EvidenceGroundedAnswer(
                subject_object_id=object_id,
                answer=(
                    f"{self.object_label(object_id)} is currently observed at its latest "
                    "bounding-box location."
                ),
                confidence=visibility.confidence,
                provenance="observed",
                primary_hypothesis="currently_observed",
                alternatives=(),
                evidence=(
                    "Current visibility is directly supported by the "
                    "latest observation.",
                ),
                caveats=(),
            )

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
            return EvidenceGroundedAnswer(
                subject_object_id=object_id,
                answer=(
                    f"{self.object_label(object_id)} is not currently observed, and I do not "
                    "have enough evidence to rank location candidates."
                ),
                confidence=0.0,
                provenance="inferred",
                primary_hypothesis="unknown",
                alternatives=(),
                evidence=(),
                caveats=(
                    "Not observed does not mean the object left the scene.",
                ),
            )

        subject_class = self.projector.object_class(object_id)
        usable = []
        for hypothesis in location.hypotheses:
            value = str(hypothesis.value)
            if value.startswith("associated_with:"):
                related_object = value.split(":", 1)[1]
                if (
                    subject_class is not None
                    and self.projector.object_class(related_object)
                    == subject_class
                ):
                    continue
            usable.append(hypothesis)

        if not usable:
            usable = list(location.hypotheses)

        total_weight = sum(hypothesis.weight for hypothesis in usable)
        ranked = sorted(
            (
                (hypothesis, hypothesis.weight / total_weight)
                for hypothesis in usable
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        primary, primary_weight = ranked[0]
        alternatives = tuple(
            (str(hypothesis.value), weight)
            for hypothesis, weight in ranked[1:]
        )

        relevant_signals = []
        for signal in self.projector.opportunities_for(object_id):
            if (
                signal.source_object_id is not None
                and subject_class is not None
                and self.projector.object_class(signal.source_object_id)
                == subject_class
            ):
                continue
            relevant_signals.append(signal)

        relevant_signals.sort(
            key=lambda signal: signal.strength,
            reverse=True,
        )
        evidence = tuple(
            dict.fromkeys(
                signal.rationale
                for signal in relevant_signals
            )
        )[:6]

        answer = self._format_location_answer(
            object_id,
            str(primary.value),
            primary_weight,
        )

        return EvidenceGroundedAnswer(
            subject_object_id=object_id,
            answer=answer,
            confidence=primary_weight,
            provenance="inferred",
            primary_hypothesis=str(primary.value),
            alternatives=alternatives,
            evidence=evidence,
            caveats=(
                "This is a ranked hypothesis, not a direct observation.",
                "Spatial association does not prove containment or transfer.",
            ),
        )

    def _format_location_answer(
        self,
        object_id: str,
        value: str,
        weight: float,
    ) -> str:
        percent = round(weight * 100)

        if value == "last_seen_area":
            location = "its last-seen area"
        elif value == "outside_observed_scene":
            location = "outside the currently observed scene"
        elif value.startswith("associated_with:"):
            related_object = value.split(":", 1)[1]
            location = f"associated with {self.object_label(related_object)}"
        else:
            location = value

        return (
            f"The leading hypothesis is that {self.object_label(object_id)} is {location} "
            f"({percent}% heuristic weight)."
        )
