import pytest

from blackbox.schemas.uncertainty import (
    CompetingHypothesis,
    HypothesisDistribution,
)


def test_distribution_requires_normalized_weights() -> None:
    distribution = HypothesisDistribution(
        subject_object_id="object-1",
        predicate="UNOBSERVED_CHANGE_STATE",
        hypotheses=(
            CompetingHypothesis(
                value="unchanged_since_last_observation",
                weight=0.5,
                provenance="baseline_prior",
                rationale="No evidence favors change or no change yet.",
            ),
            CompetingHypothesis(
                value="changed_while_unobserved",
                weight=0.5,
                provenance="baseline_prior",
                rationale="No evidence favors change or no change yet.",
            ),
        ),
        updated_frame_index=10,
        updated_timestamp_ms=330,
    )

    assert sum(
        hypothesis.weight
        for hypothesis in distribution.hypotheses
    ) == 1.0


def test_distribution_rejects_invalid_weights() -> None:
    with pytest.raises(ValueError):
        HypothesisDistribution(
            subject_object_id="object-1",
            predicate="UNOBSERVED_CHANGE_STATE",
            hypotheses=(
                CompetingHypothesis(
                    value="a",
                    weight=0.8,
                    provenance="test",
                    rationale="test",
                ),
                CompetingHypothesis(
                    value="b",
                    weight=0.3,
                    provenance="test",
                    rationale="test",
                ),
            ),
            updated_frame_index=0,
            updated_timestamp_ms=0,
        )
