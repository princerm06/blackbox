import pytest

from blackbox.schemas.opportunities import OpportunitySignal, OpportunityType
from blackbox.world.change_model import HeuristicUnobservedChangeModel


def signal(strength: float) -> OpportunitySignal:
    return OpportunitySignal(
        target_object_id="object-1",
        signal_type=OpportunityType.OTHER_OBJECT_MOVED,
        strength=strength,
        frame_index=2,
        timestamp_ms=66,
        source_object_id="object-2",
        source_track_id="track-2",
        rationale="test",
    )


def changed_weight(distribution) -> float:
    return next(
        hypothesis.weight
        for hypothesis in distribution.hypotheses
        if hypothesis.value == "changed_while_unobserved"
    )


def test_heuristic_model_preserves_prior_without_signals() -> None:
    model = HeuristicUnobservedChangeModel(prior_changed=0.2)

    distribution = model.distribution(
        "object-1",
        (),
        frame_index=1,
        timestamp_ms=33,
    )

    assert changed_weight(distribution) == pytest.approx(0.2)


def test_heuristic_model_accumulates_opportunity_shift() -> None:
    model = HeuristicUnobservedChangeModel(
        prior_changed=0.2,
        max_opportunity_shift=0.5,
    )

    distribution = model.distribution(
        "object-1",
        (signal(0.1), signal(0.15)),
        frame_index=2,
        timestamp_ms=66,
    )

    assert changed_weight(distribution) == pytest.approx(0.45)


def test_heuristic_model_caps_opportunity_shift() -> None:
    model = HeuristicUnobservedChangeModel(
        prior_changed=0.2,
        max_opportunity_shift=0.3,
    )

    distribution = model.distribution(
        "object-1",
        (signal(0.4), signal(0.4)),
        frame_index=2,
        timestamp_ms=66,
    )

    assert changed_weight(distribution) == pytest.approx(0.5)


def test_invalid_probability_configuration_is_rejected() -> None:
    with pytest.raises(ValueError):
        HeuristicUnobservedChangeModel(
            prior_changed=0.8,
            max_opportunity_shift=0.3,
        )
