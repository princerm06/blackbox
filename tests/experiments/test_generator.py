import pytest

from blackbox.experiments.generator import (
    GeneratorConfig,
    change_probability,
    generate_controlled_cases,
)


def test_generator_is_reproducible_and_balanced_by_level() -> None:
    config = GeneratorConfig(cases_per_level=5, seed=42)

    first = generate_controlled_cases(config)
    second = generate_controlled_cases(config)

    assert first == second
    assert len(first) == 20
    assert [case.case_id for case in first[:5]] == [
        f"none-{index:04d}" for index in range(5)
    ]


def test_ground_truth_probability_increases_with_opportunities() -> None:
    cases = generate_controlled_cases(
        GeneratorConfig(cases_per_level=1)
    )
    probabilities = [
        case.ground_truth_change_probability
        for case in cases
    ]

    assert probabilities == sorted(probabilities)
    assert len(set(probabilities)) == 4


def test_change_probability_is_a_valid_hazard_probability() -> None:
    probability = change_probability((0.1, 0.2), 0.05, 1.5)

    assert 0.0 < probability < 1.0


def test_generator_rejects_invalid_case_count() -> None:
    with pytest.raises(ValueError):
        GeneratorConfig(cases_per_level=0)
