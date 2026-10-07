import pytest

from blackbox.experiments.generator import (
    GeneratorConfig,
    generate_controlled_cases,
)
from blackbox.experiments.learned import LogisticChangeModel, case_features
from blackbox.experiments.split import split_cases


def test_case_features_are_interpretable() -> None:
    case = generate_controlled_cases(
        GeneratorConfig(cases_per_level=1)
    )[-1]

    features = case_features(case)

    assert features.tolist() == pytest.approx([3.0, 0.31, 0.18])


def test_episode_split_is_reproducible_and_disjoint() -> None:
    cases = generate_controlled_cases(
        GeneratorConfig(cases_per_level=25, seed=9)
    )

    first = split_cases(cases, seed=3)
    second = split_cases(cases, seed=3)

    assert first == second

    train_ids = {case.case_id for case in first[0]}
    validation_ids = {case.case_id for case in first[1]}
    test_ids = {case.case_id for case in first[2]}

    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)
    assert len(train_ids | validation_ids | test_ids) == len(cases)


def test_logistic_model_learns_higher_risk_for_more_opportunities() -> None:
    cases = generate_controlled_cases(
        GeneratorConfig(cases_per_level=250, seed=7)
    )
    train, _, test = split_cases(cases, seed=17)

    model = LogisticChangeModel(steps=2500).fit(train)

    low = next(
        case for case in test if case.case_id.startswith("none-")
    )
    high = next(
        case for case in test if case.case_id.startswith("high-")
    )

    assert model.predict(high) > model.predict(low)


def test_logistic_model_requires_fit() -> None:
    case = generate_controlled_cases(
        GeneratorConfig(cases_per_level=1)
    )[0]

    with pytest.raises(ValueError):
        LogisticChangeModel().predict(case)
