import pytest

from blackbox.experiments.baselines import (
    ConstantPriorBaseline,
    DeterministicOpportunityBaseline,
    HeuristicModelBaseline,
    LastObservationBaseline,
)
from blackbox.experiments.schema import UnseenEpisodeCase


def case(strengths: tuple[float, ...]) -> UnseenEpisodeCase:
    return UnseenEpisodeCase("case", strengths, False)


def test_last_observation_predicts_no_change() -> None:
    assert LastObservationBaseline().predict(case((0.3,))) == 0.0


def test_constant_prior_is_constant() -> None:
    assert ConstantPriorBaseline(0.2).predict(case(())) == 0.2
    assert ConstantPriorBaseline(0.2).predict(case((0.4,))) == 0.2


def test_deterministic_baseline_uses_threshold() -> None:
    model = DeterministicOpportunityBaseline(threshold=0.15)

    assert model.predict(case((0.05, 0.05))) == 0.0
    assert model.predict(case((0.08, 0.08))) == 1.0


def test_heuristic_adapter_matches_current_change_model() -> None:
    model = HeuristicModelBaseline()

    assert model.predict(case(())) == pytest.approx(0.5)
    assert model.predict(case((0.08, 0.05))) == pytest.approx(0.63)
