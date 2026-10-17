from blackbox.experiments.baselines import (
    ConstantPriorBaseline,
    HeuristicModelBaseline,
    LastObservationBaseline,
)
from blackbox.experiments.generator import (
    GeneratorConfig,
    generate_controlled_cases,
)
from blackbox.experiments.runner import evaluate_models


def test_runner_compares_models_on_same_cases() -> None:
    cases = generate_controlled_cases(
        GeneratorConfig(cases_per_level=20, seed=11)
    )
    results = evaluate_models(
        (
            LastObservationBaseline(),
            ConstantPriorBaseline(),
            HeuristicModelBaseline(),
        ),
        cases,
    )

    assert [result.model_name for result in results] == [
        "last_observation",
        "constant_prior",
        "heuristic_change_model",
    ]
    assert all(result.case_count == 80 for result in results)
    assert all(result.brier >= 0.0 for result in results)
    assert all(result.nll >= 0.0 for result in results)
    assert all(0.0 <= result.ece <= 1.0 for result in results)
