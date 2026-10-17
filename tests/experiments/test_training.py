from blackbox.experiments.generator import (
    GeneratorConfig,
    generate_controlled_cases,
)
from blackbox.experiments.split import split_cases
from blackbox.experiments.training import select_logistic_model


def test_validation_selects_a_fitted_model() -> None:
    cases = generate_controlled_cases(
        GeneratorConfig(cases_per_level=80, seed=7)
    )
    train, validation, test = split_cases(cases, seed=17)

    selection = select_logistic_model(
        train,
        validation,
        learning_rates=(0.03, 0.1),
        l2_values=(0.0, 1e-3),
        steps=800,
    )

    prediction = selection.model.predict(test[0])

    assert 0.0 <= prediction <= 1.0
    assert selection.validation_nll >= 0.0
    assert selection.learning_rate in {0.03, 0.1}
    assert selection.l2 in {0.0, 1e-3}
