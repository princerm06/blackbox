from blackbox.experiments.baselines import ChangeBaseline
from blackbox.experiments.metrics import (
    brier_score,
    expected_calibration_error,
    negative_log_likelihood,
)
from blackbox.experiments.schema import (
    BinaryPrediction,
    ModelEvaluation,
    UnseenEpisodeCase,
)


def predict_cases(
    model: ChangeBaseline,
    cases: tuple[UnseenEpisodeCase, ...],
) -> tuple[BinaryPrediction, ...]:
    return tuple(
        BinaryPrediction(
            case_id=case.case_id,
            probability_changed=model.predict(case),
            changed_while_unobserved=case.changed_while_unobserved,
        )
        for case in cases
    )


def evaluate_model(
    model: ChangeBaseline,
    cases: tuple[UnseenEpisodeCase, ...],
) -> ModelEvaluation:
    if not cases:
        raise ValueError("cases must not be empty")

    predictions = predict_cases(model, cases)
    return ModelEvaluation(
        model_name=model.name,
        case_count=len(cases),
        brier=brier_score(predictions),
        nll=negative_log_likelihood(predictions),
        ece=expected_calibration_error(predictions),
    )


def evaluate_models(
    models: tuple[ChangeBaseline, ...],
    cases: tuple[UnseenEpisodeCase, ...],
) -> tuple[ModelEvaluation, ...]:
    return tuple(
        evaluate_model(model, cases)
        for model in models
    )
