from dataclasses import dataclass

from blackbox.experiments.learned import LogisticChangeModel
from blackbox.experiments.metrics import negative_log_likelihood
from blackbox.experiments.runner import predict_cases
from blackbox.experiments.schema import UnseenEpisodeCase


@dataclass(frozen=True)
class LogisticSelection:
    model: LogisticChangeModel
    validation_nll: float
    learning_rate: float
    l2: float


def select_logistic_model(
    train_cases: tuple[UnseenEpisodeCase, ...],
    validation_cases: tuple[UnseenEpisodeCase, ...],
    learning_rates: tuple[float, ...] = (0.03, 0.1, 0.3),
    l2_values: tuple[float, ...] = (0.0, 1e-3, 1e-2),
    steps: int = 2000,
) -> LogisticSelection:
    if not train_cases:
        raise ValueError("train_cases must not be empty")
    if not validation_cases:
        raise ValueError("validation_cases must not be empty")

    best: LogisticSelection | None = None

    for learning_rate in learning_rates:
        for l2 in l2_values:
            model = LogisticChangeModel(
                learning_rate=learning_rate,
                steps=steps,
                l2=l2,
            ).fit(train_cases)
            validation_nll = negative_log_likelihood(
                predict_cases(model, validation_cases)
            )
            candidate = LogisticSelection(
                model=model,
                validation_nll=validation_nll,
                learning_rate=learning_rate,
                l2=l2,
            )

            if (
                best is None
                or candidate.validation_nll < best.validation_nll
            ):
                best = candidate

    if best is None:
        raise ValueError("no hyperparameter candidates were supplied")

    return best
