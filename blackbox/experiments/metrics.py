import math

from blackbox.experiments.schema import BinaryPrediction


_EPSILON = 1e-12


def brier_score(predictions: tuple[BinaryPrediction, ...]) -> float:
    if not predictions:
        raise ValueError("predictions must not be empty")

    return sum(
        (
            prediction.probability_changed
            - float(prediction.changed_while_unobserved)
        )
        ** 2
        for prediction in predictions
    ) / len(predictions)


def negative_log_likelihood(
    predictions: tuple[BinaryPrediction, ...],
) -> float:
    if not predictions:
        raise ValueError("predictions must not be empty")

    losses = []
    for prediction in predictions:
        probability = min(
            1.0 - _EPSILON,
            max(_EPSILON, prediction.probability_changed),
        )
        if prediction.changed_while_unobserved:
            losses.append(-math.log(probability))
        else:
            losses.append(-math.log(1.0 - probability))

    return sum(losses) / len(losses)
