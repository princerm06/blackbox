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


def expected_calibration_error(
    predictions: tuple[BinaryPrediction, ...],
    bins: int = 10,
) -> float:
    """Equal-width ECE for the binary changed-state prediction."""
    if not predictions:
        raise ValueError("predictions must not be empty")
    if bins <= 0:
        raise ValueError("bins must be positive")

    total = len(predictions)
    error = 0.0

    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        members = tuple(
            prediction
            for prediction in predictions
            if (
                lower <= prediction.probability_changed < upper
                or (
                    bin_index == bins - 1
                    and prediction.probability_changed == 1.0
                )
            )
        )
        if not members:
            continue

        mean_confidence = sum(
            prediction.probability_changed
            for prediction in members
        ) / len(members)
        empirical_frequency = sum(
            float(prediction.changed_while_unobserved)
            for prediction in members
        ) / len(members)

        error += (
            len(members)
            / total
            * abs(mean_confidence - empirical_frequency)
        )

    return error
