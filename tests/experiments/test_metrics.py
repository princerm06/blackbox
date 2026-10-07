import math

import pytest

from blackbox.experiments.metrics import (
    brier_score,
    negative_log_likelihood,
)
from blackbox.experiments.schema import BinaryPrediction


def predictions() -> tuple[BinaryPrediction, ...]:
    return (
        BinaryPrediction("a", 0.8, True),
        BinaryPrediction("b", 0.3, False),
    )


def test_brier_score() -> None:
    assert brier_score(predictions()) == pytest.approx(0.065)


def test_negative_log_likelihood() -> None:
    expected = (-math.log(0.8) - math.log(0.7)) / 2
    assert negative_log_likelihood(predictions()) == pytest.approx(
        expected
    )


def test_metrics_reject_empty_input() -> None:
    with pytest.raises(ValueError):
        brier_score(())
    with pytest.raises(ValueError):
        negative_log_likelihood(())
