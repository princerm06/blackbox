from dataclasses import dataclass

import numpy as np

from blackbox.experiments.schema import UnseenEpisodeCase


def case_features(case: UnseenEpisodeCase) -> np.ndarray:
    strengths = case.opportunity_strengths
    return np.array(
        [
            float(len(strengths)),
            float(sum(strengths)),
            float(max(strengths, default=0.0)),
        ],
        dtype=float,
    )


@dataclass
class LogisticChangeModel:
    """Small learned baseline for P(change while unseen).

    The model intentionally uses a few interpretable episode-level features.
    It is a research baseline, not a production classifier.
    """

    learning_rate: float = 0.1
    steps: int = 1500
    l2: float = 1e-3

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.l2 < 0.0:
            raise ValueError("l2 must be non-negative")

        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.feature_mean: np.ndarray | None = None
        self.feature_scale: np.ndarray | None = None

    @property
    def name(self) -> str:
        return "logistic_change_model"

    def fit(
        self,
        cases: tuple[UnseenEpisodeCase, ...],
    ) -> "LogisticChangeModel":
        if not cases:
            raise ValueError("cases must not be empty")

        x = np.vstack([case_features(case) for case in cases])
        y = np.array(
            [float(case.changed_while_unobserved) for case in cases],
            dtype=float,
        )

        self.feature_mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale == 0.0] = 1.0
        self.feature_scale = scale
        x = (x - self.feature_mean) / self.feature_scale

        self.weights = np.zeros(x.shape[1], dtype=float)
        self.bias = 0.0

        for _ in range(self.steps):
            logits = x @ self.weights + self.bias
            probabilities = _sigmoid(logits)
            errors = probabilities - y

            weight_gradient = (
                x.T @ errors / len(cases)
                + self.l2 * self.weights
            )
            bias_gradient = float(errors.mean())

            self.weights -= self.learning_rate * weight_gradient
            self.bias -= self.learning_rate * bias_gradient

        return self

    def predict(self, case: UnseenEpisodeCase) -> float:
        if (
            self.weights is None
            or self.feature_mean is None
            or self.feature_scale is None
        ):
            raise ValueError("model must be fit before prediction")

        features = (
            case_features(case) - self.feature_mean
        ) / self.feature_scale
        logit = float(features @ self.weights + self.bias)
        return float(_sigmoid(np.array([logit]))[0])


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))
