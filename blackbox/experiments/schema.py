from dataclasses import dataclass


@dataclass(frozen=True)
class UnseenEpisodeCase:
    """Ground-truth case for evaluating unseen-change belief models."""

    case_id: str
    opportunity_strengths: tuple[float, ...]
    changed_while_unobserved: bool
    ground_truth_change_probability: float | None = None

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("case_id must not be empty")
        if any(
            strength < 0.0
            for strength in self.opportunity_strengths
        ):
            raise ValueError(
                "opportunity strengths must be non-negative"
            )
        if (
            self.ground_truth_change_probability is not None
            and not 0.0
            <= self.ground_truth_change_probability
            <= 1.0
        ):
            raise ValueError(
                "ground_truth_change_probability must be between 0 and 1"
            )


@dataclass(frozen=True)
class BinaryPrediction:
    case_id: str
    probability_changed: float
    changed_while_unobserved: bool

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability_changed <= 1.0:
            raise ValueError(
                "probability_changed must be between 0 and 1"
            )


@dataclass(frozen=True)
class ModelEvaluation:
    model_name: str
    case_count: int
    brier: float
    nll: float
    ece: float
