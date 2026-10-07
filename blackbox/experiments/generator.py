import math
import random
from dataclasses import dataclass

from blackbox.experiments.schema import UnseenEpisodeCase


@dataclass(frozen=True)
class GeneratorConfig:
    cases_per_level: int = 100
    seed: int = 7
    base_hazard: float = 0.05
    opportunity_sensitivity: float = 1.6

    def __post_init__(self) -> None:
        if self.cases_per_level <= 0:
            raise ValueError("cases_per_level must be positive")
        if self.base_hazard < 0.0:
            raise ValueError("base_hazard must be non-negative")
        if self.opportunity_sensitivity < 0.0:
            raise ValueError(
                "opportunity_sensitivity must be non-negative"
            )


_LEVEL_STRENGTHS: tuple[tuple[str, tuple[float, ...]], ...] = (
    ("none", ()),
    ("low", (0.03,)),
    ("medium", (0.08, 0.05)),
    ("high", (0.18, 0.08, 0.05)),
)


def change_probability(
    opportunity_strengths: tuple[float, ...],
    base_hazard: float,
    opportunity_sensitivity: float,
) -> float:
    """Synthetic data-generating process, not a claim about the real world."""
    hazard = (
        base_hazard
        + opportunity_sensitivity * sum(opportunity_strengths)
    )
    return 1.0 - math.exp(-hazard)


def generate_controlled_cases(
    config: GeneratorConfig = GeneratorConfig(),
) -> tuple[UnseenEpisodeCase, ...]:
    rng = random.Random(config.seed)
    cases: list[UnseenEpisodeCase] = []

    for level_name, strengths in _LEVEL_STRENGTHS:
        probability = change_probability(
            strengths,
            config.base_hazard,
            config.opportunity_sensitivity,
        )

        for index in range(config.cases_per_level):
            cases.append(
                UnseenEpisodeCase(
                    case_id=f"{level_name}-{index:04d}",
                    opportunity_strengths=strengths,
                    changed_while_unobserved=(
                        rng.random() < probability
                    ),
                    ground_truth_change_probability=probability,
                )
            )

    return tuple(cases)
