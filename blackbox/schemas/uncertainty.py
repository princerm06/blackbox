from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CompetingHypothesis:
    value: Any
    weight: float
    provenance: str
    rationale: str


@dataclass(frozen=True)
class HypothesisDistribution:
    subject_object_id: str
    predicate: str
    hypotheses: tuple[CompetingHypothesis, ...]
    updated_frame_index: int
    updated_timestamp_ms: int

    def __post_init__(self) -> None:
        if not self.hypotheses:
            raise ValueError("hypotheses must not be empty")

        total = sum(hypothesis.weight for hypothesis in self.hypotheses)

        if any(
            hypothesis.weight < 0.0
            for hypothesis in self.hypotheses
        ):
            raise ValueError("hypothesis weights must be non-negative")

        if abs(total - 1.0) > 1e-9:
            raise ValueError("hypothesis weights must sum to 1.0")
