from dataclasses import dataclass


@dataclass(frozen=True)
class ReobservationTarget:
    target: str
    score: float
    rationale: str


@dataclass(frozen=True)
class ReobservationPlan:
    subject_object_id: str
    should_reobserve: bool
    uncertainty: float
    targets: tuple[ReobservationTarget, ...]
    rationale: str
