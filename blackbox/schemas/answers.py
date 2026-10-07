from dataclasses import dataclass


@dataclass(frozen=True)
class EvidenceGroundedAnswer:
    subject_object_id: str
    answer: str
    confidence: float
    provenance: str
    primary_hypothesis: str
    alternatives: tuple[tuple[str, float], ...]
    evidence: tuple[str, ...]
    caveats: tuple[str, ...]
