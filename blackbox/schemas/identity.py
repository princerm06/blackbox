from dataclasses import dataclass
from enum import Enum

from blackbox.schemas.perception import TrackObservation


class IdentityProvenance(str, Enum):
    OBSERVED_TRACK = "observed_track"
    INFERRED_REIDENTIFICATION = "inferred_reidentification"


@dataclass(frozen=True)
class IdentityAssignment:
    object_id: str
    track_id: str
    class_name: str
    confidence: float
    provenance: IdentityProvenance
    observation: TrackObservation
