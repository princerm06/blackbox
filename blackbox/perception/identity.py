from dataclasses import dataclass

from blackbox.perception.tracker import intersection_over_union
from blackbox.schemas.identity import (
    IdentityAssignment,
    IdentityProvenance,
)
from blackbox.schemas.perception import BoundingBox, TrackObservation


@dataclass
class _ObjectIdentityState:
    object_id: str
    class_name: str
    last_bbox: BoundingBox
    last_seen_frame: int
    active_track_id: str | None


class ObjectIdentityResolver:
    """Conservative bridge from short-lived tracks to persistent object IDs.

    New tracks receive new object IDs unless a recently inactive object of the
    same class has strong spatial overlap. Re-identification is explicitly
    marked inferred and carries a confidence score.
    """

    def __init__(
        self,
        reid_iou_threshold: float = 0.70,
        max_reid_gap_frames: int = 30,
    ) -> None:
        if not 0.0 <= reid_iou_threshold <= 1.0:
            raise ValueError("reid_iou_threshold must be between 0 and 1")
        if max_reid_gap_frames < 0:
            raise ValueError("max_reid_gap_frames must be non-negative")

        self.reid_iou_threshold = reid_iou_threshold
        self.max_reid_gap_frames = max_reid_gap_frames
        self._track_to_object: dict[str, str] = {}
        self._objects: dict[str, _ObjectIdentityState] = {}
        self._next_object_number = 1

    def update(
        self,
        frame_index: int,
        observations: list[TrackObservation],
    ) -> list[IdentityAssignment]:
        current_track_ids = {
            observation.track_id
            for observation in observations
        }

        for state in self._objects.values():
            if (
                state.active_track_id is not None
                and state.active_track_id not in current_track_ids
            ):
                state.active_track_id = None

        assignments: list[IdentityAssignment] = []

        for observation in observations:
            existing_object_id = self._track_to_object.get(
                observation.track_id
            )

            if existing_object_id is not None:
                state = self._objects[existing_object_id]
                state.last_bbox = observation.bbox
                state.last_seen_frame = frame_index
                state.active_track_id = observation.track_id

                assignments.append(
                    IdentityAssignment(
                        object_id=existing_object_id,
                        track_id=observation.track_id,
                        class_name=observation.class_name,
                        confidence=1.0,
                        provenance=IdentityProvenance.OBSERVED_TRACK,
                        observation=observation,
                    )
                )
                continue

            candidate = self._best_inactive_candidate(
                frame_index=frame_index,
                observation=observation,
            )

            if candidate is None:
                object_id = f"object-{self._next_object_number}"
                self._next_object_number += 1

                self._objects[object_id] = _ObjectIdentityState(
                    object_id=object_id,
                    class_name=observation.class_name,
                    last_bbox=observation.bbox,
                    last_seen_frame=frame_index,
                    active_track_id=observation.track_id,
                )
                self._track_to_object[observation.track_id] = object_id

                assignments.append(
                    IdentityAssignment(
                        object_id=object_id,
                        track_id=observation.track_id,
                        class_name=observation.class_name,
                        confidence=1.0,
                        provenance=IdentityProvenance.OBSERVED_TRACK,
                        observation=observation,
                    )
                )
                continue

            state, score = candidate
            state.last_bbox = observation.bbox
            state.last_seen_frame = frame_index
            state.active_track_id = observation.track_id
            self._track_to_object[observation.track_id] = state.object_id

            assignments.append(
                IdentityAssignment(
                    object_id=state.object_id,
                    track_id=observation.track_id,
                    class_name=observation.class_name,
                    confidence=score,
                    provenance=IdentityProvenance.INFERRED_REIDENTIFICATION,
                    observation=observation,
                )
            )

        return assignments

    def object_id_for_track(
        self,
        track_id: str,
    ) -> str | None:
        return self._track_to_object.get(track_id)

    def _best_inactive_candidate(
        self,
        frame_index: int,
        observation: TrackObservation,
    ) -> tuple[_ObjectIdentityState, float] | None:
        best: tuple[_ObjectIdentityState, float] | None = None

        for state in self._objects.values():
            if state.active_track_id is not None:
                continue
            if state.class_name != observation.class_name:
                continue

            gap = frame_index - state.last_seen_frame
            if gap < 0 or gap > self.max_reid_gap_frames:
                continue

            score = intersection_over_union(
                state.last_bbox,
                observation.bbox,
            )

            if score < self.reid_iou_threshold:
                continue

            if best is None or score > best[1]:
                best = (state, score)

        return best
