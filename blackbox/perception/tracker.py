from dataclasses import dataclass
from typing import Protocol

from blackbox.schemas.perception import (
    BoundingBox,
    Detection,
    TrackObservation,
)


class Tracker(Protocol):
    def update(
        self,
        detections: list[Detection],
    ) -> list[TrackObservation]:
        ...


def intersection_over_union(
    first: BoundingBox,
    second: BoundingBox,
) -> float:
    x1 = max(first.x1, second.x1)
    y1 = max(first.y1, second.y1)
    x2 = min(first.x2, second.x2)
    y2 = min(first.y2, second.y2)

    intersection_width = max(0.0, x2 - x1)
    intersection_height = max(0.0, y2 - y1)
    intersection = intersection_width * intersection_height

    union = first.area + second.area - intersection

    if union <= 0.0:
        return 0.0

    return intersection / union


@dataclass
class _TrackState:
    track_id: str
    class_name: str
    bbox: BoundingBox
    missed_frames: int = 0


class IoUTracker:
    """Small deterministic baseline tracker.

    This preserves short-term track continuity only. A track ID is not a
    persistent physical object identity and should never be treated as one.
    """

    def __init__(
        self,
        iou_threshold: float = 0.30,
        max_missed_frames: int = 3,
    ) -> None:
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1")

        if max_missed_frames < 0:
            raise ValueError("max_missed_frames must be non-negative")

        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self._tracks: dict[str, _TrackState] = {}
        self._next_track_number = 1

    def update(
        self,
        detections: list[Detection],
    ) -> list[TrackObservation]:
        unmatched_track_ids = set(self._tracks)
        unmatched_detection_indices = set(range(len(detections)))
        matches: list[tuple[str, int]] = []

        candidates: list[tuple[float, str, int]] = []

        for track_id, track in self._tracks.items():
            for detection_index, detection in enumerate(detections):
                if track.class_name != detection.class_name:
                    continue

                iou = intersection_over_union(
                    track.bbox,
                    detection.bbox,
                )

                if iou >= self.iou_threshold:
                    candidates.append(
                        (
                            iou,
                            track_id,
                            detection_index,
                        )
                    )

        candidates.sort(reverse=True)

        for _, track_id, detection_index in candidates:
            if track_id not in unmatched_track_ids:
                continue

            if detection_index not in unmatched_detection_indices:
                continue

            unmatched_track_ids.remove(track_id)
            unmatched_detection_indices.remove(detection_index)
            matches.append((track_id, detection_index))

        observations: list[TrackObservation] = []

        for track_id, detection_index in matches:
            detection = detections[detection_index]
            track = self._tracks[track_id]
            track.bbox = detection.bbox
            track.class_name = detection.class_name
            track.missed_frames = 0

            observations.append(
                TrackObservation(
                    track_id=track_id,
                    class_name=detection.class_name,
                    confidence=detection.confidence,
                    bbox=detection.bbox,
                )
            )

        for track_id in unmatched_track_ids:
            self._tracks[track_id].missed_frames += 1

        expired_track_ids = [
            track_id
            for track_id, track in self._tracks.items()
            if track.missed_frames > self.max_missed_frames
        ]

        for track_id in expired_track_ids:
            del self._tracks[track_id]

        for detection_index in sorted(unmatched_detection_indices):
            detection = detections[detection_index]
            track_id = f"track-{self._next_track_number}"
            self._next_track_number += 1

            self._tracks[track_id] = _TrackState(
                track_id=track_id,
                class_name=detection.class_name,
                bbox=detection.bbox,
            )

            observations.append(
                TrackObservation(
                    track_id=track_id,
                    class_name=detection.class_name,
                    confidence=detection.confidence,
                    bbox=detection.bbox,
                )
            )

        observations.sort(key=lambda observation: observation.track_id)
        return observations
