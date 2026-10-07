import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from blackbox.events.extractor import EventExtractor
from blackbox.perception.detector import Detector
from blackbox.perception.identity import ObjectIdentityResolver
from blackbox.perception.tracker import IoUTracker, Tracker
from blackbox.perception.video import VideoReader
from blackbox.schemas.beliefs import ObjectStateHypothesis
from blackbox.schemas.events import WorldEvent
from blackbox.schemas.identity import IdentityAssignment
from blackbox.schemas.perception import Detection, TrackObservation
from blackbox.world.projector import WorldStateProjector


@dataclass(frozen=True)
class ProcessingSummary:
    frames_processed: int
    detections_emitted: int
    track_observations_emitted: int
    identity_assignments_emitted: int
    events_emitted: int
    observations_path: Path
    annotated_video_path: Path | None


def detection_to_dict(detection: Detection) -> dict[str, object]:
    return {
        "class_name": detection.class_name,
        "confidence": detection.confidence,
        "bbox": {
            "x1": detection.bbox.x1,
            "y1": detection.bbox.y1,
            "x2": detection.bbox.x2,
            "y2": detection.bbox.y2,
        },
    }


def track_observation_to_dict(
    observation: TrackObservation,
) -> dict[str, object]:
    return {
        "track_id": observation.track_id,
        "class_name": observation.class_name,
        "confidence": observation.confidence,
        "bbox": {
            "x1": observation.bbox.x1,
            "y1": observation.bbox.y1,
            "x2": observation.bbox.x2,
            "y2": observation.bbox.y2,
        },
    }


def identity_assignment_to_dict(
    assignment: IdentityAssignment,
) -> dict[str, object]:
    return {
        "object_id": assignment.object_id,
        "track_id": assignment.track_id,
        "class_name": assignment.class_name,
        "confidence": assignment.confidence,
        "provenance": assignment.provenance.value,
    }


def world_event_to_dict(event: WorldEvent) -> dict[str, object]:
    return {
        "event_type": event.event_type.value,
        "frame_index": event.frame_index,
        "timestamp_ms": event.timestamp_ms,
        "subject_track_id": event.subject_track_id,
        "class_name": event.class_name,
        "confidence": event.confidence,
        "provenance": event.provenance.value,
        "metadata": event.metadata,
    }


def belief_to_dict(
    belief: ObjectStateHypothesis,
) -> dict[str, object]:
    return {
        "subject_track_id": belief.subject_track_id,
        "predicate": belief.predicate,
        "value": belief.value,
        "confidence": belief.confidence,
        "provenance": belief.provenance.value,
        "updated_frame_index": belief.updated_frame_index,
        "updated_timestamp_ms": belief.updated_timestamp_ms,
        "supporting_event_types": list(
            belief.supporting_event_types
        ),
    }


def annotate_frame(
    frame: np.ndarray,
    observations: list[TrackObservation],
    assignments: list[IdentityAssignment] | None = None,
) -> np.ndarray:
    annotated = frame.copy()
    object_ids_by_track = {
        assignment.track_id: assignment.object_id
        for assignment in (assignments or [])
    }

    for observation in observations:
        bbox = observation.bbox
        x1, y1, x2, y2 = (
            int(bbox.x1),
            int(bbox.y1),
            int(bbox.x2),
            int(bbox.y2),
        )

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2,
        )

        object_id = object_ids_by_track.get(observation.track_id)
        identity_label = (
            f"{object_id} "
            if object_id is not None
            else ""
        )
        label = (
            f"{identity_label}{observation.track_id} "
            f"{observation.class_name} "
            f"{observation.confidence:.2f}"
        )

        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return annotated


class VideoDetectionPipeline:
    def __init__(
        self,
        detector: Detector,
        tracker: Tracker | None = None,
        identity_resolver: ObjectIdentityResolver | None = None,
        event_extractor: EventExtractor | None = None,
        world_state: WorldStateProjector | None = None,
    ):
        self.detector = detector
        self.tracker = tracker or IoUTracker()
        self.identity_resolver = (
            identity_resolver or ObjectIdentityResolver()
        )
        self.event_extractor = event_extractor or EventExtractor()
        self.world_state = world_state or WorldStateProjector()

    def process(
        self,
        input_path: Path,
        output_dir: Path,
        write_annotated_video: bool = True,
    ) -> ProcessingSummary:
        output_dir.mkdir(parents=True, exist_ok=True)

        observations_path = output_dir / "observations.jsonl"
        annotated_video_path = (
            output_dir / "annotated.mp4"
            if write_annotated_video
            else None
        )

        reader = VideoReader(input_path)
        writer: cv2.VideoWriter | None = None

        if annotated_video_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                str(annotated_video_path),
                fourcc,
                reader.fps,
                (reader.width, reader.height),
            )

            if not writer.isOpened():
                reader.close()
                raise ValueError(
                    f"Could not open video writer: {annotated_video_path}"
                )

        frames_processed = 0
        detections_emitted = 0
        track_observations_emitted = 0
        identity_assignments_emitted = 0
        events_emitted = 0

        try:
            with observations_path.open(
                "w",
                encoding="utf-8",
            ) as output_file:
                for video_frame in reader.frames():
                    detections = self.detector.detect(
                        video_frame.image
                    )

                    track_observations = self.tracker.update(
                        detections
                    )

                    identity_assignments = self.identity_resolver.update(
                        frame_index=video_frame.frame_index,
                        observations=track_observations,
                    )

                    events = self.event_extractor.update(
                        frame_index=video_frame.frame_index,
                        timestamp_ms=video_frame.timestamp_ms,
                        observations=track_observations,
                    )

                    self.world_state.update(
                        frame_index=video_frame.frame_index,
                        timestamp_ms=video_frame.timestamp_ms,
                        observations=track_observations,
                        events=events,
                    )

                    beliefs = {
                        track_id: [
                            belief_to_dict(belief)
                            for belief in track_beliefs
                        ]
                        for track_id, track_beliefs
                        in self.world_state.all_beliefs().items()
                    }

                    record = {
                        "frame_index": video_frame.frame_index,
                        "timestamp_ms": video_frame.timestamp_ms,
                        "detections": [
                            detection_to_dict(detection)
                            for detection in detections
                        ],
                        "tracks": [
                            track_observation_to_dict(observation)
                            for observation in track_observations
                        ],
                        "identities": [
                            identity_assignment_to_dict(assignment)
                            for assignment in identity_assignments
                        ],
                        "events": [
                            world_event_to_dict(event)
                            for event in events
                        ],
                        "beliefs": beliefs,
                    }

                    output_file.write(
                        json.dumps(record) + "\n"
                    )

                    if writer is not None:
                        writer.write(
                            annotate_frame(
                                video_frame.image,
                                track_observations,
                                identity_assignments,
                            )
                        )

                    frames_processed += 1
                    detections_emitted += len(detections)
                    track_observations_emitted += len(
                        track_observations
                    )
                    identity_assignments_emitted += len(
                        identity_assignments
                    )
                    events_emitted += len(events)
        finally:
            reader.close()

            if writer is not None:
                writer.release()

        return ProcessingSummary(
            frames_processed=frames_processed,
            detections_emitted=detections_emitted,
            track_observations_emitted=track_observations_emitted,
            identity_assignments_emitted=identity_assignments_emitted,
            events_emitted=events_emitted,
            observations_path=observations_path,
            annotated_video_path=annotated_video_path,
        )
