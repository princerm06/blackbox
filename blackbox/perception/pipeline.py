import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from blackbox.events.extractor import EventExtractor
from blackbox.perception.detector import Detector
from blackbox.perception.tracker import IoUTracker, Tracker
from blackbox.perception.video import VideoReader
from blackbox.schemas.events import WorldEvent
from blackbox.schemas.perception import Detection, TrackObservation


@dataclass(frozen=True)
class ProcessingSummary:
    frames_processed: int
    detections_emitted: int
    track_observations_emitted: int
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


def annotate_frame(
    frame: np.ndarray,
    observations: list[TrackObservation],
) -> np.ndarray:
    annotated = frame.copy()

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

        label = (
            f"{observation.track_id} "
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
        event_extractor: EventExtractor | None = None,
    ):
        self.detector = detector
        self.tracker = tracker or IoUTracker()
        self.event_extractor = event_extractor or EventExtractor()

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

                    events = self.event_extractor.update(
                        frame_index=video_frame.frame_index,
                        timestamp_ms=video_frame.timestamp_ms,
                        observations=track_observations,
                    )

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
                        "events": [
                            world_event_to_dict(event)
                            for event in events
                        ],
                    }

                    output_file.write(
                        json.dumps(record) + "\n"
                    )

                    if writer is not None:
                        writer.write(
                            annotate_frame(
                                video_frame.image,
                                track_observations,
                            )
                        )

                    frames_processed += 1
                    detections_emitted += len(detections)
                    track_observations_emitted += len(
                        track_observations
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
            events_emitted=events_emitted,
            observations_path=observations_path,
            annotated_video_path=annotated_video_path,
        )
