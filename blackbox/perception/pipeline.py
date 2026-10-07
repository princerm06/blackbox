import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from blackbox.perception.detector import Detector
from blackbox.perception.video import VideoReader
from blackbox.schemas.perception import Detection


@dataclass(frozen=True)
class ProcessingSummary:
    frames_processed: int
    detections_emitted: int
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


def annotate_frame(
    frame: np.ndarray,
    detections: list[Detection],
) -> np.ndarray:
    annotated = frame.copy()

    for detection in detections:
        bbox = detection.bbox
        x1, y1, x2, y2 = (
            int(bbox.x1),
            int(bbox.y1),
            int(bbox.x2),
            int(bbox.y2),
        )

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 255), 2)

        label = f"{detection.class_name} {detection.confidence:.2f}"
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
    def __init__(self, detector: Detector):
        self.detector = detector

    def process(
        self,
        input_path: Path,
        output_dir: Path,
        write_annotated_video: bool = True,
    ) -> ProcessingSummary:
        output_dir.mkdir(parents=True, exist_ok=True)

        observations_path = output_dir / "detections.jsonl"
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

        try:
            with observations_path.open("w", encoding="utf-8") as output_file:
                for video_frame in reader.frames():
                    detections = self.detector.detect(video_frame.image)

                    record = {
                        "frame_index": video_frame.frame_index,
                        "timestamp_ms": video_frame.timestamp_ms,
                        "detections": [
                            detection_to_dict(detection)
                            for detection in detections
                        ],
                    }

                    output_file.write(json.dumps(record) + "\n")

                    if writer is not None:
                        writer.write(
                            annotate_frame(
                                video_frame.image,
                                detections,
                            )
                        )

                    frames_processed += 1
                    detections_emitted += len(detections)
        finally:
            reader.close()

            if writer is not None:
                writer.release()

        return ProcessingSummary(
            frames_processed=frames_processed,
            detections_emitted=detections_emitted,
            observations_path=observations_path,
            annotated_video_path=annotated_video_path,
        )
