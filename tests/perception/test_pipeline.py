import json
from pathlib import Path

import cv2
import numpy as np

from blackbox.perception.pipeline import (
    VideoDetectionPipeline,
    annotate_frame,
)
from blackbox.schemas.perception import BoundingBox, Detection


class FakeDetector:
    def detect(self, frame: np.ndarray) -> list[Detection]:
        return [
            Detection(
                class_name="backpack",
                confidence=0.9,
                bbox=BoundingBox(
                    x1=5,
                    y1=5,
                    x2=30,
                    y2=30,
                ),
            )
        ]


def create_test_video(path: Path, frame_count: int = 4) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (64, 48),
    )

    assert writer.isOpened()

    for _ in range(frame_count):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        writer.write(frame)

    writer.release()


def test_pipeline_emits_one_jsonl_record_per_frame(tmp_path: Path) -> None:
    input_path = tmp_path / "input.avi"
    output_dir = tmp_path / "run"

    create_test_video(input_path)

    pipeline = VideoDetectionPipeline(
        detector=FakeDetector(),
    )

    summary = pipeline.process(
        input_path=input_path,
        output_dir=output_dir,
        write_annotated_video=False,
    )

    assert summary.frames_processed == 4
    assert summary.detections_emitted == 4
    assert summary.annotated_video_path is None
    assert summary.observations_path.exists()

    lines = summary.observations_path.read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(lines) == 4

    first_record = json.loads(lines[0])

    assert first_record["frame_index"] == 0
    assert first_record["timestamp_ms"] == 0
    assert first_record["detections"][0]["class_name"] == "backpack"
    assert first_record["detections"][0]["confidence"] == 0.9
    assert first_record["detections"][0]["bbox"]["x1"] == 5


def test_annotate_frame_does_not_mutate_input() -> None:
    frame = np.zeros((48, 64, 3), dtype=np.uint8)

    detections = [
        Detection(
            class_name="laptop",
            confidence=0.8,
            bbox=BoundingBox(
                x1=5,
                y1=5,
                x2=30,
                y2=30,
            ),
        )
    ]

    annotated = annotate_frame(frame, detections)

    assert np.array_equal(
        frame,
        np.zeros((48, 64, 3), dtype=np.uint8),
    )
    assert not np.array_equal(annotated, frame)
