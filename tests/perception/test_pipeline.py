import json
from pathlib import Path

import cv2
import numpy as np

from blackbox.perception.pipeline import (
    VideoDetectionPipeline,
    annotate_frame,
)
from blackbox.perception.tracker import IoUTracker
from blackbox.schemas.perception import (
    BoundingBox,
    Detection,
    TrackObservation,
)


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


def create_test_video(
    path: Path,
    frame_count: int = 4,
) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (64, 48),
    )

    assert writer.isOpened()

    for _ in range(frame_count):
        frame = np.zeros(
            (48, 64, 3),
            dtype=np.uint8,
        )
        writer.write(frame)

    writer.release()


def test_pipeline_emits_tracked_observations(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.avi"
    output_dir = tmp_path / "run"

    create_test_video(input_path)

    pipeline = VideoDetectionPipeline(
        detector=FakeDetector(),
        tracker=IoUTracker(),
    )

    summary = pipeline.process(
        input_path=input_path,
        output_dir=output_dir,
        write_annotated_video=False,
    )

    assert summary.frames_processed == 4
    assert summary.detections_emitted == 4
    assert summary.track_observations_emitted == 4
    assert summary.events_emitted == 1
    assert summary.annotated_video_path is None
    assert summary.observations_path.exists()

    lines = summary.observations_path.read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(lines) == 4

    records = [
        json.loads(line)
        for line in lines
    ]

    first_record = records[0]

    assert first_record["frame_index"] == 0
    assert first_record["timestamp_ms"] == 0
    assert (
        first_record["detections"][0]["class_name"]
        == "backpack"
    )
    assert (
        first_record["tracks"][0]["track_id"]
        == "track-1"
    )
    assert (
        first_record["tracks"][0]["class_name"]
        == "backpack"
    )
    assert first_record["events"][0]["event_type"] == "OBJECT_APPEARED"
    assert first_record["events"][0]["provenance"] == "observed"

    assert {
        record["tracks"][0]["track_id"]
        for record in records
    } == {"track-1"}


def test_annotate_frame_does_not_mutate_input() -> None:
    frame = np.zeros(
        (48, 64, 3),
        dtype=np.uint8,
    )

    observations = [
        TrackObservation(
            track_id="track-1",
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

    annotated = annotate_frame(
        frame,
        observations,
    )

    assert np.array_equal(
        frame,
        np.zeros(
            (48, 64, 3),
            dtype=np.uint8,
        ),
    )
    assert not np.array_equal(
        annotated,
        frame,
    )
