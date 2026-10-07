from pathlib import Path

import cv2
import numpy as np

from blackbox.perception.video import VideoReader


def create_test_video(path: Path, frame_count: int = 6) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        30.0,
        (64, 48),
    )

    assert writer.isOpened()

    for frame_index in range(frame_count):
        frame = np.full(
            (48, 64, 3),
            frame_index,
            dtype=np.uint8,
        )
        writer.write(frame)

    writer.release()


def test_video_reader_reads_frames(tmp_path: Path) -> None:
    path = tmp_path / "test.avi"
    create_test_video(path)

    reader = VideoReader(path)

    try:
        frames = list(reader.frames())
    finally:
        reader.close()

    assert reader.width == 64
    assert reader.height == 48
    assert reader.fps == 30.0
    assert reader.frame_count == 6

    assert len(frames) == 6
    assert frames[0].frame_index == 0
    assert frames[-1].frame_index == 5

    timestamps = [frame.timestamp_ms for frame in frames]

    assert timestamps == sorted(timestamps)
    assert timestamps[0] == 0

    for frame in frames:
        assert frame.image is not None
        assert frame.image.size > 0
