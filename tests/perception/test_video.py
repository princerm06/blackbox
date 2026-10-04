from pathlib import Path

from blackbox.perception.video import VideoReader


def test_video_reader_reads_frames() -> None:
    path = Path("data/samples/test.mp4")
    reader = VideoReader(path)

    try:
        frames = list(reader.frames())
    finally:
        reader.close()

    assert reader.width == 640
    assert reader.height == 360
    assert reader.fps == 30.0
    assert reader.frame_count == 90

    assert len(frames) == 90

    assert frames[0].frame_index == 0
    assert frames[-1].frame_index == 89

    timestamps = [frame.timestamp_ms for frame in frames]

    assert timestamps == sorted(timestamps)
    assert timestamps[0] == 0
    assert timestamps[-1] > 2900

    for frame in frames:
        assert frame.image is not None
        assert frame.image.size > 0