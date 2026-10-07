from blackbox.perception.tracker import (
    IoUTracker,
    intersection_over_union,
)
from blackbox.schemas.perception import BoundingBox, Detection


def detection(
    class_name: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    confidence: float = 0.9,
) -> Detection:
    return Detection(
        class_name=class_name,
        confidence=confidence,
        bbox=BoundingBox(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
        ),
    )


def test_intersection_over_union() -> None:
    first = BoundingBox(0, 0, 10, 10)
    second = BoundingBox(5, 0, 15, 10)

    assert intersection_over_union(first, second) == 50 / 150


def test_tracker_preserves_id_for_overlapping_same_class() -> None:
    tracker = IoUTracker(iou_threshold=0.3)

    first = tracker.update(
        [detection("backpack", 0, 0, 100, 100)]
    )
    second = tracker.update(
        [detection("backpack", 5, 0, 105, 100)]
    )

    assert first[0].track_id == "track-1"
    assert second[0].track_id == "track-1"


def test_tracker_creates_new_id_for_different_class() -> None:
    tracker = IoUTracker(iou_threshold=0.3)

    tracker.update(
        [detection("backpack", 0, 0, 100, 100)]
    )
    second = tracker.update(
        [detection("laptop", 0, 0, 100, 100)]
    )

    assert second[0].track_id == "track-2"


def test_tracker_survives_short_gap() -> None:
    tracker = IoUTracker(
        iou_threshold=0.3,
        max_missed_frames=2,
    )

    first = tracker.update(
        [detection("mug", 0, 0, 40, 40)]
    )

    tracker.update([])

    third = tracker.update(
        [detection("mug", 2, 0, 42, 40)]
    )

    assert first[0].track_id == third[0].track_id


def test_tracker_expires_after_missed_frame_limit() -> None:
    tracker = IoUTracker(
        iou_threshold=0.3,
        max_missed_frames=1,
    )

    tracker.update(
        [detection("mug", 0, 0, 40, 40)]
    )

    tracker.update([])
    tracker.update([])

    new_observation = tracker.update(
        [detection("mug", 0, 0, 40, 40)]
    )

    assert new_observation[0].track_id == "track-2"
