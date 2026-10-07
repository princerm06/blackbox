from blackbox.events.extractor import EventExtractor
from blackbox.schemas.events import EventType
from blackbox.schemas.perception import BoundingBox, TrackObservation


def observation(
    track_id: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    confidence: float = 0.9,
) -> TrackObservation:
    return TrackObservation(
        track_id=track_id,
        class_name="backpack",
        confidence=confidence,
        bbox=BoundingBox(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
        ),
    )


def test_first_observation_emits_appeared() -> None:
    extractor = EventExtractor()

    events = extractor.update(
        frame_index=0,
        timestamp_ms=0,
        observations=[
            observation("track-1", 0, 0, 100, 100)
        ],
    )

    assert [event.event_type for event in events] == [
        EventType.OBJECT_APPEARED
    ]


def test_large_motion_emits_moved() -> None:
    extractor = EventExtractor(
        movement_threshold_px=10.0,
    )

    extractor.update(
        frame_index=0,
        timestamp_ms=0,
        observations=[
            observation("track-1", 0, 0, 100, 100)
        ],
    )

    events = extractor.update(
        frame_index=1,
        timestamp_ms=33,
        observations=[
            observation("track-1", 20, 0, 120, 100)
        ],
    )

    assert [event.event_type for event in events] == [
        EventType.OBJECT_MOVED
    ]


def test_missing_track_emits_disappeared() -> None:
    extractor = EventExtractor()

    extractor.update(
        frame_index=0,
        timestamp_ms=0,
        observations=[
            observation("track-1", 0, 0, 100, 100)
        ],
    )

    events = extractor.update(
        frame_index=1,
        timestamp_ms=33,
        observations=[],
    )

    assert [event.event_type for event in events] == [
        EventType.OBJECT_DISAPPEARED
    ]


def test_returning_track_emits_reappeared() -> None:
    extractor = EventExtractor()

    extractor.update(
        frame_index=0,
        timestamp_ms=0,
        observations=[
            observation("track-1", 0, 0, 100, 100)
        ],
    )

    extractor.update(
        frame_index=1,
        timestamp_ms=33,
        observations=[],
    )

    events = extractor.update(
        frame_index=2,
        timestamp_ms=66,
        observations=[
            observation("track-1", 2, 0, 102, 100)
        ],
    )

    assert [event.event_type for event in events] == [
        EventType.OBJECT_REAPPEARED
    ]


def test_disappearance_is_not_repeated_every_frame() -> None:
    extractor = EventExtractor()

    extractor.update(
        frame_index=0,
        timestamp_ms=0,
        observations=[
            observation("track-1", 0, 0, 100, 100)
        ],
    )

    extractor.update(
        frame_index=1,
        timestamp_ms=33,
        observations=[],
    )

    events = extractor.update(
        frame_index=2,
        timestamp_ms=66,
        observations=[],
    )

    assert events == []
