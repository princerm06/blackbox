from blackbox.schemas.perception import (
    BoundingBox,
    Detection,
    Observation,
    TrackObservation,
)


def test_bounding_box_dimensions() -> None:
    bbox = BoundingBox(
        x1=10,
        y1=20,
        x2=110,
        y2=220,
    )

    assert bbox.width == 100
    assert bbox.height == 200
    assert bbox.area == 20000


def test_detection_schema() -> None:
    detection = Detection(
        class_name="backpack",
        confidence=0.91,
        bbox=BoundingBox(
            x1=100,
            y1=150,
            x2=300,
            y2=500,
        ),
    )

    assert detection.class_name == "backpack"
    assert detection.confidence == 0.91


def test_track_observation_schema() -> None:
    track = TrackObservation(
        track_id="track-1",
        class_name="backpack",
        confidence=0.88,
        bbox=BoundingBox(
            x1=100,
            y1=150,
            x2=300,
            y2=500,
        ),
    )

    assert track.track_id == "track-1"


def test_observation_schema() -> None:
    observation = Observation(
        frame_index=42,
        timestamp_ms=1400,
        track_id="track-1",
        object_id=None,
        class_name="backpack",
        confidence=0.88,
        bbox=BoundingBox(
            x1=100,
            y1=150,
            x2=300,
            y2=500,
        ),
    )

    assert observation.visible is True
    assert observation.object_id is None