from blackbox.perception.identity import ObjectIdentityResolver
from blackbox.schemas.identity import IdentityProvenance
from blackbox.schemas.perception import BoundingBox, TrackObservation


def observation(
    track_id: str,
    class_name: str = "backpack",
    x1: float = 0,
    y1: float = 0,
    x2: float = 100,
    y2: float = 100,
) -> TrackObservation:
    return TrackObservation(
        track_id=track_id,
        class_name=class_name,
        confidence=0.9,
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
    )


def test_same_track_keeps_same_object_id() -> None:
    resolver = ObjectIdentityResolver()

    first = resolver.update(0, [observation("track-1")])
    second = resolver.update(1, [observation("track-1", x1=2, x2=102)])

    assert first[0].object_id == "object-1"
    assert second[0].object_id == "object-1"
    assert second[0].provenance == IdentityProvenance.OBSERVED_TRACK


def test_new_track_can_infer_same_object_after_gap() -> None:
    resolver = ObjectIdentityResolver(
        reid_iou_threshold=0.7,
        max_reid_gap_frames=10,
    )

    first = resolver.update(0, [observation("track-1")])
    resolver.update(1, [])
    second = resolver.update(
        2,
        [observation("track-2", x1=5, x2=105)],
    )

    assert first[0].object_id == "object-1"
    assert second[0].object_id == "object-1"
    assert (
        second[0].provenance
        == IdentityProvenance.INFERRED_REIDENTIFICATION
    )
    assert second[0].confidence < 1.0


def test_far_reappearance_gets_new_object_id() -> None:
    resolver = ObjectIdentityResolver(
        reid_iou_threshold=0.7,
        max_reid_gap_frames=10,
    )

    resolver.update(0, [observation("track-1")])
    resolver.update(1, [])
    second = resolver.update(
        2,
        [observation("track-2", x1=300, x2=400)],
    )

    assert second[0].object_id == "object-2"


def test_different_class_never_reuses_object_id() -> None:
    resolver = ObjectIdentityResolver()

    resolver.update(0, [observation("track-1", class_name="backpack")])
    resolver.update(1, [])
    second = resolver.update(
        2,
        [observation("track-2", class_name="laptop")],
    )

    assert second[0].object_id == "object-2"
