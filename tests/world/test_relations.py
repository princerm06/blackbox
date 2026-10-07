from blackbox.schemas.identity import (
    IdentityAssignment,
    IdentityProvenance,
)
from blackbox.schemas.perception import BoundingBox, TrackObservation
from blackbox.world.relations import infer_spatial_relations


def assignment(
    object_id: str,
    track_id: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> IdentityAssignment:
    observation = TrackObservation(
        track_id=track_id,
        class_name="object",
        confidence=0.9,
        bbox=BoundingBox(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
        ),
    )

    return IdentityAssignment(
        object_id=object_id,
        track_id=track_id,
        class_name="object",
        confidence=1.0,
        provenance=IdentityProvenance.OBSERVED_TRACK,
        observation=observation,
    )


def test_nearby_objects_create_spatial_relation() -> None:
    relations = infer_spatial_relations(
        assignments=[
            assignment(
                "object-1",
                "track-1",
                0,
                0,
                100,
                100,
            ),
            assignment(
                "object-2",
                "track-2",
                70,
                0,
                170,
                100,
            ),
        ],
        frame_index=4,
        timestamp_ms=132,
    )

    assert len(relations) == 1
    assert relations[0].first_object_id == "object-1"
    assert relations[0].second_object_id == "object-2"
    assert relations[0].confidence > 0.0


def test_far_objects_do_not_create_spatial_relation() -> None:
    relations = infer_spatial_relations(
        assignments=[
            assignment(
                "object-1",
                "track-1",
                0,
                0,
                100,
                100,
            ),
            assignment(
                "object-2",
                "track-2",
                500,
                0,
                600,
                100,
            ),
        ],
        frame_index=4,
        timestamp_ms=132,
    )

    assert relations == []
