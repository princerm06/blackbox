from blackbox.schemas.events import (
    EventProvenance,
    EventType,
    WorldEvent,
)
from blackbox.schemas.identity import (
    IdentityAssignment,
    IdentityProvenance,
)
from blackbox.schemas.perception import BoundingBox, TrackObservation
from blackbox.world.projector import WorldStateProjector


def observation(
    track_id: str = "track-1",
) -> TrackObservation:
    return TrackObservation(
        track_id=track_id,
        class_name="backpack",
        confidence=0.9,
        bbox=BoundingBox(
            x1=10,
            y1=20,
            x2=110,
            y2=220,
        ),
    )


def assignment(
    track_id: str = "track-1",
    object_id: str = "object-1",
) -> IdentityAssignment:
    obs = observation(track_id)
    return IdentityAssignment(
        object_id=object_id,
        track_id=track_id,
        class_name=obs.class_name,
        confidence=1.0,
        provenance=IdentityProvenance.OBSERVED_TRACK,
        observation=obs,
    )


def event(
    event_type: EventType,
    frame_index: int,
    timestamp_ms: int,
    track_id: str = "track-1",
    provenance: EventProvenance = EventProvenance.OBSERVED,
) -> WorldEvent:
    return WorldEvent(
        event_type=event_type,
        frame_index=frame_index,
        timestamp_ms=timestamp_ms,
        subject_track_id=track_id,
        class_name="backpack",
        confidence=0.9,
        provenance=provenance,
        metadata={},
    )


def test_projector_keeps_event_history_by_object_id() -> None:
    projector = WorldStateProjector()

    appeared = event(
        EventType.OBJECT_APPEARED,
        0,
        0,
    )

    projector.update(
        frame_index=0,
        timestamp_ms=0,
        identity_assignments=[assignment()],
        events=[appeared],
    )

    assert projector.history_for("object-1") == (appeared,)
    assert projector.tracks_for("object-1") == ("track-1",)


def test_projector_preserves_last_observed_evidence() -> None:
    projector = WorldStateProjector()

    projector.update(
        frame_index=0,
        timestamp_ms=0,
        identity_assignments=[assignment()],
        events=[
            event(
                EventType.OBJECT_APPEARED,
                0,
                0,
            )
        ],
    )

    beliefs = projector.beliefs_for("object-1")

    bbox_belief = next(
        belief
        for belief in beliefs
        if belief.predicate == "LAST_OBSERVED_BBOX"
    )

    assert bbox_belief.subject_object_id == "object-1"
    assert bbox_belief.provenance.value == "observed"
    assert bbox_belief.value["x1"] == 10
    assert bbox_belief.evidence_track_ids == ("track-1",)


def test_disappearance_does_not_erase_last_observation() -> None:
    projector = WorldStateProjector()

    projector.update(
        frame_index=0,
        timestamp_ms=0,
        identity_assignments=[assignment()],
        events=[
            event(
                EventType.OBJECT_APPEARED,
                0,
                0,
            )
        ],
    )

    projector.update(
        frame_index=1,
        timestamp_ms=33,
        identity_assignments=[],
        events=[
            event(
                EventType.OBJECT_DISAPPEARED,
                1,
                33,
                provenance=EventProvenance.INFERRED,
            )
        ],
    )

    beliefs = projector.beliefs_for("object-1")

    predicates = {
        belief.predicate: belief
        for belief in beliefs
    }

    assert "LAST_OBSERVED_BBOX" in predicates
    assert predicates["CURRENT_VISIBILITY"].value == "not_observed"
    assert (
        predicates["CURRENT_VISIBILITY"].provenance.value
        == "inferred"
    )


def test_new_track_can_extend_same_object_history() -> None:
    projector = WorldStateProjector()

    projector.update(
        frame_index=0,
        timestamp_ms=0,
        identity_assignments=[assignment("track-1", "object-1")],
        events=[
            event(
                EventType.OBJECT_APPEARED,
                0,
                0,
                track_id="track-1",
            )
        ],
    )

    projector.update(
        frame_index=5,
        timestamp_ms=165,
        identity_assignments=[assignment("track-2", "object-1")],
        events=[
            event(
                EventType.OBJECT_APPEARED,
                5,
                165,
                track_id="track-2",
            )
        ],
    )

    assert projector.tracks_for("object-1") == (
        "track-1",
        "track-2",
    )

    beliefs = projector.beliefs_for("object-1")
    for belief in beliefs:
        assert belief.subject_object_id == "object-1"
        assert belief.evidence_track_ids == (
            "track-1",
            "track-2",
        )
