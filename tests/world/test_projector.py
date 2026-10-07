from blackbox.schemas.events import (
    EventProvenance,
    EventType,
    WorldEvent,
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


def event(
    event_type: EventType,
    frame_index: int,
    timestamp_ms: int,
    provenance: EventProvenance,
) -> WorldEvent:
    return WorldEvent(
        event_type=event_type,
        frame_index=frame_index,
        timestamp_ms=timestamp_ms,
        subject_track_id="track-1",
        class_name="backpack",
        confidence=0.9,
        provenance=provenance,
        metadata={},
    )


def test_projector_keeps_event_history() -> None:
    projector = WorldStateProjector()

    appeared = event(
        EventType.OBJECT_APPEARED,
        0,
        0,
        EventProvenance.OBSERVED,
    )

    projector.update(
        frame_index=0,
        timestamp_ms=0,
        observations=[observation()],
        events=[appeared],
    )

    assert projector.history_for("track-1") == (appeared,)


def test_projector_preserves_last_observed_evidence() -> None:
    projector = WorldStateProjector()

    projector.update(
        frame_index=0,
        timestamp_ms=0,
        observations=[observation()],
        events=[
            event(
                EventType.OBJECT_APPEARED,
                0,
                0,
                EventProvenance.OBSERVED,
            )
        ],
    )

    beliefs = projector.beliefs_for("track-1")

    bbox_belief = next(
        belief
        for belief in beliefs
        if belief.predicate == "LAST_OBSERVED_BBOX"
    )

    assert bbox_belief.provenance.value == "observed"
    assert bbox_belief.value["x1"] == 10


def test_disappearance_does_not_erase_last_observation() -> None:
    projector = WorldStateProjector()

    projector.update(
        frame_index=0,
        timestamp_ms=0,
        observations=[observation()],
        events=[
            event(
                EventType.OBJECT_APPEARED,
                0,
                0,
                EventProvenance.OBSERVED,
            )
        ],
    )

    projector.update(
        frame_index=1,
        timestamp_ms=33,
        observations=[],
        events=[
            event(
                EventType.OBJECT_DISAPPEARED,
                1,
                33,
                EventProvenance.INFERRED,
            )
        ],
    )

    beliefs = projector.beliefs_for("track-1")

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
