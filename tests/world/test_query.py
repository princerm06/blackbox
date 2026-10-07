from blackbox.reasoning.query import BeliefQueryEngine
from blackbox.schemas.events import EventProvenance, EventType, WorldEvent
from blackbox.schemas.identity import IdentityAssignment, IdentityProvenance
from blackbox.schemas.perception import BoundingBox, TrackObservation
from blackbox.world.projector import WorldStateProjector


def assignment(
    track_id: str,
    object_id: str,
    bbox: BoundingBox,
    class_name: str,
) -> IdentityAssignment:
    observation = TrackObservation(
        track_id=track_id,
        class_name=class_name,
        confidence=0.9,
        bbox=bbox,
    )
    return IdentityAssignment(
        object_id=object_id,
        track_id=track_id,
        class_name=class_name,
        confidence=1.0,
        provenance=IdentityProvenance.OBSERVED_TRACK,
        observation=observation,
    )


def appeared(track_id: str, class_name: str) -> WorldEvent:
    return WorldEvent(
        event_type=EventType.OBJECT_APPEARED,
        frame_index=0,
        timestamp_ms=0,
        subject_track_id=track_id,
        class_name=class_name,
        confidence=0.9,
        provenance=EventProvenance.OBSERVED,
        metadata={},
    )


def test_query_reports_current_observation_without_inference() -> None:
    projector = WorldStateProjector()
    projector.update(
        frame_index=0,
        timestamp_ms=0,
        identity_assignments=[
            assignment(
                "keys-track",
                "keys-object",
                BoundingBox(0, 0, 40, 40),
                "keys",
            )
        ],
        events=[appeared("keys-track", "keys")],
    )

    answer = BeliefQueryEngine(projector).where_is("keys-object")

    assert answer.provenance == "observed"
    assert answer.primary_hypothesis == "currently_observed"
    assert answer.caveats == ()


def test_query_ranks_related_candidate_but_preserves_caveat() -> None:
    projector = WorldStateProjector()
    projector.update(
        frame_index=0,
        timestamp_ms=0,
        identity_assignments=[
            assignment(
                "keys-track",
                "keys-object",
                BoundingBox(0, 0, 40, 40),
                "keys",
            ),
            assignment(
                "bag-track",
                "bag-object",
                BoundingBox(20, 0, 120, 100),
                "backpack",
            ),
        ],
        events=[
            appeared("keys-track", "keys"),
            appeared("bag-track", "backpack"),
        ],
    )
    projector.update(
        frame_index=1,
        timestamp_ms=33,
        identity_assignments=[
            assignment(
                "bag-track",
                "bag-object",
                BoundingBox(20, 0, 120, 100),
                "backpack",
            )
        ],
        events=[
            WorldEvent(
                event_type=EventType.OBJECT_DISAPPEARED,
                frame_index=1,
                timestamp_ms=33,
                subject_track_id="keys-track",
                class_name="keys",
                confidence=1.0,
                provenance=EventProvenance.INFERRED,
                metadata={},
            )
        ],
    )
    projector.update(
        frame_index=2,
        timestamp_ms=66,
        identity_assignments=[
            assignment(
                "bag-track",
                "bag-object",
                BoundingBox(20, 0, 120, 100),
                "backpack",
            )
        ],
        events=[
            WorldEvent(
                event_type=EventType.PERSON_OBJECT_INTERACTION,
                frame_index=2,
                timestamp_ms=66,
                subject_track_id="bag-track",
                class_name="backpack",
                confidence=0.8,
                provenance=EventProvenance.INFERRED,
                metadata={},
            )
        ],
    )

    answer = BeliefQueryEngine(projector).where_is("keys-object")

    assert answer.provenance == "inferred"
    assert answer.primary_hypothesis in {
        "last_seen_area",
        "associated_with:bag-object",
    }
    assert answer.alternatives
    assert any(
        "not a direct observation" in caveat
        for caveat in answer.caveats
    )
    assert "inside" not in answer.answer.lower()


def test_query_handles_unknown_object() -> None:
    answer = BeliefQueryEngine(
        WorldStateProjector()
    ).where_is("missing-object")

    assert answer.confidence == 0.0
    assert answer.primary_hypothesis == "unknown"
