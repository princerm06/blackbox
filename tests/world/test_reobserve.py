from blackbox.reasoning.reobserve import (
    ReobservationPlanner,
    normalized_entropy,
)
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


def hidden_keys_projector() -> WorldStateProjector:
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
    return projector


def test_normalized_entropy_distinguishes_ambiguity() -> None:
    assert normalized_entropy([1.0, 0.0]) == 0.0
    assert normalized_entropy([0.5, 0.5]) > 0.99


def test_hidden_object_requests_reobservation_when_ambiguous() -> None:
    planner = ReobservationPlanner(
        hidden_keys_projector(),
        uncertainty_threshold=0.5,
    )

    plan = planner.plan_for("keys-object")

    assert plan.should_reobserve
    assert plan.uncertainty >= 0.5
    assert plan.targets
    assert any(
        target.target == "inspect:object:bag-object"
        for target in plan.targets
    )


def test_visible_object_does_not_request_reobservation() -> None:
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

    plan = ReobservationPlanner(projector).plan_for("keys-object")

    assert not plan.should_reobserve
    assert plan.targets == ()


def test_unknown_object_does_not_request_reobservation() -> None:
    plan = ReobservationPlanner(
        WorldStateProjector()
    ).plan_for("missing-object")

    assert not plan.should_reobserve
    assert plan.uncertainty == 0.0
