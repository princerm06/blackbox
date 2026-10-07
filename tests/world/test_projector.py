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


def test_disappearance_creates_competing_uncertainty_hypotheses() -> None:
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

    distributions = projector.uncertainty_for("object-1")

    assert len(distributions) == 1

    distribution = distributions[0]

    assert distribution.predicate == "UNOBSERVED_CHANGE_STATE"
    assert {
        hypothesis.value
        for hypothesis in distribution.hypotheses
    } == {
        "unchanged_since_last_observation",
        "changed_while_unobserved",
    }
    assert sum(
        hypothesis.weight
        for hypothesis in distribution.hypotheses
    ) == 1.0


def test_observed_object_has_no_unobserved_change_distribution() -> None:
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

    assert projector.uncertainty_for("object-1") == ()


def test_elapsed_unseen_time_increases_changed_weight() -> None:
    projector = WorldStateProjector()

    projector.update(
        frame_index=0,
        timestamp_ms=0,
        identity_assignments=[assignment()],
        events=[event(EventType.OBJECT_APPEARED, 0, 0)],
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

    initial = projector.uncertainty_for("object-1")[0]
    initial_changed = next(
        hypothesis.weight
        for hypothesis in initial.hypotheses
        if hypothesis.value == "changed_while_unobserved"
    )

    projector.update(
        frame_index=11,
        timestamp_ms=363,
        identity_assignments=[],
        events=[],
    )

    later = projector.uncertainty_for("object-1")[0]
    later_changed = next(
        hypothesis.weight
        for hypothesis in later.hypotheses
        if hypothesis.value == "changed_while_unobserved"
    )

    assert initial_changed == 0.5
    assert later_changed > initial_changed


def test_other_object_motion_is_change_opportunity() -> None:
    projector = WorldStateProjector()

    projector.update(
        frame_index=0,
        timestamp_ms=0,
        identity_assignments=[
            assignment("track-1", "object-1"),
            assignment("track-2", "object-2"),
        ],
        events=[
            event(
                EventType.OBJECT_APPEARED,
                0,
                0,
                track_id="track-1",
            ),
            event(
                EventType.OBJECT_APPEARED,
                0,
                0,
                track_id="track-2",
            ),
        ],
    )

    projector.update(
        frame_index=1,
        timestamp_ms=33,
        identity_assignments=[
            assignment("track-2", "object-2"),
        ],
        events=[
            event(
                EventType.OBJECT_DISAPPEARED,
                1,
                33,
                track_id="track-1",
                provenance=EventProvenance.INFERRED,
            )
        ],
    )

    projector.update(
        frame_index=2,
        timestamp_ms=66,
        identity_assignments=[
            assignment("track-2", "object-2"),
        ],
        events=[
            event(
                EventType.OBJECT_MOVED,
                2,
                66,
                track_id="track-2",
            )
        ],
    )

    signals = projector.opportunities_for("object-1")

    assert any(
        signal.signal_type.value == "OTHER_OBJECT_MOVED"
        and signal.source_object_id == "object-2"
        for signal in signals
    )

    distribution = projector.uncertainty_for("object-1")[0]
    changed_weight = next(
        hypothesis.weight
        for hypothesis in distribution.hypotheses
        if hypothesis.value == "changed_while_unobserved"
    )

    assert changed_weight > 0.5


def test_person_object_interaction_is_strong_change_opportunity() -> None:
    projector = WorldStateProjector()

    projector.update(
        frame_index=0,
        timestamp_ms=0,
        identity_assignments=[
            assignment("track-1", "object-1"),
            assignment("track-2", "object-2"),
        ],
        events=[
            event(
                EventType.OBJECT_APPEARED,
                0,
                0,
                track_id="track-1",
            ),
            event(
                EventType.OBJECT_APPEARED,
                0,
                0,
                track_id="track-2",
            ),
        ],
    )

    projector.update(
        frame_index=1,
        timestamp_ms=33,
        identity_assignments=[
            assignment("track-2", "object-2"),
        ],
        events=[
            event(
                EventType.OBJECT_DISAPPEARED,
                1,
                33,
                track_id="track-1",
                provenance=EventProvenance.INFERRED,
            )
        ],
    )

    before = projector.uncertainty_for("object-1")[0]
    before_changed = next(
        hypothesis.weight
        for hypothesis in before.hypotheses
        if hypothesis.value == "changed_while_unobserved"
    )

    interaction = WorldEvent(
        event_type=EventType.PERSON_OBJECT_INTERACTION,
        frame_index=2,
        timestamp_ms=66,
        subject_track_id="track-2",
        class_name="backpack",
        confidence=0.8,
        provenance=EventProvenance.INFERRED,
        metadata={"person_track_id": "person-1"},
        related_track_ids=("person-1",),
    )

    projector.update(
        frame_index=2,
        timestamp_ms=66,
        identity_assignments=[
            assignment("track-2", "object-2"),
        ],
        events=[interaction],
    )

    signals = projector.opportunities_for("object-1")

    assert any(
        signal.signal_type.value == "PERSON_OBJECT_INTERACTION"
        and signal.source_object_id == "object-2"
        for signal in signals
    )

    after = projector.uncertainty_for("object-1")[0]
    after_changed = next(
        hypothesis.weight
        for hypothesis in after.hypotheses
        if hypothesis.value == "changed_while_unobserved"
    )

    assert after_changed > before_changed
    assert after_changed - before_changed >= 0.18


def positioned_assignment(
    track_id: str,
    object_id: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> IdentityAssignment:
    obs = TrackObservation(
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
        observation=obs,
    )


def test_related_source_event_gets_stronger_opportunity_weight() -> None:
    projector = WorldStateProjector()

    projector.update(
        frame_index=0,
        timestamp_ms=0,
        identity_assignments=[
            positioned_assignment(
                "keys-track",
                "keys-object",
                0,
                0,
                40,
                40,
            ),
            positioned_assignment(
                "bag-track",
                "bag-object",
                20,
                0,
                120,
                100,
            ),
            positioned_assignment(
                "mug-track",
                "mug-object",
                500,
                0,
                560,
                60,
            ),
        ],
        events=[
            WorldEvent(
                event_type=EventType.OBJECT_APPEARED,
                frame_index=0,
                timestamp_ms=0,
                subject_track_id="keys-track",
                class_name="keys",
                confidence=0.9,
                provenance=EventProvenance.OBSERVED,
                metadata={},
            ),
            WorldEvent(
                event_type=EventType.OBJECT_APPEARED,
                frame_index=0,
                timestamp_ms=0,
                subject_track_id="bag-track",
                class_name="backpack",
                confidence=0.9,
                provenance=EventProvenance.OBSERVED,
                metadata={},
            ),
            WorldEvent(
                event_type=EventType.OBJECT_APPEARED,
                frame_index=0,
                timestamp_ms=0,
                subject_track_id="mug-track",
                class_name="mug",
                confidence=0.9,
                provenance=EventProvenance.OBSERVED,
                metadata={},
            ),
        ],
    )

    projector.update(
        frame_index=1,
        timestamp_ms=33,
        identity_assignments=[
            positioned_assignment(
                "bag-track",
                "bag-object",
                20,
                0,
                120,
                100,
            ),
            positioned_assignment(
                "mug-track",
                "mug-object",
                500,
                0,
                560,
                60,
            ),
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
            positioned_assignment(
                "bag-track",
                "bag-object",
                20,
                0,
                120,
                100,
            ),
            positioned_assignment(
                "mug-track",
                "mug-object",
                500,
                0,
                560,
                60,
            ),
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
            ),
            WorldEvent(
                event_type=EventType.PERSON_OBJECT_INTERACTION,
                frame_index=2,
                timestamp_ms=66,
                subject_track_id="mug-track",
                class_name="mug",
                confidence=0.8,
                provenance=EventProvenance.INFERRED,
                metadata={},
            ),
        ],
    )

    interaction_signals = [
        signal
        for signal in projector.opportunities_for("keys-object")
        if signal.signal_type.value == "PERSON_OBJECT_INTERACTION"
    ]

    bag_signal = next(
        signal
        for signal in interaction_signals
        if signal.source_object_id == "bag-object"
    )
    mug_signal = next(
        signal
        for signal in interaction_signals
        if signal.source_object_id == "mug-object"
    )

    assert bag_signal.relation_type == "SPATIALLY_ASSOCIATED"
    assert bag_signal.relation_confidence is not None
    assert bag_signal.strength > mug_signal.strength
    assert mug_signal.relation_type is None


def test_spatial_relation_is_not_claimed_as_containment() -> None:
    projector = WorldStateProjector()

    projector.update(
        frame_index=0,
        timestamp_ms=0,
        identity_assignments=[
            positioned_assignment(
                "keys-track",
                "keys-object",
                0,
                0,
                40,
                40,
            ),
            positioned_assignment(
                "bag-track",
                "bag-object",
                20,
                0,
                120,
                100,
            ),
        ],
        events=[],
    )

    relations = projector.relations_for("keys-object")

    assert len(relations) == 1
    assert relations[0].relation_type.value == "SPATIALLY_ASSOCIATED"
