from pathlib import Path

from fastapi.testclient import TestClient

from blackbox.api import create_app
from blackbox.app_processing import ProcessedRun
from blackbox.perception.pipeline import ProcessingSummary
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
from blackbox.sessions import SessionStore
from blackbox.world.projector import WorldStateProjector


class QueryableProcessor:
    def process(self, registry, input_path, output_dir):
        observation = TrackObservation(
            track_id="track-1",
            class_name="keys",
            confidence=0.91,
            bbox=BoundingBox(10, 20, 50, 60),
        )
        assignment = IdentityAssignment(
            object_id="object-1",
            track_id="track-1",
            class_name="keys",
            confidence=0.91,
            provenance=IdentityProvenance.OBSERVED_TRACK,
            observation=observation,
        )
        event = WorldEvent(
            event_type=EventType.OBJECT_APPEARED,
            frame_index=0,
            timestamp_ms=0,
            subject_track_id="track-1",
            class_name="keys",
            confidence=0.91,
            provenance=EventProvenance.OBSERVED,
            metadata={},
        )

        world_state = WorldStateProjector()
        world_state.update(
            frame_index=0,
            timestamp_ms=0,
            identity_assignments=[assignment],
            events=[event],
        )

        observations = output_dir / "observations.jsonl"
        observations.write_text("{}\n", encoding="utf-8")

        return ProcessedRun(
            summary=ProcessingSummary(
                frames_processed=1,
                detections_emitted=1,
                track_observations_emitted=1,
                identity_assignments_emitted=1,
                events_emitted=1,
                observations_path=observations,
                annotated_video_path=None,
            ),
            world_state=world_state,
            source_video_path=input_path,
        )


def test_where_query_returns_grounded_answer(tmp_path: Path) -> None:
    client = TestClient(
        create_app(
            store=SessionStore(),
            processor=QueryableProcessor(),
            workspace_root=tmp_path,
        )
    )
    session_id = client.post("/sessions").json()["session_id"]
    client.post(
        f"/sessions/{session_id}/targets",
        json={"display_name": "My Keys", "prompt": "keys"},
    )
    upload = client.post(
        f"/sessions/{session_id}/videos",
        files={"video": ("clip.mp4", b"video", "video/mp4")},
    )
    run_id = upload.json()["run_id"]

    response = client.get(
        f"/sessions/{session_id}/runs/{run_id}"
        "/targets/my-keys/where"
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["target"]["display_name"] == "My Keys"
    assert payload["object_id"] == "object-1"
    assert payload["answer"]["provenance"] == "observed"
    assert "My Keys" in payload["answer"]["answer"]
    assert payload["answer"]["confidence"] == 1.0
    assert payload["reobservation"]["should_reobserve"] is False
    assert payload["ambiguity"] is None


def test_where_query_reports_target_not_detected(tmp_path: Path) -> None:
    client = TestClient(
        create_app(
            store=SessionStore(),
            processor=QueryableProcessor(),
            workspace_root=tmp_path,
        )
    )
    session_id = client.post("/sessions").json()["session_id"]
    client.post(
        f"/sessions/{session_id}/targets",
        json={"display_name": "Wallet", "prompt": "wallet"},
    )
    upload = client.post(
        f"/sessions/{session_id}/videos",
        files={"video": ("clip.mp4", b"video", "video/mp4")},
    )
    run_id = upload.json()["run_id"]

    response = client.get(
        f"/sessions/{session_id}/runs/{run_id}"
        "/targets/wallet/where"
    )

    assert response.status_code == 404
    assert "no detected object matched" in response.json()["detail"]



def test_run_objects_endpoint_returns_status_summary(tmp_path: Path) -> None:
    client = TestClient(
        create_app(
            store=SessionStore(),
            processor=QueryableProcessor(),
            workspace_root=tmp_path,
        )
    )
    session_id = client.post("/sessions").json()["session_id"]
    client.post(
        f"/sessions/{session_id}/targets",
        json={"display_name": "My Keys", "prompt": "keys"},
    )
    upload = client.post(
        f"/sessions/{session_id}/videos",
        files={"video": ("clip.mp4", b"video", "video/mp4")},
    )
    run_id = upload.json()["run_id"]

    response = client.get(
        f"/sessions/{session_id}/runs/{run_id}/objects"
    )

    assert response.status_code == 200
    objects = response.json()["objects"]
    assert len(objects) == 1
    assert objects[0]["object_id"] == "object-1"
    assert objects[0]["class_name"] == "keys"
    assert objects[0]["visibility"] == "observed"
    assert objects[0]["last_observed_frame"] == 0
    assert objects[0]["has_uncertainty"] is False


def test_run_events_endpoint_returns_timeline(tmp_path: Path) -> None:
    client = TestClient(
        create_app(
            store=SessionStore(),
            processor=QueryableProcessor(),
            workspace_root=tmp_path,
        )
    )
    session_id = client.post("/sessions").json()["session_id"]
    client.post(
        f"/sessions/{session_id}/targets",
        json={"display_name": "My Keys", "prompt": "keys"},
    )
    upload = client.post(
        f"/sessions/{session_id}/videos",
        files={"video": ("clip.mp4", b"video", "video/mp4")},
    )
    run_id = upload.json()["run_id"]

    response = client.get(
        f"/sessions/{session_id}/runs/{run_id}/events"
    )

    assert response.status_code == 200
    events = response.json()["events"]
    assert len(events) == 1
    assert events[0]["object_id"] == "object-1"
    assert events[0]["event_type"] == "OBJECT_APPEARED"
    assert events[0]["provenance"] == "observed"
    assert events[0]["frame_index"] == 0
