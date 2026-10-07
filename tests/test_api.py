from pathlib import Path

from fastapi.testclient import TestClient

from blackbox.api import create_app
from blackbox.app_processing import ProcessedRun
from blackbox.perception.pipeline import ProcessingSummary
from blackbox.sessions import SessionStore
from blackbox.world.projector import WorldStateProjector


class FakeProcessor:
    def __init__(self) -> None:
        self.calls = []

    def process(self, registry, input_path, output_dir):
        self.calls.append(
            {
                "prompts": registry.prompts(),
                "input_path": input_path,
                "output_dir": output_dir,
            }
        )
        observations = output_dir / "observations.jsonl"
        observations.write_text("{}\n", encoding="utf-8")
        annotated = output_dir / "annotated.mp4"
        annotated.write_bytes(b"fake")

        summary = ProcessingSummary(
            frames_processed=12,
            detections_emitted=8,
            track_observations_emitted=7,
            identity_assignments_emitted=7,
            events_emitted=3,
            observations_path=observations,
            annotated_video_path=annotated,
        )
        return ProcessedRun(
            summary=summary,
            world_state=WorldStateProjector(),
            source_video_path=input_path,
        )


def make_client(
    processor=None,
    workspace_root: Path = Path("runs") / "test-api",
) -> TestClient:
    return TestClient(
        create_app(
            SessionStore(),
            processor=processor,
            workspace_root=workspace_root,
        )
    )


def test_health_endpoint() -> None:
    response = make_client().get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_register_list_disable_and_remove_target() -> None:
    client = make_client()
    session_id = client.post("/sessions").json()["session_id"]

    create = client.post(
        f"/sessions/{session_id}/targets",
        json={
            "display_name": "My Keys",
            "prompt": "keys",
        },
    )
    assert create.status_code == 201
    target = create.json()
    assert target["target_id"] == "my-keys"
    assert target["enabled"] is True

    listed = client.get(
        f"/sessions/{session_id}/targets"
    )
    assert listed.status_code == 200
    assert listed.json()["targets"] == [target]

    disabled = client.patch(
        f"/sessions/{session_id}/targets/my-keys",
        json={"enabled": False},
    )
    assert disabled.status_code == 200
    assert disabled.json()["enabled"] is False

    removed = client.delete(
        f"/sessions/{session_id}/targets/my-keys"
    )
    assert removed.status_code == 204

    listed_again = client.get(
        f"/sessions/{session_id}/targets"
    )
    assert listed_again.json()["targets"] == []


def test_unknown_session_returns_404() -> None:
    response = make_client().get("/sessions/missing/targets")

    assert response.status_code == 404


def test_video_upload_requires_registered_target(tmp_path) -> None:
    client = make_client(FakeProcessor(), tmp_path)
    session_id = client.post("/sessions").json()["session_id"]

    response = client.post(
        f"/sessions/{session_id}/videos",
        files={"video": ("clip.mp4", b"video-bytes", "video/mp4")},
    )

    assert response.status_code == 409


def test_video_upload_uses_session_targets(tmp_path) -> None:
    processor = FakeProcessor()
    client = make_client(processor, tmp_path)
    session_id = client.post("/sessions").json()["session_id"]

    client.post(
        f"/sessions/{session_id}/targets",
        json={
            "display_name": "My Keys",
            "prompt": "keys",
        },
    )
    client.post(
        f"/sessions/{session_id}/targets",
        json={
            "display_name": "Blue Backpack",
            "prompt": "backpack",
        },
    )

    response = client.post(
        f"/sessions/{session_id}/videos",
        files={"video": ("clip.mp4", b"video-bytes", "video/mp4")},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["session_id"] == session_id
    assert [target["prompt"] for target in payload["targets"]] == [
        "keys",
        "backpack",
    ]
    assert payload["summary"]["frames_processed"] == 12
    assert processor.calls[0]["prompts"] == ("keys", "backpack")
    assert processor.calls[0]["input_path"].read_bytes() == b"video-bytes"


def test_empty_video_is_rejected(tmp_path) -> None:
    client = make_client(FakeProcessor(), tmp_path)
    session_id = client.post("/sessions").json()["session_id"]
    client.post(
        f"/sessions/{session_id}/targets",
        json={"display_name": "My Keys", "prompt": "keys"},
    )

    response = client.post(
        f"/sessions/{session_id}/videos",
        files={"video": ("clip.mp4", b"", "video/mp4")},
    )

    assert response.status_code == 400



def test_run_snapshot_exposes_frontend_contract(tmp_path) -> None:
    client = make_client(FakeProcessor(), tmp_path)
    session_id = client.post("/sessions").json()["session_id"]
    client.post(
        f"/sessions/{session_id}/targets",
        json={"display_name": "My Keys", "prompt": "keys"},
    )
    upload = client.post(
        f"/sessions/{session_id}/videos",
        files={"video": ("clip.mp4", b"video-bytes", "video/mp4")},
    )
    run_id = upload.json()["run_id"]

    response = client.get(
        f"/sessions/{session_id}/runs/{run_id}"
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == session_id
    assert payload["run_id"] == run_id
    assert payload["summary"]["frames_processed"] == 12
    assert payload["objects"] == []
    assert payload["events"] == []
    assert payload["media"]["source"].endswith("/media/source")
    assert payload["media"]["annotated"].endswith("/media/annotated")


def test_run_media_endpoints_serve_source_and_annotated_video(tmp_path) -> None:
    client = make_client(FakeProcessor(), tmp_path)
    session_id = client.post("/sessions").json()["session_id"]
    client.post(
        f"/sessions/{session_id}/targets",
        json={"display_name": "My Keys", "prompt": "keys"},
    )
    upload = client.post(
        f"/sessions/{session_id}/videos",
        files={"video": ("clip.mp4", b"video-bytes", "video/mp4")},
    )
    run_id = upload.json()["run_id"]

    source = client.get(
        f"/sessions/{session_id}/runs/{run_id}/media/source"
    )
    annotated = client.get(
        f"/sessions/{session_id}/runs/{run_id}/media/annotated"
    )

    assert source.status_code == 200
    assert source.content == b"video-bytes"
    assert source.headers["content-type"].startswith("video/mp4")

    assert annotated.status_code == 200
    assert annotated.content == b"fake"
    assert annotated.headers["content-type"].startswith("video/mp4")
