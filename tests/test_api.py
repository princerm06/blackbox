from fastapi.testclient import TestClient

from blackbox.api import create_app
from blackbox.sessions import SessionStore


def make_client() -> TestClient:
    return TestClient(create_app(SessionStore()))


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
