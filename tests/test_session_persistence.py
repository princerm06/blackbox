"""Tests for session and target persistence across restarts.

The key invariant: a SessionStore constructed with a persist_path will
restore all sessions and their targets exactly as they were when the
previous store was shut down.
"""

import json
from pathlib import Path

import pytest

from blackbox.sessions import SessionStore


# ---------------------------------------------------------------------------
# SessionStore persistence unit tests
# ---------------------------------------------------------------------------


def test_empty_store_creates_file_on_first_session(tmp_path: Path) -> None:
    p = tmp_path / "sessions.json"
    store = SessionStore(persist_path=p)
    store.create()

    assert p.exists()
    data = json.loads(p.read_text())
    assert len(data) == 1


def test_session_survives_restart(tmp_path: Path) -> None:
    p = tmp_path / "sessions.json"

    store1 = SessionStore(persist_path=p)
    session = store1.create()
    session_id = session.session_id

    # Simulate restart by constructing a new store from the same file.
    store2 = SessionStore(persist_path=p)
    restored = store2.get(session_id)

    assert restored.session_id == session_id


def test_targets_survive_restart(tmp_path: Path) -> None:
    p = tmp_path / "sessions.json"

    store1 = SessionStore(persist_path=p)
    session = store1.create()
    session.targets.register(display_name="My Keys", prompt="keys")
    session.targets.register(display_name="Blue Backpack", prompt="backpack")
    store1.save()

    store2 = SessionStore(persist_path=p)
    restored = store2.get(session.session_id)
    targets = {t.target_id: t for t in restored.targets.targets()}

    assert set(targets) == {"my-keys", "blue-backpack"}
    assert targets["my-keys"].display_name == "My Keys"
    assert targets["my-keys"].prompt == "keys"
    assert targets["blue-backpack"].prompt == "backpack"


def test_disabled_target_survives_restart(tmp_path: Path) -> None:
    p = tmp_path / "sessions.json"

    store1 = SessionStore(persist_path=p)
    session = store1.create()
    session.targets.register(display_name="My Keys", prompt="keys")
    session.targets.set_enabled("my-keys", False)
    store1.save()

    store2 = SessionStore(persist_path=p)
    restored = store2.get(session.session_id)
    target = next(t for t in restored.targets.targets() if t.target_id == "my-keys")

    assert target.enabled is False


def test_deleted_session_not_restored(tmp_path: Path) -> None:
    p = tmp_path / "sessions.json"

    store1 = SessionStore(persist_path=p)
    session = store1.create()
    session_id = session.session_id
    store1.delete(session_id)

    store2 = SessionStore(persist_path=p)

    with pytest.raises(KeyError, match="unknown session_id"):
        store2.get(session_id)


def test_multiple_sessions_survive_restart(tmp_path: Path) -> None:
    p = tmp_path / "sessions.json"

    store1 = SessionStore(persist_path=p)
    s1 = store1.create()
    s2 = store1.create()
    s1.targets.register(display_name="Keys", prompt="keys")
    store1.save()  # flush target mutation (mirrors what the API layer does)

    store2 = SessionStore(persist_path=p)

    assert len(list(store2.get(s1.session_id).targets.targets())) == 1
    assert len(list(store2.get(s2.session_id).targets.targets())) == 0


def test_corrupt_persist_file_starts_fresh(tmp_path: Path) -> None:
    p = tmp_path / "sessions.json"
    p.write_text("not valid json", encoding="utf-8")

    # Must not raise; starts with an empty store.
    store = SessionStore(persist_path=p)
    assert store._sessions == {}


def test_missing_field_in_persist_file_starts_fresh(tmp_path: Path) -> None:
    """A session entry missing a required key is treated as corrupt."""
    import json

    p = tmp_path / "sessions.json"
    # 'session_id' key is missing — _session_from_dict raises KeyError.
    p.write_text(json.dumps([{"targets": []}]), encoding="utf-8")

    store = SessionStore(persist_path=p)
    assert store._sessions == {}


def test_programming_error_in_load_propagates(tmp_path: Path) -> None:
    """Non-persistence errors (e.g. TypeError) must not be silenced."""
    import json

    p = tmp_path / "sessions.json"
    # Top-level JSON is a dict, not a list — iterating it yields string keys,
    # and accessing item["session_id"] on a plain string raises TypeError,
    # which is a programming/format contract violation, not a corrupt file.
    p.write_text(json.dumps({"bad": "shape"}), encoding="utf-8")

    with pytest.raises(TypeError):
        SessionStore(persist_path=p)


def test_missing_persist_file_starts_fresh(tmp_path: Path) -> None:
    p = tmp_path / "no-such-file.json"

    store = SessionStore(persist_path=p)
    assert store._sessions == {}


def test_store_without_persist_path_is_in_memory_only(tmp_path: Path) -> None:
    """A store with no path never touches the filesystem."""
    store = SessionStore()
    s = store.create()
    # No file should have been written.
    assert not any(tmp_path.iterdir())
    assert store.get(s.session_id) is s


def test_write_is_atomic(tmp_path: Path) -> None:
    """The .tmp file should not be left behind after a successful save."""
    p = tmp_path / "sessions.json"
    store = SessionStore(persist_path=p)
    store.create()

    tmp = p.with_suffix(".tmp")
    assert not tmp.exists(), ".tmp file was not cleaned up after save"


# ---------------------------------------------------------------------------
# API-level restart simulation
# ---------------------------------------------------------------------------


def test_api_session_and_targets_survive_restart(tmp_path: Path) -> None:
    """Verify the full API stack round-trip through a restart."""
    from fastapi.testclient import TestClient
    from blackbox.api import create_app

    p = tmp_path / "sessions.json"

    # --- First app instance: create session, register targets ---
    app1 = create_app(persist_path=p, workspace_root=tmp_path / "runs")
    client1 = TestClient(app1)

    session_id = client1.post("/sessions").json()["session_id"]

    r = client1.post(
        f"/sessions/{session_id}/targets",
        json={"display_name": "My Keys", "prompt": "keys"},
    )
    assert r.status_code == 201

    r = client1.post(
        f"/sessions/{session_id}/targets",
        json={"display_name": "Wallet", "prompt": "wallet"},
    )
    assert r.status_code == 201

    # Disable wallet
    client1.patch(
        f"/sessions/{session_id}/targets/wallet",
        json={"enabled": False},
    )

    # --- Second app instance: should see the same session and targets ---
    app2 = create_app(persist_path=p, workspace_root=tmp_path / "runs")
    client2 = TestClient(app2)

    listed = client2.get(f"/sessions/{session_id}/targets")
    assert listed.status_code == 200

    targets = {t["target_id"]: t for t in listed.json()["targets"]}
    assert set(targets) == {"my-keys", "wallet"}
    assert targets["my-keys"]["enabled"] is True
    assert targets["wallet"]["enabled"] is False


def test_api_unknown_session_still_404_after_restart(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient
    from blackbox.api import create_app

    p = tmp_path / "sessions.json"
    app = create_app(persist_path=p, workspace_root=tmp_path / "runs")
    client = TestClient(app)

    response = client.get("/sessions/does-not-exist/targets")
    assert response.status_code == 404
