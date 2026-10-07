import pytest

from blackbox.sessions import SessionStore


def test_session_store_creates_and_gets_session() -> None:
    store = SessionStore()
    session = store.create()

    assert store.get(session.session_id) is session


def test_session_store_rejects_unknown_session() -> None:
    store = SessionStore()

    with pytest.raises(KeyError, match="unknown session_id"):
        store.get("missing")
