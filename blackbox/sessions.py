import json
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from blackbox.perception.targets import TargetRegistry, TrackingTarget


@dataclass
class BlackBoxSession:
    session_id: str
    targets: TargetRegistry = field(default_factory=TargetRegistry)


class SessionStore:
    """Session state store.

    When *persist_path* is given the store serialises itself to that JSON file
    after every mutation and restores from it on construction.  Omitting the
    path gives the original in-memory-only behaviour.
    """

    def __init__(self, persist_path: Path | None = None) -> None:
        self._sessions: dict[str, BlackBoxSession] = {}
        self._persist_path = persist_path
        if persist_path is not None:
            self._load()

    # ------------------------------------------------------------------
    # Public API (unchanged from original)
    # ------------------------------------------------------------------

    def create(self) -> BlackBoxSession:
        session = BlackBoxSession(session_id=str(uuid4()))
        self._sessions[session.session_id] = session
        self._save()
        return session

    def get(self, session_id: str) -> BlackBoxSession:
        try:
            return self._sessions[session_id]
        except KeyError as error:
            raise KeyError(f"unknown session_id: {session_id}") from error

    def delete(self, session_id: str) -> None:
        if session_id not in self._sessions:
            raise KeyError(f"unknown session_id: {session_id}")
        del self._sessions[session_id]
        self._save()

    def save(self) -> None:
        """Persist the current state explicitly (call after target mutations)."""
        self._save()

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def _session_to_dict(self, session: BlackBoxSession) -> dict[str, object]:
        return {
            "session_id": session.session_id,
            "targets": [
                {
                    "target_id": t.target_id,
                    "display_name": t.display_name,
                    "prompt": t.prompt,
                    "enabled": t.enabled,
                }
                for t in session.targets.targets()
            ],
        }

    def _session_from_dict(self, data: dict[str, object]) -> BlackBoxSession:
        session = BlackBoxSession(session_id=str(data["session_id"]))
        for t in data.get("targets", []):
            session.targets.register(
                display_name=str(t["display_name"]),
                prompt=str(t["prompt"]),
                target_id=str(t["target_id"]),
            )
            if not t.get("enabled", True):
                session.targets.set_enabled(str(t["target_id"]), False)
        return session

    def _save(self) -> None:
        if self._persist_path is None:
            return
        payload = [
            self._session_to_dict(s) for s in self._sessions.values()
        ]
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._persist_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self._persist_path)

    def _load(self) -> None:
        if self._persist_path is None or not self._persist_path.exists():
            return
        try:
            raw = json.loads(self._persist_path.read_text(encoding="utf-8"))
            for item in raw:
                session = self._session_from_dict(item)
                self._sessions[session.session_id] = session
        except (json.JSONDecodeError, KeyError):
            # json.JSONDecodeError  – file is not valid JSON (truncated/corrupt)
            # KeyError              – a required field is missing from a JSON object
            # Any other exception is a programming error and should propagate.
            self._sessions = {}


def target_to_dict(target: TrackingTarget) -> dict[str, object]:
    return {
        "target_id": target.target_id,
        "display_name": target.display_name,
        "prompt": target.prompt,
        "enabled": target.enabled,
    }
