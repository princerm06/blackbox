from dataclasses import dataclass, field
from uuid import uuid4

from blackbox.perception.targets import TargetRegistry, TrackingTarget


@dataclass
class BlackBoxSession:
    session_id: str
    targets: TargetRegistry = field(default_factory=TargetRegistry)


class SessionStore:
    """In-memory session state used by the first local BlackBox app."""

    def __init__(self) -> None:
        self._sessions: dict[str, BlackBoxSession] = {}

    def create(self) -> BlackBoxSession:
        session = BlackBoxSession(session_id=str(uuid4()))
        self._sessions[session.session_id] = session
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


def target_to_dict(target: TrackingTarget) -> dict[str, object]:
    return {
        "target_id": target.target_id,
        "display_name": target.display_name,
        "prompt": target.prompt,
        "enabled": target.enabled,
    }
