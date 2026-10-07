from dataclasses import dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class RecordedEpisode:
    """Ground-truth annotation for one real recorded unseen episode."""

    episode_id: str
    video_path: str
    target_class: str
    changed_while_unobserved: bool
    final_location_label: str
    evaluation_frame: int | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id must not be empty")
        if not self.video_path:
            raise ValueError("video_path must not be empty")
        if not self.target_class:
            raise ValueError("target_class must not be empty")
        if not self.final_location_label:
            raise ValueError("final_location_label must not be empty")
        if self.evaluation_frame is not None and self.evaluation_frame < 0:
            raise ValueError("evaluation_frame must be non-negative")


def load_episode_manifest(path: Path) -> tuple[RecordedEpisode, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("episode manifest must be a JSON list")

    episodes = tuple(
        RecordedEpisode(
            episode_id=str(item["episode_id"]),
            video_path=str(item["video_path"]),
            target_class=str(item["target_class"]),
            changed_while_unobserved=bool(
                item["changed_while_unobserved"]
            ),
            final_location_label=str(item["final_location_label"]),
            evaluation_frame=(
                int(item["evaluation_frame"])
                if item.get("evaluation_frame") is not None
                else None
            ),
            notes=str(item.get("notes", "")),
        )
        for item in payload
    )

    ids = [episode.episode_id for episode in episodes]
    if len(ids) != len(set(ids)):
        raise ValueError("episode_id values must be unique")

    return episodes
