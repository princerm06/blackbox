import json

import pytest

from blackbox.experiments.recorded import load_episode_manifest


def test_load_episode_manifest(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            [
                {
                    "episode_id": "keys-001",
                    "video_path": "data/episodes/raw/keys-001.mp4",
                    "target_class": "keys",
                    "changed_while_unobserved": True,
                    "final_location_label": "inside_backpack",
                    "notes": "controlled clip",
                }
            ]
        ),
        encoding="utf-8",
    )

    episodes = load_episode_manifest(path)

    assert len(episodes) == 1
    assert episodes[0].episode_id == "keys-001"
    assert episodes[0].changed_while_unobserved
    assert episodes[0].final_location_label == "inside_backpack"


def test_manifest_rejects_duplicate_episode_ids(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    item = {
        "episode_id": "keys-001",
        "video_path": "video.mp4",
        "target_class": "keys",
        "changed_while_unobserved": False,
        "final_location_label": "desk",
    }
    path.write_text(json.dumps([item, item]), encoding="utf-8")

    with pytest.raises(ValueError):
        load_episode_manifest(path)


def test_manifest_requires_json_list(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"episode_id": "bad"}), encoding="utf-8")

    with pytest.raises(ValueError):
        load_episode_manifest(path)
