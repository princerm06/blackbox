import json

import pytest

from blackbox.experiments.recorded import RecordedEpisode
from blackbox.experiments.recorded_eval import (
    evaluate_recorded_predictions,
    extract_change_prediction,
)


def write_records(path) -> None:
    records = [
        {
            "frame_index": 0,
            "identities": [
                {
                    "object_id": "object-1",
                    "class_name": "keys",
                }
            ],
            "uncertainty": {},
        },
        {
            "frame_index": 5,
            "identities": [],
            "uncertainty": {
                "object-1": [
                    {
                        "predicate": "UNOBSERVED_CHANGE_STATE",
                        "hypotheses": [
                            {
                                "value": "unchanged_since_last_observation",
                                "weight": 0.35,
                            },
                            {
                                "value": "changed_while_unobserved",
                                "weight": 0.65,
                            },
                        ],
                    }
                ]
            },
        },
        {
            "frame_index": 9,
            "identities": [
                {
                    "object_id": "object-1",
                    "class_name": "keys",
                }
            ],
            "uncertainty": {},
        },
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_extract_change_prediction_uses_evaluation_frame(tmp_path) -> None:
    observations = tmp_path / "observations.jsonl"
    write_records(observations)
    episode = RecordedEpisode(
        episode_id="keys-001",
        video_path="keys-001.mp4",
        target_class="keys",
        changed_while_unobserved=True,
        final_location_label="inside_backpack",
        evaluation_frame=5,
    )

    prediction = extract_change_prediction(episode, observations)

    assert prediction.case_id == "keys-001"
    assert prediction.probability_changed == pytest.approx(0.65)
    assert prediction.changed_while_unobserved


def test_extract_change_prediction_rejects_missing_target(tmp_path) -> None:
    observations = tmp_path / "observations.jsonl"
    write_records(observations)
    episode = RecordedEpisode(
        episode_id="wallet-001",
        video_path="wallet-001.mp4",
        target_class="wallet",
        changed_while_unobserved=False,
        final_location_label="desk",
        evaluation_frame=5,
    )

    with pytest.raises(ValueError, match="never assigned"):
        extract_change_prediction(episode, observations)


def test_recorded_metrics_are_computed() -> None:
    predictions = (
        type("P", (), {})(),
    )
    from blackbox.experiments.schema import BinaryPrediction

    result = evaluate_recorded_predictions(
        (
            BinaryPrediction("a", 0.8, True),
            BinaryPrediction("b", 0.2, False),
        ),
        bins=2,
    )

    assert result.episode_count == 2
    assert result.brier == pytest.approx(0.04)
    assert result.nll >= 0.0
    assert result.ece >= 0.0
