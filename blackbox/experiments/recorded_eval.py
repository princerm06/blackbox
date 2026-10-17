import json
from dataclasses import dataclass
from pathlib import Path

from blackbox.experiments.metrics import (
    brier_score,
    expected_calibration_error,
    negative_log_likelihood,
)
from blackbox.experiments.recorded import RecordedEpisode
from blackbox.experiments.schema import BinaryPrediction


@dataclass(frozen=True)
class RecordedEvaluation:
    episode_count: int
    brier: float
    nll: float
    ece: float


def _records_until_frame(
    observations_path: Path,
    evaluation_frame: int | None,
) -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    with observations_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            frame_index = int(record["frame_index"])
            if (
                evaluation_frame is not None
                and frame_index > evaluation_frame
            ):
                break
            records.append(record)

    if not records:
        raise ValueError(
            f"no observation records found in {observations_path}"
        )
    return tuple(records)


def _target_object_id(
    records: tuple[dict[str, object], ...],
    target_class: str,
) -> str:
    counts: dict[str, int] = {}

    for record in records:
        identities = record.get("identities", [])
        if not isinstance(identities, list):
            continue
        for identity in identities:
            if not isinstance(identity, dict):
                continue
            if identity.get("class_name") != target_class:
                continue
            object_id = str(identity["object_id"])
            counts[object_id] = counts.get(object_id, 0) + 1

    if not counts:
        raise ValueError(
            f"target class {target_class!r} was never assigned an object id"
        )

    ranked = sorted(
        counts.items(),
        key=lambda item: (-item[1], item[0]),
    )
    return ranked[0][0]


def extract_change_prediction(
    episode: RecordedEpisode,
    observations_path: Path,
) -> BinaryPrediction:
    records = _records_until_frame(
        observations_path,
        episode.evaluation_frame,
    )
    object_id = _target_object_id(records, episode.target_class)
    record = records[-1]

    uncertainty = record.get("uncertainty", {})
    if not isinstance(uncertainty, dict):
        raise ValueError("uncertainty payload is not a mapping")

    distributions = uncertainty.get(object_id, [])
    if not isinstance(distributions, list):
        raise ValueError("object uncertainty payload is not a list")

    for distribution in distributions:
        if not isinstance(distribution, dict):
            continue
        if (
            distribution.get("predicate")
            != "UNOBSERVED_CHANGE_STATE"
        ):
            continue

        hypotheses = distribution.get("hypotheses", [])
        if not isinstance(hypotheses, list):
            continue
        for hypothesis in hypotheses:
            if not isinstance(hypothesis, dict):
                continue
            if (
                hypothesis.get("value")
                == "changed_while_unobserved"
            ):
                return BinaryPrediction(
                    case_id=episode.episode_id,
                    probability_changed=float(
                        hypothesis["weight"]
                    ),
                    changed_while_unobserved=(
                        episode.changed_while_unobserved
                    ),
                )

    raise ValueError(
        "no UNOBSERVED_CHANGE_STATE prediction found for "
        f"{episode.episode_id} at evaluation frame"
    )


def evaluate_recorded_predictions(
    predictions: tuple[BinaryPrediction, ...],
    bins: int = 10,
) -> RecordedEvaluation:
    if not predictions:
        raise ValueError("predictions must not be empty")

    return RecordedEvaluation(
        episode_count=len(predictions),
        brier=brier_score(predictions),
        nll=negative_log_likelihood(predictions),
        ece=expected_calibration_error(predictions, bins=bins),
    )
