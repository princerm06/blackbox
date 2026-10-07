import csv
import json
from dataclasses import asdict
from pathlib import Path

from blackbox.experiments.baselines import ChangeBaseline
from blackbox.experiments.runner import evaluate_models, predict_cases
from blackbox.experiments.schema import (
    BinaryPrediction,
    ModelEvaluation,
    UnseenEpisodeCase,
)


def calibration_bins(
    predictions: tuple[BinaryPrediction, ...],
    bins: int = 10,
) -> tuple[dict[str, float | int], ...]:
    if not predictions:
        raise ValueError("predictions must not be empty")
    if bins <= 0:
        raise ValueError("bins must be positive")

    rows: list[dict[str, float | int]] = []
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = tuple(
            prediction
            for prediction in predictions
            if (
                lower <= prediction.probability_changed < upper
                or (
                    index == bins - 1
                    and prediction.probability_changed == 1.0
                )
            )
        )
        if not members:
            continue

        rows.append(
            {
                "bin_lower": lower,
                "bin_upper": upper,
                "count": len(members),
                "mean_predicted_probability": sum(
                    member.probability_changed
                    for member in members
                )
                / len(members),
                "empirical_change_frequency": sum(
                    float(member.changed_while_unobserved)
                    for member in members
                )
                / len(members),
            }
        )

    return tuple(rows)


def markdown_summary(
    evaluations: tuple[ModelEvaluation, ...],
) -> str:
    lines = [
        "# BlackBox Calibration Experiment",
        "",
        "| Model | Cases | Brier ↓ | NLL ↓ | ECE ↓ |",
        "|---|---:|---:|---:|---:|",
    ]
    for result in evaluations:
        lines.append(
            "| "
            f"{result.model_name} | "
            f"{result.case_count} | "
            f"{result.brier:.6f} | "
            f"{result.nll:.6f} | "
            f"{result.ece:.6f} |"
        )

    lines.extend(
        [
            "",
            "> Synthetic controlled experiment. Results validate the "
            "experiment machinery and baseline behavior; they are not "
            "evidence of real-world calibration.",
            "",
        ]
    )
    return "\n".join(lines)


def write_experiment_report(
    output_dir: Path,
    models: tuple[ChangeBaseline, ...],
    cases: tuple[UnseenEpisodeCase, ...],
    bins: int = 10,
) -> tuple[ModelEvaluation, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluations = evaluate_models(models, cases)

    (output_dir / "summary.json").write_text(
        json.dumps(
            [asdict(result) for result in evaluations],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(
        markdown_summary(evaluations),
        encoding="utf-8",
    )

    with (output_dir / "predictions.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "model_name",
                "case_id",
                "probability_changed",
                "changed_while_unobserved",
            ),
        )
        writer.writeheader()
        for model in models:
            for prediction in predict_cases(model, cases):
                writer.writerow(
                    {
                        "model_name": model.name,
                        **asdict(prediction),
                    }
                )

    with (output_dir / "calibration.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "model_name",
                "bin_lower",
                "bin_upper",
                "count",
                "mean_predicted_probability",
                "empirical_change_frequency",
            ),
        )
        writer.writeheader()
        for model in models:
            rows = calibration_bins(
                predict_cases(model, cases),
                bins=bins,
            )
            for row in rows:
                writer.writerow(
                    {
                        "model_name": model.name,
                        **row,
                    }
                )

    return evaluations
