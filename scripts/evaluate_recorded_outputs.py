import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from blackbox.experiments.recorded import load_episode_manifest
from blackbox.experiments.recorded_eval import (
    evaluate_recorded_predictions,
    extract_change_prediction,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate BlackBox predictions from processed real-world episodes."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs") / "recorded-evaluation.json",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    episodes = load_episode_manifest(args.manifest)

    predictions = []
    for episode in episodes:
        observations_path = (
            args.runs_dir
            / episode.episode_id
            / "observations.jsonl"
        )
        predictions.append(
            extract_change_prediction(
                episode,
                observations_path,
            )
        )

    result = evaluate_recorded_predictions(
        tuple(predictions),
        bins=args.bins,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(asdict(result), indent=2) + "\n",
        encoding="utf-8",
    )

    predictions_path = args.output.with_suffix(".csv")
    with predictions_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "case_id",
                "probability_changed",
                "changed_while_unobserved",
            ),
        )
        writer.writeheader()
        for prediction in predictions:
            writer.writerow(asdict(prediction))

    print(f"episodes={result.episode_count}")
    print(f"brier={result.brier:.6f}")
    print(f"nll={result.nll:.6f}")
    print(f"ece={result.ece:.6f}")
    print(f"summary={args.output}")
    print(f"predictions={predictions_path}")


if __name__ == "__main__":
    main()
