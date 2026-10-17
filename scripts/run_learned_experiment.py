import argparse
import json
from pathlib import Path

from blackbox.experiments.baselines import (
    ConstantPriorBaseline,
    DeterministicOpportunityBaseline,
    HeuristicModelBaseline,
    LastObservationBaseline,
)
from blackbox.experiments.generator import (
    GeneratorConfig,
    generate_controlled_cases,
)
from blackbox.experiments.report import write_experiment_report
from blackbox.experiments.split import split_cases
from blackbox.experiments.training import select_logistic_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate BlackBox's first learned unseen-change model."
        )
    )
    parser.add_argument("--cases-per-level", type=int, default=500)
    parser.add_argument("--data-seed", type=int, default=7)
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs") / "learned-calibration",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cases = generate_controlled_cases(
        GeneratorConfig(
            cases_per_level=args.cases_per_level,
            seed=args.data_seed,
        )
    )
    train, validation, test = split_cases(
        cases,
        seed=args.split_seed,
    )

    selection = select_logistic_model(train, validation)
    models = (
        LastObservationBaseline(),
        ConstantPriorBaseline(),
        DeterministicOpportunityBaseline(),
        HeuristicModelBaseline(),
        selection.model,
    )
    evaluations = write_experiment_report(
        args.output_dir,
        models,
        test,
        bins=args.bins,
    )

    metadata = {
        "data_seed": args.data_seed,
        "split_seed": args.split_seed,
        "train_cases": len(train),
        "validation_cases": len(validation),
        "test_cases": len(test),
        "selected_learning_rate": selection.learning_rate,
        "selected_l2": selection.l2,
        "validation_nll": selection.validation_nll,
    }
    (args.output_dir / "training.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"train={len(train)} validation={len(validation)} "
        f"test={len(test)}"
    )
    print(
        "selected logistic model: "
        f"lr={selection.learning_rate}, "
        f"l2={selection.l2}, "
        f"validation_nll={selection.validation_nll:.6f}"
    )
    print()
    print(f"{'model':30} {'brier':>10} {'nll':>10} {'ece':>10}")
    for result in evaluations:
        print(
            f"{result.model_name:30} "
            f"{result.brier:10.6f} "
            f"{result.nll:10.6f} "
            f"{result.ece:10.6f}"
        )


if __name__ == "__main__":
    main()
