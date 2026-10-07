import argparse
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run BlackBox's controlled unseen-change calibration experiment."
        )
    )
    parser.add_argument(
        "--cases-per-level",
        type=int,
        default=250,
        help="Synthetic episodes per opportunity level (default: 250).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for reproducibility (default: 7).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=10,
        help="Calibration bins (default: 10).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs") / "calibration",
        help="Directory for experiment artifacts.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cases = generate_controlled_cases(
        GeneratorConfig(
            cases_per_level=args.cases_per_level,
            seed=args.seed,
        )
    )
    models = (
        LastObservationBaseline(),
        ConstantPriorBaseline(),
        DeterministicOpportunityBaseline(),
        HeuristicModelBaseline(),
    )
    results = write_experiment_report(
        output_dir=args.output_dir,
        models=models,
        cases=cases,
        bins=args.bins,
    )

    print(
        f"Generated {len(cases)} synthetic unseen episodes "
        f"with seed {args.seed}."
    )
    print()
    print(
        f"{'model':30} {'brier':>10} {'nll':>10} {'ece':>10}"
    )
    for result in results:
        print(
            f"{result.model_name:30} "
            f"{result.brier:10.6f} "
            f"{result.nll:10.6f} "
            f"{result.ece:10.6f}"
        )
    print()
    print(f"Report written to: {args.output_dir}")


if __name__ == "__main__":
    main()
