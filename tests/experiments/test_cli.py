from scripts.run_calibration_experiment import build_parser


def test_cli_defaults_are_reproducible() -> None:
    args = build_parser().parse_args([])

    assert args.cases_per_level == 250
    assert args.seed == 7
    assert args.bins == 10
    assert str(args.output_dir).endswith("runs/calibration")


def test_cli_accepts_experiment_overrides() -> None:
    args = build_parser().parse_args(
        [
            "--cases-per-level",
            "10",
            "--seed",
            "99",
            "--bins",
            "5",
            "--output-dir",
            "runs/test-calibration",
        ]
    )

    assert args.cases_per_level == 10
    assert args.seed == 99
    assert args.bins == 5
    assert str(args.output_dir) == "runs/test-calibration"
