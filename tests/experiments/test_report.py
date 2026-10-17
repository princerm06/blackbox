import pytest
import csv
import json

from blackbox.experiments.baselines import (
    ConstantPriorBaseline,
    HeuristicModelBaseline,
)
from blackbox.experiments.generator import (
    GeneratorConfig,
    generate_controlled_cases,
)
from blackbox.experiments.report import (
    calibration_bins,
    markdown_summary,
    write_experiment_report,
)
from blackbox.experiments.runner import evaluate_models
from blackbox.experiments.schema import BinaryPrediction


def test_calibration_bins_summarize_prediction_frequency() -> None:
    rows = calibration_bins(
        (
            BinaryPrediction("a", 0.1, False),
            BinaryPrediction("b", 0.2, True),
            BinaryPrediction("c", 0.9, True),
        ),
        bins=2,
    )

    assert len(rows) == 2
    assert rows[0]["count"] == 2
    assert rows[0]["mean_predicted_probability"] == pytest.approx(0.15)
    assert rows[0]["empirical_change_frequency"] == 0.5
    assert rows[1]["count"] == 1


def test_markdown_summary_contains_model_metrics() -> None:
    cases = generate_controlled_cases(
        GeneratorConfig(cases_per_level=2, seed=3)
    )
    evaluations = evaluate_models(
        (ConstantPriorBaseline(0.2),),
        cases,
    )

    report = markdown_summary(evaluations)

    assert "constant_prior" in report
    assert "Brier" in report
    assert "Synthetic controlled experiment" in report


def test_write_experiment_report_creates_research_artifacts(
    tmp_path,
) -> None:
    cases = generate_controlled_cases(
        GeneratorConfig(cases_per_level=3, seed=5)
    )
    models = (
        ConstantPriorBaseline(0.2),
        HeuristicModelBaseline(),
    )

    evaluations = write_experiment_report(
        tmp_path,
        models,
        cases,
        bins=5,
    )

    assert len(evaluations) == 2
    expected = {
        "summary.json",
        "summary.md",
        "predictions.csv",
        "calibration.csv",
        "metric_comparison.svg",
        "reliability.svg",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected

    summary = json.loads(
        (tmp_path / "summary.json").read_text(encoding="utf-8")
    )
    assert [row["model_name"] for row in summary] == [
        "constant_prior",
        "heuristic_change_model",
    ]

    with (tmp_path / "predictions.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        prediction_rows = list(csv.DictReader(handle))
    assert len(prediction_rows) == len(cases) * len(models)

    with (tmp_path / "calibration.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        calibration_rows = list(csv.DictReader(handle))
    assert calibration_rows


def test_report_figures_are_svg(tmp_path) -> None:
    cases = generate_controlled_cases(
        GeneratorConfig(cases_per_level=2, seed=13)
    )
    write_experiment_report(
        tmp_path,
        (ConstantPriorBaseline(0.2), HeuristicModelBaseline()),
        cases,
        bins=4,
    )

    metric_svg = (tmp_path / "metric_comparison.svg").read_text(
        encoding="utf-8"
    )
    reliability_svg = (tmp_path / "reliability.svg").read_text(
        encoding="utf-8"
    )

    assert metric_svg.startswith("<svg")
    assert "BlackBox baseline metric comparison" in metric_svg
    assert reliability_svg.startswith("<svg")
    assert "Reliability diagram" in reliability_svg
