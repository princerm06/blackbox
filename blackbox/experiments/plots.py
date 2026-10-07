from pathlib import Path

from blackbox.experiments.schema import ModelEvaluation


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def write_metric_comparison_svg(
    path: Path,
    evaluations: tuple[ModelEvaluation, ...],
) -> None:
    if not evaluations:
        raise ValueError("evaluations must not be empty")

    width = 900
    height = 520
    left = 210
    right = 40
    top = 70
    bottom = 70
    chart_width = width - left - right
    chart_height = height - top - bottom

    max_value = max(
        max(result.brier, result.nll, result.ece)
        for result in evaluations
    )
    max_value = max(max_value, 1e-9)

    group_height = chart_height / len(evaluations)
    bar_height = min(24.0, group_height / 4.0)
    metric_offsets = (-bar_height, 0.0, bar_height)
    metrics = (
        ("Brier", "brier"),
        ("NLL", "nll"),
        ("ECE", "ece"),
    )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="30" y="38" font-size="24" font-family="sans-serif">'
        'BlackBox baseline metric comparison</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" '
        f'y2="{height-bottom}" stroke="black"/>',
        f'<line x1="{left}" y1="{height-bottom}" '
        f'x2="{width-right}" y2="{height-bottom}" stroke="black"/>',
    ]

    for index, result in enumerate(evaluations):
        center_y = top + group_height * (index + 0.5)
        parts.append(
            f'<text x="{left-12}" y="{center_y+5:.1f}" '
            'text-anchor="end" font-size="13" font-family="sans-serif">'
            f'{_escape(result.model_name)}</text>'
        )

        for metric_index, (label, field_name) in enumerate(metrics):
            value = getattr(result, field_name)
            bar_width = chart_width * value / max_value
            y = center_y + metric_offsets[metric_index] - bar_height / 2
            opacity = (0.35, 0.6, 0.85)[metric_index]
            parts.append(
                f'<rect x="{left}" y="{y:.1f}" width="{bar_width:.1f}" '
                f'height="{bar_height:.1f}" fill="black" '
                f'opacity="{opacity}"/>'
            )
            parts.append(
                f'<text x="{left+bar_width+6:.1f}" y="{y+bar_height-5:.1f}" '
                'font-size="11" font-family="sans-serif">'
                f'{label} {value:.3f}</text>'
            )

    parts.append(
        f'<text x="{left + chart_width/2:.1f}" y="{height-22}" '
        'text-anchor="middle" font-size="12" font-family="sans-serif">'
        'Lower is better; synthetic controlled experiment</text>'
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_reliability_svg(
    path: Path,
    series: tuple[
        tuple[str, tuple[dict[str, float | int], ...]],
        ...,
    ],
) -> None:
    if not series:
        raise ValueError("series must not be empty")

    width = 700
    height = 650
    left = 80
    right = 40
    top = 70
    bottom = 80
    size = min(width - left - right, height - top - bottom)
    x0 = left
    y0 = top + size

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="30" y="38" font-size="24" font-family="sans-serif">'
        'Reliability diagram</text>',
        f'<line x1="{x0}" y1="{y0}" x2="{x0+size}" y2="{top}" '
        'stroke="gray" stroke-dasharray="6 6"/>',
        f'<line x1="{x0}" y1="{top}" x2="{x0}" y2="{y0}" stroke="black"/>',
        f'<line x1="{x0}" y1="{y0}" x2="{x0+size}" y2="{y0}" stroke="black"/>',
    ]

    for tick in range(6):
        value = tick / 5
        x = x0 + value * size
        y = y0 - value * size
        parts.extend(
            [
                f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" '
                f'y2="{y0+6}" stroke="black"/>',
                f'<text x="{x:.1f}" y="{y0+24}" text-anchor="middle" '
                'font-size="11" font-family="sans-serif">'
                f'{value:.1f}</text>',
                f'<line x1="{x0-6}" y1="{y:.1f}" x2="{x0}" '
                f'y2="{y:.1f}" stroke="black"/>',
                f'<text x="{x0-12}" y="{y+4:.1f}" text-anchor="end" '
                'font-size="11" font-family="sans-serif">'
                f'{value:.1f}</text>',
            ]
        )

    for series_index, (model_name, rows) in enumerate(series):
        points: list[str] = []
        opacity = min(1.0, 0.35 + 0.18 * series_index)
        for row in rows:
            predicted = float(row["mean_predicted_probability"])
            empirical = float(row["empirical_change_frequency"])
            x = x0 + predicted * size
            y = y0 - empirical * size
            points.append(f"{x:.1f},{y:.1f}")
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" '
                f'fill="black" opacity="{opacity}"/>'
            )
        if len(points) >= 2:
            parts.append(
                f'<polyline points="{" ".join(points)}" '
                f'fill="none" stroke="black" stroke-width="2" '
                f'opacity="{opacity}"/>'
            )

        legend_y = top + 20 + series_index * 20
        parts.append(
            f'<text x="{x0+size-5}" y="{legend_y}" text-anchor="end" '
            f'font-size="12" font-family="sans-serif" '
            f'opacity="{opacity}">{_escape(model_name)}</text>'
        )

    parts.extend(
        [
            f'<text x="{x0+size/2:.1f}" y="{height-28}" '
            'text-anchor="middle" font-size="13" font-family="sans-serif">'
            'Mean predicted probability of change</text>',
            f'<text x="22" y="{top+size/2:.1f}" '
            'text-anchor="middle" font-size="13" font-family="sans-serif" '
            f'transform="rotate(-90 22 {top+size/2:.1f})">'
            'Empirical change frequency</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
