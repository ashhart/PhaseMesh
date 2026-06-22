from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create the one-page adaptive phase-field report.")
    parser.add_argument("--metrics", type=Path, default=Path("benchmark/frontier_comparison/out/frontier_metrics.json"))
    parser.add_argument("--out", type=Path, default=Path("benchmark/frontier_comparison/adaptive_phase_field_reasoning.pdf"))
    args = parser.parse_args(argv)
    payload = json.loads(args.metrics.read_text(encoding="utf-8"))
    build_pdf(payload, args.out)
    print(args.out)
    return 0


def build_pdf(payload: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        rightMargin=14 * mm,
        leftMargin=14 * mm,
        topMargin=12 * mm,
        bottomMargin=10 * mm,
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=18,
        leading=21,
        textColor=colors.HexColor("#16212a"),
        spaceAfter=5,
    )
    subtitle = ParagraphStyle(
        "Subtitle",
        parent=styles["BodyText"],
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#46535c"),
        spaceAfter=8,
    )
    small = ParagraphStyle(
        "Small",
        parent=styles["BodyText"],
        fontSize=7.6,
        leading=9.2,
        textColor=colors.HexColor("#2b343a"),
    )

    adaptive = payload["adaptive_compute"]
    context = payload["context_retention"]
    footprint = payload["footprint"]
    easy = adaptive["easy"]
    hard = adaptive["hard"]

    story: list[Any] = [
        Paragraph("Adaptive Phase-Field Reasoning", title),
        Paragraph(
            "Pinned phase-field run with residual carry. This page reports local prototype measurements and transparent estimates; dense LLM values are reference anchors, not local model runs.",
            subtitle,
        ),
    ]

    kpi_data = [
        ["Metric", "Measured phase mesh result", "Why it matters"],
        [
            "Adaptive compute",
            f"{adaptive['hard_to_easy_steps']:.1f}x hard/easy steps ({easy['steps_used']} -> {hard['steps_used']})",
            "Budget rises with prompt difficulty.",
        ],
        [
            "Prediction proxy",
            f"{easy['prediction_accuracy_proxy']:.3f} easy / {hard['prediction_accuracy_proxy']:.3f} hard",
            "No proxy collapse under harder budget.",
        ],
        [
            "Context retention",
            f"gradient {context['gradient']:.4f} under pin {payload['config']['pin_strength']}",
            "Salient-token phase structure stays pinned.",
        ],
        [
            "Footprint",
            f"Q8 state {format_bytes(footprint['state_q8_bytes'])}",
            "Flat local topology state in this run.",
        ],
    ]
    story.append(styled_table(kpi_data, [33 * mm, 72 * mm, 63 * mm]))
    story.append(Spacer(1, 5 * mm))

    plot_table_data = [[
        image_or_note(payload["plots"]["accuracy_vs_compute"], 82 * mm, 48 * mm),
        image_or_note(payload["plots"]["ram_vs_context"], 82 * mm, 48 * mm),
    ]]
    story.append(Table(plot_table_data, colWidths=[86 * mm, 86 * mm]))
    story.append(Spacer(1, 3 * mm))

    comparison_rows = [["Metric", "Phase mesh", "Reference anchor"]]
    for row in payload["comparison_table"]:
        comparison_rows.append([row["metric"], row["phase_mesh"], row["reference_anchor"]])
    story.append(styled_table(comparison_rows, [38 * mm, 65 * mm, 65 * mm], font_size=7.0))
    story.append(Spacer(1, 3 * mm))
    story.append(
        Paragraph(
            "Boundary: prediction_accuracy_proxy is 1 - mean_prediction_error, not benchmark answer accuracy. Next proof step is external answer-scored GSM8K/LongBench evaluation with the same steps/RAM/FLOPs logging.",
            small,
        )
    )
    doc.build(story)


def styled_table(data: list[list[Any]], widths: list[float], font_size: float = 7.4) -> Table:
    table = Table(data, colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe8ec")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#17232a")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("LEADING", (0, 0), (-1, -1), font_size + 2),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#9fb0b8")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f8f9")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def image_or_note(path: str, width: float, height: float) -> Image | Paragraph:
    file_path = Path(path)
    if file_path.exists():
        return Image(str(file_path), width=width, height=height)
    styles = getSampleStyleSheet()
    return Paragraph(f"Missing plot: {path}", styles["BodyText"])


def format_bytes(value: int | float) -> str:
    value = float(value)
    units = ["B", "KB", "MB", "GB", "TB"]
    index = 0
    while value >= 1000 and index < len(units) - 1:
        value /= 1000
        index += 1
    if index == 0:
        return f"{value:.0f} {units[index]}"
    return f"{value:.2f} {units[index]}"


if __name__ == "__main__":
    raise SystemExit(main())

