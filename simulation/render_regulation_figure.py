#!/usr/bin/env python3
"""Render the recovery article's regulation comparison from recorded outputs.

This renderer deliberately consumes the checked `summary.csv` created by
`regulation_research.py`; it does not recompute trajectories or use human data.
It writes one PDF intended for visual reproduction of the article's Figure 5.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tmrd-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/tmrd-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = ROOT / "regulation-research-results"
LABELS = {
    "review": "R",
    "boundary": "B",
    "boundary_review": "B+R",
    "delay": "D",
    "delay_review": "D+R",
    "delay_boundary": "D+B",
    "delay_boundary_review": "D+B+R",
}
PDF_METADATA = {
    "CreationDate": datetime(2026, 11, 15, tzinfo=timezone.utc),
    "ModDate": datetime(2026, 11, 15, tzinfo=timezone.utc),
    "Creator": "Reproducible research presentation",
}


def load_costly_programs(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    programs = [
        row for row in rows
        if row["family"] == "input_factorial" and float(row["cost"]) == 0.2
        and row["subset"] != "none"
    ]
    expected = set(LABELS)
    observed = {row["subset"] for row in programs}
    if observed != expected or len(programs) != len(expected):
        raise ValueError("summary.csv does not contain exactly the costly seven-program factorial")
    return sorted(programs, key=lambda row: list(LABELS).index(row["subset"]))


def render(results: Path, output: Path) -> None:
    programs = load_costly_programs(results / "summary.csv")
    labels = [LABELS[row["subset"]] for row in programs]
    positions = list(range(len(programs)))
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "font.family": "DejaVu Sans",
    })
    figure, axes = plt.subplots(1, 2, figsize=(6.15, 3.8), layout="constrained")
    axes[0].barh(
        [value - 0.17 for value in positions],
        [float(row["input_program_effect_r_10"]) for row in programs],
        height=0.32, color="#b85728", label="t = 10",
    )
    axes[0].barh(
        [value + 0.17 for value in positions],
        [float(row["input_program_effect_r_20"]) for row in programs],
        height=0.32, color="#167575", label="t = 20",
    )
    axes[0].set_yticks(positions, labels)
    axes[0].invert_yaxis()
    axes[0].axvline(0, color=".5", linewidth=0.6)
    axes[0].set_xlabel("Resource difference")
    axes[0].set_xticks([-0.08, -0.04, 0, 0.04])
    axes[0].set_title("A. Evaluation time")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    axes[1].barh(
        positions,
        [float(row["input_program_effect_w2_10"]) for row in programs],
        color="#185f85", height=0.6,
    )
    axes[1].set_yticks(positions, labels)
    axes[1].invert_yaxis()
    axes[1].axvline(0, color=".5", linewidth=0.6)
    axes[1].set_xlabel("Load difference at t = 10")
    axes[1].set_xticks([-0.24, -0.16, -0.08, 0])
    axes[1].set_title("B. Experienced load")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, metadata=PDF_METADATA)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=ROOT / "regulation-programs.pdf")
    args = parser.parse_args()
    render(args.results.resolve(), args.output.resolve())
    print(f"Rendered {args.output.resolve()}")


if __name__ == "__main__":
    main()
