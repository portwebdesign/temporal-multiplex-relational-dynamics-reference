#!/usr/bin/env python3
"""Render the public v0.5.1 stationary-branch and cost--time figures.

The renderer reads the committed synthetic robustness records; it does not fit
data, select outcomes, or modify those records.  The resulting PDFs correspond
to the two robustness figures used by the accompanying formal article.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/directional-demand-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/directional-demand-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parent
HISTORY = ROOT / "recovery-research-results"
REGULATION = ROOT / "regulation-research-results"
DATE = datetime(2026, 11, 15, tzinfo=timezone.utc)
PDF_META = {"CreationDate": DATE, "ModDate": DATE,
            "Creator": "Reproducible research presentation"}
LABELS = {"none": "Baseline", "review": "R", "boundary": "B",
          "boundary_review": "B+R", "delay": "D",
          "delay_review": "D+R", "delay_boundary": "D+B",
          "delay_boundary_review": "D+B+R"}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open() as stream:
        return list(csv.DictReader(stream))


def render(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    branches = rows(HISTORY / "equilibrium-robustness-refined-branches.csv")
    roots = rows(HISTORY / "equilibrium-robustness-roots.csv")
    folds = rows(HISTORY / "equilibrium-robustness-folds.csv")
    rankings = rows(REGULATION / "regulation-robustness-rankings.csv")
    crossings = json.loads(
        (REGULATION / "regulation-robustness-crossings.json").read_text())
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10,
                         "axes.labelsize": 9, "legend.fontsize": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "font.family": "DejaVu Sans"})

    parameters = ["a_s", "a_e", "eta_r", "rho_p", "d_w1", "h_w2"]
    names = [r"$a_s$", r"$a_e$", r"$\eta_r$", r"$\rho_p$",
             r"$d_{w1}$", r"$h_{w2}$"]
    figure, axes = plt.subplots(
        1, 2, figsize=(6.15, 3.15), layout="constrained",
        gridspec_kw={"width_ratios": [1.18, 1]})
    example = [row for row in branches if row["parameter"] == "a_s"]
    for sweep in sorted({row["sweep"] for row in example
                         if "_upper_" in row["sweep"]}):
        trace = sorted((row for row in example if row["sweep"] == sweep),
                       key=lambda row: int(row["index"]))
        chunks: list[tuple[bool, list[tuple[float, float]]]] = []
        for left, right in zip(trace[:-1], trace[1:]):
            stable = left["stability"] == right["stability"] == "stable"
            if not chunks or chunks[-1][0] != stable:
                chunks.append((stable, [(float(left["factor"]), float(left["r"]))]))
            chunks[-1][1].append((float(right["factor"]), float(right["r"])))
        for stable, points in chunks:
            axes[0].plot(*zip(*points), color="#185f85" if stable else "#b85728",
                         ls="-" if stable else "--", lw=1.3)
    for row in roots:
        if row["parameter"] == "a_s" and abs(float(row["factor"]) - 1) < 1e-10:
            stable = row["stability"] == "stable"
            axes[0].plot(1, float(row["r"]), "o" if stable else "x",
                         color="#185f85" if stable else "#b85728", ms=4)
    axes[0].axvline(1, color=".6", lw=.6, ls=":")
    axes[0].set(xlim=(.982, 1.014), ylim=(.38, .65), xlabel=r"Multiplier of $a_s$",
                ylabel="Stationary resource r", title="A. Stable and unstable branches")
    axes[0].set_xticks([.985, 1, 1.01], ["0.985", "1.000", "1.010"])
    axes[0].plot([], [], color="#185f85", label="Stable")
    axes[0].plot([], [], color="#b85728", ls="--", label="Unstable")
    axes[0].legend(loc="lower right", frameon=False)
    for index, parameter in enumerate(parameters):
        local = sorted((row for row in folds if row["parameter"] == parameter),
                       key=lambda row: float(row["factor"]))
        if len(local) not in {1, 2} or any(
                row["status"] != "nondegenerate_local_fold_numerical" for row in local):
            raise ValueError("unexpected fold inventory")
        left = float(local[0]["factor"])
        right = float(local[-1]["factor"]) if len(local) == 2 else 1.1
        axes[1].plot([left, right], [index, index], color="#185f85", lw=3,
                     solid_capstyle="butt")
        axes[1].plot(left, index, "|", color="#185f85", ms=9)
        if len(local) == 2:
            axes[1].plot(right, index, "|", color="#185f85", ms=9)
        else:
            axes[1].annotate("", xy=(1.099, index), xytext=(1.08, index),
                             arrowprops={"arrowstyle": "->", "color": "#185f85", "lw": 1.3})
    axes[1].axvline(1, color=".5", lw=.7, ls=":")
    axes[1].set(yticks=np.arange(6), yticklabels=names, xlim=(.895, 1.105),
                xlabel="Coefficient multiplier", title="B. Tracked coexistence intervals")
    axes[1].set_xticks([.9, 1, 1.1])
    axes[1].invert_yaxis()
    for axis in axes:
        axis.grid(alpha=.14)
    figure.savefig(output / "stationary-robustness.pdf", metadata=PDF_META)
    plt.close(figure)

    grid = [row for row in rankings if row["family"] == "cost_grid"]
    costs = sorted({float(row["cost"]) for row in grid})
    times = sorted({float(row["time"]) for row in grid})
    codes = list(LABELS)
    leaders = sorted({row["resource_groups"].split(";")[0] for row in grid},
                     key=lambda label: codes.index(label))
    if any("|" in leader for leader in leaders):
        raise ValueError("resource-leader ties need an explicit visual category")
    palette = {"none": "#b8c1c5", "delay": "#bf713b",
               "delay_review": "#1b7a78", "delay_boundary_review": "#66559b"}
    if not set(leaders) <= set(palette):
        raise ValueError("unexpected leader category")
    lookup = {(float(row["cost"]), float(row["time"])): row for row in grid}
    matrix = np.array([[leaders.index(lookup[cost, time]["resource_groups"].split(";")[0])
                        for time in times] for cost in costs])
    figure, axes = plt.subplots(2, 1, figsize=(6.15, 4.45), layout="constrained",
                                gridspec_kw={"height_ratios": [1.1, 1]})
    cmap = ListedColormap([palette[name] for name in leaders])
    norm = BoundaryNorm(np.arange(len(leaders) + 1) - .5, len(leaders))
    axes[0].pcolormesh(np.r_[np.array(times) - .25, times[-1] + .25],
                       np.r_[np.array(costs) - .025, costs[-1] + .025], matrix,
                       cmap=cmap, norm=norm, shading="flat", rasterized=False,
                       edgecolors="face", linewidth=0, antialiased=False)
    axes[0].set(xlim=(10, 60), ylim=(-.025, .425), yticks=costs[::2],
                ylabel=r"Input cost $k_{\rm input}$",
                title="A. Highest resource among eight tested programs")
    axes[0].legend(handles=[Patch(facecolor=palette[name], label=LABELS[name])
                            for name in leaders], loc="lower center",
                   bbox_to_anchor=(.5, 1.14), ncol=len(leaders), frameon=False)
    for cost, colour in zip([.05, .1, .2], ["#b85728", "#185f85", "#167575"]):
        records = sorted((row for row in grid if abs(float(row["cost"]) - cost) < 1e-12),
                         key=lambda row: float(row["time"]))
        axes[1].plot([float(row["time"]) for row in records],
                     [float(row["D_R_minus_all3_resource"]) for row in records],
                     color=colour, lw=1.4, label=f"Cost {cost:g}")
    axes[1].axhline(0, color=".4", lw=.7)
    axes[1].set(xlim=(16, 36), ylim=(-.001, .005),
                xlabel="Synthetic model time (detail)",
                ylabel=r"$r_{D+R}-r_{D+B+R}$",
                title="B. Pairwise reversals, magnified interval")
    for cost, colour in zip([.05, .1], ["#b85728", "#185f85"]):
        record = next(row for row in crossings
                      if row["family"] == "cost_grid" and abs(row["cost"] - cost) < 1e-12)
        root = record["strict_crossings"][0]["time"]
        axes[1].plot(root, 0, "o", color=colour, ms=3)
        offset = (-8, 12) if cost == .05 else (8, -12)
        axes[1].annotate(f"{root:.2f}", (root, 0), xytext=offset,
                         textcoords="offset points", fontsize=8, color=colour,
                         bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none", "pad": .4},
                         arrowprops={"arrowstyle": "-", "color": colour, "lw": .6})
    axes[1].legend(loc="upper right", frameon=False, ncol=3)
    axes[1].grid(alpha=.14)
    figure.savefig(output / "regulation-robustness.pdf", metadata=PDF_META)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="Directory receiving stationary-robustness.pdf and regulation-robustness.pdf.")
    args = parser.parse_args()
    render(args.output.resolve())


if __name__ == "__main__":
    main()
