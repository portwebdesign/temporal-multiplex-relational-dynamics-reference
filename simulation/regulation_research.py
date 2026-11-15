"""Additional regulation and boundary-support experiments; existing ODE unchanged.

Run from the repository root. Outputs are separate from the approved research
record. Context controls retain each intervention program and remove only the
elevated demand; raw intervention effects have separate comparator columns.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import hashlib
import itertools
import json
from pathlib import Path

from model import STATE_NAMES, simulate
from network import aggregate_network
from referee_experiments import Experiment, raw_scenarios

ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = ROOT / "configs/regulation_research_protocol.json"
DEFAULT_OUTPUT = ROOT / "regulation-research-results"


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, record):
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def factorial_subsets(names):
    return [tuple(name for name, selected in zip(names, bits) if selected)
            for bits in itertools.product((False, True), repeat=len(names))]


def context_configuration(configuration):
    """Remove elevated demand while retaining support, input and target programs."""
    cfg = deepcopy(configuration)
    demands = deepcopy(cfg["network"]["baseline"]["demand_edges"])
    for segment in cfg["network"]["schedule"]:
        segment["network"]["demand_edges"] = deepcopy(demands)
    return cfg


def subset_label(subset):
    return "_".join(subset) if subset else "none"


def construct_conditions(protocol):
    source = read_json(ROOT / protocol["source_protocol"])
    raw = raw_scenarios(source)["opposed_tradeoffs"]
    experiment = Experiment(source)
    conditions = []
    references = {}
    starts = {}
    start, end = protocol["interval"]
    for cost in protocol["input_costs"]:
        seed_cfg = deepcopy(raw)
        seed_cfg["parameters"].update(protocol["counter_parameters"])
        seed_cfg["parameters"]["k_input"] = cost
        prepared = experiment.initialize(seed_cfg)
        starts[cost] = prepared
        references[str(cost)] = list(prepared["initial_state"])
        for subset in factorial_subsets(protocol["input_order"]):
            cfg = deepcopy(prepared)
            cfg["solver"]["time_horizon"] = protocol["horizon"]
            cfg["inputs"]["schedule"] = []
            if subset:
                cfg["inputs"]["schedule"] = [{"start": start, "end": end,
                    "values": {name: protocol["strong_inputs"][name] for name in subset}}]
            name = f"input_cost_{cost:g}_{subset_label(subset)}"
            cfg["metadata"] = {"name": name, "experiment": "input_factorial"}
            conditions.append({"name": name, "family": "input_factorial",
                               "cost": cost, "subset": subset_label(subset),
                               "configuration": cfg})
    boundary = protocol["boundary_support"]
    seed_cfg = starts[boundary["input_cost"]]
    if (seed_cfg["network"]["baseline"]["support_edges"][0]["support"]
            != boundary["baseline_support_mark"]):
        raise ValueError("declared common support baseline differs from source")
    for target, support_mark in itertools.product(boundary["targets"], boundary["support_marks"]):
        cfg = deepcopy(seed_cfg)
        cfg["solver"]["time_horizon"] = protocol["horizon"]
        cfg["inputs"]["schedule"] = []
        cfg["permeability_target"]["schedule"] = [{"start": start, "end": end, "value": target}]
        cfg["network"]["schedule"][0]["network"]["support_edges"][0]["support"] = support_mark
        name = f"boundary_{target:g}_support_{support_mark:g}"
        cfg["metadata"] = {"name": name, "experiment": "boundary_support"}
        conditions.append({"name": name, "family": "boundary_support",
                           "cost": boundary["input_cost"], "target": target,
                           "support_mark": support_mark, "configuration": cfg})
    for condition in conditions:
        condition["context"] = context_configuration(condition["configuration"])
    return conditions, references, experiment.baselines


def point(rows, time):
    step = rows[1]["time"] - rows[0]["time"]
    index = round(time / step)
    if index >= len(rows) or abs(rows[index]["time"] - time) > 1e-10:
        raise ValueError("outcome time absent from grid")
    return rows[index]


def refine(configuration, steps):
    traces = []
    for step in steps:
        cfg = deepcopy(configuration)
        cfg["solver"]["dt"] = step
        traces.append(simulate(cfg))
    if len(steps) != 3 or abs(steps[0] / steps[1] - 2) > 1e-12 or abs(steps[1] / steps[2] - 2) > 1e-12:
        raise ValueError("three successive halved grids are required")
    coarse, medium, fine = traces
    medium_error = max(abs(a[name] - b[name]) for a, b in zip(coarse, medium[::2]) for name in STATE_NAMES)
    fine_error = max(abs(a[name] - b[name]) for a, b in zip(medium, fine[::2]) for name in STATE_NAMES)
    return fine, {"coarse_medium_max_state_error": medium_error,
                  "medium_fine_max_state_error": fine_error,
                  "coarse_rows": len(coarse), "medium_rows": len(medium), "fine_rows": len(fine)}


def run(protocol_path, output):
    protocol_path = Path(protocol_path).resolve()
    output = Path(output).resolve()
    protected = [ROOT / name for name in ("revision-results", "dynamics-results", "referee-results", "presentation-results")]
    if any(output == path.resolve() or path.resolve() in output.parents for path in protected):
        raise ValueError("historical outputs are protected")
    protocol = read_json(protocol_path)
    conditions, references, baselines = construct_conditions(protocol)
    output.mkdir(parents=True, exist_ok=True)
    summary, refinements = [], []
    cache = {}
    for index, condition in enumerate(conditions, 1):
        outcomes = {}
        for arm in ("configuration", "context"):
            cfg = condition[arm]
            key = json.dumps({k: v for k, v in cfg.items() if k != "metadata"}, sort_keys=True)
            if key not in cache:
                cache[key] = refine(cfg, protocol["steps"])
            rows, errors = cache[key]
            outcomes[arm] = rows
            refinements.append({"name": condition["name"], "arm": arm, **errors})
            write_json(output / (condition["name"] + ("_context" if arm == "context" else "") + ".json"), cfg)
            write_csv(output / (condition["name"] + ("_context" if arm == "context" else "") + ".csv"), rows[::4])
        cfg = condition["configuration"]
        baseline_signals = aggregate_network(cfg["network"]["baseline"])
        pulse_signals = aggregate_network(cfg["network"]["schedule"][0]["network"])
        row = {key: condition[key] for key in ("name", "family", "cost", "subset", "target", "support_mark") if key in condition}
        row.update({"initial_r": cfg["initial_state"][2], "baseline_support": baseline_signals.support,
                    "pulse_support": pulse_signals.support, "pulse_external_demand": pulse_signals.external_demand,
                    "pulse_conflict": pulse_signals.conflict})
        for time in (10, 20):
            pulse_row, context_row = point(outcomes["configuration"], time), point(outcomes["context"], time)
            for name in ("r", "w1", "w2"):
                row[f"{name}_{time}"] = pulse_row[name]
                row[f"context_{name}_{time}"] = context_row[name]
                row[f"demand_delta_{name}_{time}"] = pulse_row[name] - context_row[name]
        summary.append(row)
        print(f"Completed regulation condition {index}/{len(conditions)}: {condition['name']}", flush=True)
    for row in summary:
        if row["family"] == "input_factorial":
            comparator = next(other for other in summary if other["family"] == row["family"]
                              and other["cost"] == row["cost"] and other["subset"] == "none")
            labels = [("input_program", comparator)]
        else:
            no_target = next(other for other in summary if other["family"] == row["family"]
                             and other["support_mark"] == row["support_mark"] and other["target"] == 0.45)
            no_support = next(other for other in summary if other["family"] == row["family"]
                              and other["target"] == row["target"] and other["support_mark"] == 1.0)
            labels = [("target_program", no_target), ("support_program", no_support)]
        for label, comparator in labels:
            row[f"{label}_comparator"] = comparator["name"]
            for time in (10, 20):
                for name in ("r", "w1", "w2"):
                    row[f"{label}_effect_{name}_{time}"] = row[f"{name}_{time}"] - comparator[f"{name}_{time}"]
                    row[f"{label}_demand_contrast_effect_{name}_{time}"] = row[f"demand_delta_{name}_{time}"] - comparator[f"demand_delta_{name}_{time}"]
    write_csv(output / "summary.csv", summary)
    write_csv(output / "refinement.csv", refinements)
    write_json(output / "stationary_references.json", {"states": references, "records": baselines})
    max_error = max(row["medium_fine_max_state_error"] for row in refinements)
    report = {
        "version": protocol["version"], "condition_pairs": len(conditions), "recorded_arms": len(refinements),
        "unique_numerical_arms": len(cache), "protocol_sha256": sha(protocol_path),
        "sources": {str(path.relative_to(ROOT)): sha(path) for path in [protocol_path, Path(__file__).resolve(), ROOT / protocol["source_protocol"], ROOT / "model.py", ROOT / "network.py", ROOT / "regimes.py", ROOT / "analysis.py", ROOT / "referee_experiments.py"]},
        "steps": protocol["steps"], "maximum_medium_fine_state_difference": max_error,
        "numerical_tolerance": protocol["numerical_tolerance"],
        "refinement_pass": max_error <= protocol["numerical_tolerance"],
        "stationary_maximum_residual": max(record["residual"] for record in baselines.values()),
        "meaning": "Synthetic intervention comparisons under the unchanged ODE; not a human efficacy test. All outcomes, including opposite effects, are retained."
    }
    write_json(output / "report.json", report)
    if not report["refinement_pass"]:
        raise RuntimeError("declared refinement tolerance exceeded; all outcomes retained")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.protocol, args.output), indent=2))


if __name__ == "__main__":
    main()
