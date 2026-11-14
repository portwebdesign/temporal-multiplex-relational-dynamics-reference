"""Run a transparent one-factor illustrative sensitivity sweep.

This utility reports how a declared numerical scenario changes under a declared
range of one coefficient.  It is a mechanism check, not an empirical analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path

from model import simulate, validate_parameters


def parse_values(raw: str) -> list[float]:
    values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if len(values) < 2:
        raise ValueError("provide at least two comma-separated parameter values")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an illustrative one-factor sensitivity sweep.")
    parser.add_argument("scenario", type=Path, help="base JSON scenario")
    parser.add_argument("--parameter", required=True, help="Appendix A coefficient to vary")
    parser.add_argument("--values", required=True, help="at least two comma-separated coefficient values")
    parser.add_argument("--output", type=Path, required=True, help="CSV summary path to create")
    args = parser.parse_args()

    with args.scenario.open(encoding="utf-8") as handle:
        base = json.load(handle)
    if args.parameter not in base.get("parameters", {}):
        raise ValueError(f"unknown scenario parameter {args.parameter!r}")

    rows: list[dict[str, float]] = []
    for value in parse_values(args.values):
        scenario = deepcopy(base)
        scenario["parameters"][args.parameter] = value
        # Validate before integration, making the declared coefficient domain
        # part of the sensitivity contract rather than an after-the-fact note.
        validate_parameters(scenario["parameters"])
        trace = simulate(scenario)
        final = trace[-1]
        rows.append({args.parameter: value, **final})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} sensitivity rows to {args.output}")


if __name__ == "__main__":
    main()
