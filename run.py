"""Run a declared illustrative scenario from the command line.

Usage:
    python3 run.py configs/illustrative_baseline.json \
        --output results/illustrative_baseline.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from model import simulate


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Appendix A illustrative ODE scenario.")
    parser.add_argument("scenario", type=Path, help="JSON scenario with all parameters and inputs declared")
    parser.add_argument("--output", type=Path, required=True, help="CSV path to create")
    args = parser.parse_args()

    with args.scenario.open(encoding="utf-8") as handle:
        configuration = json.load(handle)
    rows = simulate(configuration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
