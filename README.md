# Temporal--Multiplex Relational Dynamics: Reference Implementation

This repository is the MIT-licensed reference implementation for the
synthetic, source-preserving mechanism reported in:

> Yalçın, M., & Karslı, O. (2026). *A Transparent Temporal--Multiplex
> Framework for Relational Dynamics, Cross-Domain Spillover, and
> Self-Regulation*.

Release **v0.3.0** implements the stationary-baseline dynamics used by the
current article. It supersedes the historical v0.1.1 implementation, which
remains available through its Git tag but does not reproduce this model.

## Scope

The model keeps each source demand's action direction while direct and
cross-layer transmission scale its magnitude. It computes external demand,
source-preserving directional conflict, support, and a bounded seven-coordinate
ODE. The included experiment uses declared, synthetic and uncalibrated
scenarios; it is not fitted to human data and does not supply clinical,
diagnostic, employment, insurance, legal, or person-level predictions.

The repository contains only the implementation, configurations, tests, and
recorded synthetic outputs needed to reproduce the v0.3.0 article results. It
does not include the authors' private notes, reviewer material, or manuscripts.

## Requirements

Python 3.9 or later is required. The numerical model uses only the Python
standard library. Rebuilding the two vector figures also requires Tectonic and
Poppler (`pdftoppm`).

## Reproduce the declared experiment

From the repository root:

```sh
python3 -m unittest discover -s simulation/tests -v
python3 simulation/revision_experiments.py \
  --protocol simulation/configs/dynamics_protocol.json \
  --output /tmp/temporal-multiplex-results --render
```

The full run evaluates stationary baselines, pulse/control comparisons,
sensitivity conditions, continuous cycles, and numerical step refinements. It
writes a self-contained result directory under the path passed to `--output`.
The committed `simulation/dynamics-results/` directory is the v0.3.0 result
snapshot used for the release.

For a single declared scenario or a simple one-factor sweep:

```sh
python3 simulation/run.py simulation/configs/illustrative_baseline.json \
  --output /tmp/illustrative_baseline.csv
python3 simulation/sensitivity.py simulation/configs/illustrative_baseline.json \
  --parameter a1 --values 0.10,0.20,0.30,0.40 \
  --output /tmp/a1_terminal_sensitivity.csv
```

## Repository layout

```text
simulation/
  analysis.py                    # Stationary states, deviations, and cycle statistics
  model.py                       # Bounded gain--loss ODE and RK4 solver
  network.py                     # Source-preserving temporal-multiplex aggregation
  revision_experiments.py        # Full v0.3.0 experiment and figure generator
  configs/dynamics_protocol.json # Declared protocol
  tests/                         # 53 regression and mathematical checks
  dynamics-results/              # Recorded v0.3.0 synthetic results and figures
```

## Licence and citation

All code, configurations, tests, and committed synthetic outputs in this
repository are released under the [MIT License](LICENSE). Please cite release
v0.3.0 and the accompanying article; machine-readable metadata are in
[`CITATION.cff`](CITATION.cff).

The mathematical and numerical checks establish consistency of the declared
mechanism. They do not validate the proposed psychological constructs or prove
that the synthetic effect sizes occur in human observations.
