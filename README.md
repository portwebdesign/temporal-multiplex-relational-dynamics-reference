# Directional Demand and Resource Dynamics: Reference Implementation

This MIT-licensed repository is the executable reference implementation for:

> Yalçın, M., & Karslı, O. (2026). *Directional Demand and Resource
> Dynamics: A Source-Preserving Model*.

Release **v0.4.0** implements the source-preserving, branch-aware
resource--load experiment used by the current article. It supersedes the
historical v0.3.0 implementation for reproducing the current article; earlier
tags remain available as preserved historical records.

## Scope

The model keeps each declared source demand separate until its directional
disagreement is computed. It then maps external demand, support, and conflict
signals into a bounded seven-coordinate gain--loss system. The protocol
constructs a specified interior stationary reference, retains a second branch
under the same coefficients, compares demand pulses at a constant permeability
target, and treats a target shift as a separate intervention.

All coefficients, states, time units, scenarios, and outputs are synthetic and
uncalibrated. This software is not a clinical, diagnostic, employment,
insurance, legal, or person-level forecasting tool. Its tests verify the
declared mathematics and implementation; they do not validate the proposed
psychological constructs in human observations.

The repository contains only reusable software, configurations, tests, and
recorded synthetic outputs. It deliberately excludes manuscripts, private
working notes, reviewer material, and personal data.

## Requirements

Python 3.9 or later is required. The numerical model uses only the Python
standard library. Rebuilding vector figures requires Tectonic and Poppler
(`pdftoppm`).

## Reproduce v0.4.0

From the repository root:

```sh
python3 -m unittest discover -s simulation/tests -v
python3 simulation/referee_experiments.py \
  --protocol simulation/configs/referee_protocol.json \
  --output /tmp/directional-demand-results --render
```

The complete run performs branch-aware stationary preparation, pulse and
matched-control comparisons, parameter checks, continuous cycles, local return
probes, long-run terminal-state checks, and numerical refinements. It takes
several minutes. The committed `simulation/referee-results/` directory is the
v0.4.0 result snapshot.

To regenerate only presentation assets from an already verified result set:

```sh
python3 simulation/referee_experiments.py \
  --protocol simulation/configs/referee_protocol.json \
  --from-results simulation/referee-results \
  --output /tmp/directional-demand-presentation --render
```

## Contents

```text
simulation/
  analysis.py                    # equilibrium, deviation, and cycle summaries
  model.py                       # bounded gain--loss ODE and RK4 solver
  network.py                     # source-preserving temporal aggregation
  regimes.py                     # stationary construction and local diagnostics
  referee_experiments.py         # complete v0.4.0 runner and figures
  configs/referee_protocol.json  # declared current protocol
  tests/                         # 74 regression and mathematical checks
  referee-results/               # 158 recorded v0.4.0 outputs and figures
```

The protocol keeps the historical source files required by the test suite, but
the current executable entry point, configuration, and results are the
`referee_*` files above.

## Licence and citation

All code, configurations, tests, and committed synthetic outputs in this
repository are released under the [MIT License](LICENSE). Please cite release
v0.4.0 and the accompanying article; machine-readable metadata are in
[`CITATION.cff`](CITATION.cff).
