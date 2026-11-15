# Directional Demand and Resource Dynamics: Reference Implementation

This MIT-licensed repository is the executable reference implementation for:

> Yalçın, M., & Karslı, O. (2026). *Directional Demand and Resource
> Dynamics: A Source-Preserving Model*.

Release **v0.5.0** preserves the v0.4.1 source-preserving numerical record and
adds the input-factorial regulation experiment and exposure-matched robustness
checks reported in the current article. It is the exact public computational
reference for that article. Historical v0.3.0, numerical v0.4.0, and
presentation v0.4.1 tags remain available as preserved records.

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
standard library. Rebuilding the v0.4.1 presentation figures requires Tectonic
and Poppler (`pdftoppm`); rendering the v0.5.0 regulation figure additionally
requires Matplotlib.

## Reproduce v0.5.0

From the repository root:

```sh
python3 -m unittest discover -s simulation/tests -v
python3 simulation/referee_experiments.py \
  --protocol simulation/configs/referee_protocol.json \
  --output /tmp/directional-demand-presentation --render

python3 simulation/recovery_research.py \
  --protocol simulation/configs/recovery_research_protocol.json \
  --output /tmp/directional-demand-recovery --render

python3 simulation/regulation_research.py \
  --protocol simulation/configs/regulation_research_protocol.json \
  --output /tmp/directional-demand-regulation

python3 simulation/render_regulation_figure.py \
  --results /tmp/directional-demand-regulation \
  --output /tmp/directional-demand-regulation/regulation-programs.pdf
```

The complete run performs branch-aware stationary preparation, pulse and
matched-control comparisons, parameter checks, continuous cycles, local return
probes, long-run terminal-state checks, and numerical refinements. It takes
several minutes. The committed `simulation/referee-results/` directory is the
preserved v0.4.0 numerical snapshot. `simulation/presentation-results/` is the
v0.4.1 article-presentation record: it retains the same 150 numerical
artifacts and adds the complete parameter table and presentation assets.
`recovery-research-results/` records the equal-exposure history controls;
`regulation-research-results/` records both input-factorial cost regimes and
the boundary/support comparisons. All new outcomes are synthetic, are retained
whether favourable or unfavourable, and are not human intervention effects.
`recovery-presentation-assets/` contains the exact public synthetic figure and
tables used by the article; `render_regulation_figure.py` regenerates that
figure from the recorded regulation summary.

To regenerate only presentation assets from an already verified result set:

```sh
python3 simulation/referee_experiments.py \
  --protocol simulation/configs/referee_protocol.json \
  --from-results simulation/presentation-results \
  --output /tmp/directional-demand-presentation --render
```

## Contents

```text
simulation/
  analysis.py                    # equilibrium, deviation, and cycle summaries
  model.py                       # bounded gain--loss ODE and RK4 solver
  network.py                     # source-preserving temporal aggregation
  regimes.py                     # stationary construction and local diagnostics
  referee_experiments.py         # complete v0.4.1 runner and figures
  recovery_research.py           # exposure-matched history and scalar-lag controls
  regulation_research.py         # 2^3 regulatory-input and boundary/support comparisons
  render_regulation_figure.py    # article Figure 5 renderer from saved synthetic outputs
  configs/referee_protocol.json  # declared current protocol
  configs/recovery_research_protocol.json
  configs/regulation_research_protocol.json
  tests/                         # 83 regression and mathematical checks
  referee-results/               # preserved v0.4.0 numerical snapshot
  presentation-results/          # v0.4.1 presentation record and assets
  recovery-research-results/     # v0.5.0 exposure-matched record
  regulation-research-results/   # v0.5.0 input-factorial and boundary/support record
  recovery-presentation-assets/  # exact public synthetic assets used in the article
```

The protocol keeps the historical source files required by the test suite, but
the current executable entry point, configuration, and results are the
`referee_*` files above.

## Licence and citation

All code, configurations, tests, and committed synthetic outputs in this
repository are released under the [MIT License](LICENSE). Please cite release
v0.5.0 and the accompanying article; machine-readable metadata are in
[`CITATION.cff`](CITATION.cff).
