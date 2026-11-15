# Directional Demand and Resource Dynamics: Reference Implementation

This MIT-licensed repository is the executable reference implementation for:

> Yalçın, M., & Karslı, O. (2026). *Directional Demand and Resource
> Dynamics: A Source-Preserving Model*.

Release **v0.5.1** preserves the v0.5.0 record and adds the local stationary-
branch continuation and cost--time regulatory-program robustness analyses
reported in the current article. It is the exact public computational reference
for that article. Historical v0.3.0, numerical v0.4.0, presentation v0.4.1,
and v0.5.0 tags remain preserved records.

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

Python 3.9 or later is required. The original numerical model uses only the
Python standard library. The v0.5.1 continuation and high-accuracy
cost--time integration need NumPy and SciPy; the public figure renderers need
Matplotlib. Install the tested public environment with:

```sh
python3 -m pip install -r requirements.txt
```

Rebuilding the v0.4.1 presentation figures additionally requires Tectonic and
Poppler (`pdftoppm`).

## Reproduce v0.5.1

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

mkdir -p /tmp/directional-demand-equilibrium /tmp/directional-demand-cost-time
python3 simulation/equilibrium_robustness.py \
  --protocol simulation/configs/equilibrium_robustness_protocol.json \
  --output /tmp/directional-demand-equilibrium

python3 simulation/regulation_robustness.py \
  --protocol simulation/configs/regulation_robustness_protocol.json \
  --output /tmp/directional-demand-cost-time

python3 simulation/render_robustness_figures.py \
  --output /tmp/directional-demand-robustness-figures
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

The v0.5.1 additions are `equilibrium_robustness.py`, which uses
pseudo-arclength continuation for six separately varied synthetic coefficients,
and `regulation_robustness.py`, which maps eight specified programs across nine
input costs and 101 post-pulse observation times.  They preserve all accepted,
rejected, and unresolved numerical records. `render_robustness_figures.py`
rebuilds the stationary-branch and cost--time figures from those saved records.
The new analyses are local, one-parameter or finite-grid synthetic mechanism
studies; they do not estimate human thresholds, establish an optimal program,
or validate the psychological constructs.

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
  equilibrium_robustness.py      # local stationary-branch continuation study
  regulation_robustness.py       # cost--time programme-ranking robustness study
  render_regulation_figure.py    # article Figure 5 renderer from saved synthetic outputs
  render_robustness_figures.py   # article Figures 6--7 renderer from saved records
  configs/referee_protocol.json  # declared current protocol
  configs/recovery_research_protocol.json
  configs/regulation_research_protocol.json
  configs/equilibrium_robustness_protocol.json
  configs/regulation_robustness_protocol.json
  tests/                         # 92 regression and mathematical checks
  referee-results/               # preserved v0.4.0 numerical snapshot
  presentation-results/          # v0.4.1 presentation record and assets
  recovery-research-results/     # v0.5.0 exposure-matched plus v0.5.1 branch records
  regulation-research-results/   # v0.5.0 factorial plus v0.5.1 cost--time records
  recovery-presentation-assets/  # exact public synthetic assets used in the article
```

The `referee_*` files preserve the historical v0.4.1 record required by the
test suite. The declared v0.5.1 entry points and configuration files are the
four research and robustness runners listed above.

## Licence and citation

All code, configurations, tests, and committed synthetic outputs in this
repository are released under the [MIT License](LICENSE). Please cite release
v0.5.1 and the accompanying article; machine-readable metadata are in
[`CITATION.cff`](CITATION.cff).
