# Branch-aware resource--load experiment — v0.5.0

This directory contains the executable synthetic experiment accompanying the
v0.5.0 reference release. Its earlier numerical version remains v0.4.0. The
source-preserving entry point is `referee_experiments.py`; its declared
configuration is `configs/referee_protocol.json`; and its article-presentation
record is `presentation-results/`. The v0.5.0 extensions are
`recovery_research.py` and `regulation_research.py`, with separately declared
protocols and recorded outputs. The earlier `referee-results/` directory
remains the preserved numerical snapshot.

All coordinates, coefficients, schedules, and time units are synthetic and
uncalibrated.  They demonstrate consequences of the implemented equations,
not human effect sizes or personal predictions.

## Reproduce

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
```

The test suite contains 83 checks. A complete run includes stationary
construction, branch selection, controls, parameter conditions, continuous
cycles, long-run relaxation checks, and numerical refinements, so it takes
several minutes.  The code uses the Python standard library; rendering uses
Tectonic and Poppler for the earlier presentation figures. The v0.5.0
regulation figure renderer uses Matplotlib only.

For a presentation-only rerender of the checked result set:

```sh
python3 simulation/referee_experiments.py \
  --protocol simulation/configs/referee_protocol.json \
  --from-results simulation/presentation-results \
  --output /tmp/directional-demand-presentation --render
```

The `--from-results` mode verifies the saved protocol, code fingerprints,
scenario records, grids, and endpoints before reusing numerical outputs.

## Current data contract

`referee-results/` contains the preserved 158-file v0.4.0 numerical snapshot.
`presentation-results/` contains the same 150 numerical artifacts, six figure
artifacts, an explicit 49-coefficient table, and a deterministic presentation
manifest: 160 files in total. The complete scenario settings remain in
`presentation-results/report.json`.

The primary direction and full-versus-core comparisons hold the permeability
target fixed.  A target change is a separately recorded intervention.  The
interior reference and lower stationary branch use the same constructed
coefficient set; a preserved saturation regime is included for comparison.

Historical `revision_*` and `dynamics_*` files remain to retain executable
provenance for previous tags. They are not the v0.4.1 reproduction path.

## v0.5.0 recovery extensions

`recovery_research.py` tests one-, two-, and four-block equal-exposure
histories with matched final exposure end points, continuation to stationary
references, frozen-background controls, and an exact scalar-lag comparator.
`regulation_research.py` evaluates all $2^3$ combinations of delay (D),
boundary coordination (B), and review (R), plus declared boundary/support
comparisons. The committed recovery and regulation result folders contain all
recorded synthetic outcomes. They are mechanism illustrations, not estimated
or clinical effects. `render_regulation_figure.py` recreates the article's
regulation comparison from `regulation-research-results/summary.csv`.
