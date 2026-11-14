# Branch-aware resource--load experiment — v0.4.1

This directory contains the executable synthetic experiment accompanying the
v0.4.1 reference release. Its numerical version remains v0.4.0. The current
entry point is `referee_experiments.py`; the declared configuration is
`configs/referee_protocol.json`; and the complete article-presentation record
is `presentation-results/`. The earlier `referee-results/` directory remains
the preserved numerical snapshot.

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
```

The test suite contains 74 checks.  A complete run includes stationary
construction, branch selection, controls, parameter conditions, continuous
cycles, long-run relaxation checks, and numerical refinements, so it takes
several minutes.  The code uses the Python standard library; rendering uses
Tectonic and Poppler.

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
