# Temporal-Multiplex Relational Dynamics: Reference Implementation

This repository is a small, dependency-free reference implementation of the
illustrative bounded seven-coordinate dynamical system in Appendix A of:

> Yalçın, M., & Karslı, O. (2026). *A Transparent Temporal--Multiplex
> Framework for Relational Dynamics, Cross-Domain Spillover, and
> Self-Regulation*.

It is deliberately independent of the authors' working archive, exploratory
notes, and any future product. It contains only the declared ODE mechanism, a
fully specified non-empirical scenario, a command-line runner, a one-factor
sensitivity utility, and regression tests.

## What this repository does—and does not—provide

The code implements the declared Appendix A mechanism used by the baseline
scenario, including the scalar pressure and exposure proxies and classical
fourth-order Runge--Kutta (RK4) integration. Every numerical value is visible
in the scenario JSON file. The baseline specializes the appendix's exogenous
$c_{\mathrm{tar}}(t)$ target to a constant `c_target`; it does not implement a
time-varying target-path interface, the main-text directional-conflict vectors,
or the path statistic.

It does **not** contain fitted parameters, human-subject data, clinical
thresholds, a validated measurement instrument, person-level predictions, or
a decision-support system. A successful numerical run illustrates the behavior
of declared assumptions; it is not empirical evidence that those assumptions
hold in a population.

## Requirements

Python 3.9 or later. The reference implementation uses only the Python
standard library; no package installation is required.

## Run the declared baseline scenario

From the repository root:

```sh
python3 run.py configs/illustrative_baseline.json \
  --output results/illustrative_baseline.csv
```

The resulting CSV contains time, the seven normalized coordinates, the three
declared inputs, and the scalar pressure and exposure proxies.

## Run a declared sensitivity sweep

```sh
python3 sensitivity.py configs/illustrative_baseline.json \
  --parameter a1 --values 0.10,0.20,0.30,0.40 \
  --output results/a1_terminal_sensitivity.csv
```

This is a transparent one-factor mechanism check, not uncertainty
quantification, parameter estimation, or a population-level result.

## Run the tests

```sh
python3 -m unittest discover -s tests -v
```

The regression suite checks the declared pressure bound, the baseline
trajectory's normalized domain, Appendix A parameter-domain validation,
configuration validation, and a documented structural invariant of this
candidate mechanism.

## Repository layout

```text
model.py                              # Appendix A equations and RK4 solver
run.py                                # Scenario-to-CSV command-line runner
sensitivity.py                        # One-factor sensitivity utility
configs/illustrative_baseline.json    # Fully declared non-empirical scenario
tests/test_model.py                   # Regression tests
```

## Reuse and citation

This software is released under the [MIT License](LICENSE). Reuse, adaptation,
redistribution, and commercial use are permitted subject to that license. If
you use it in research, cite the software release and the accompanying
conceptual article; machine-readable citation metadata are in
[`CITATION.cff`](CITATION.cff).

The article and this reference implementation should not be represented as a
validated analytical, diagnostic, clinical, employment, insurance, legal, or
relationship-decision system.
