"""Reference implementation of Appendix A of the canonical manuscript.

The module implements one *illustrative* seven-coordinate ODE instance.  It
does not contain fitted parameters, empirical observations, or a predictive
model of any person.  Every numerical value is supplied in a scenario file so
that it can be inspected, replaced, or rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Mapping, Sequence


STATE_NAMES = ("s", "e", "r", "c", "p", "w1", "w2")
INPUT_NAMES = ("delay", "boundary", "review")

# Appendix A: coefficients constrained to the unit interval.
UNIT_INTERVAL_PARAMETERS = (
    "b2",
    "b3",
    "q1",
    "q_s",
    "q_e",
    "q_g",
    "q_h",
    "c_target",
    "c_star",
)

# Every other named coefficient in the illustrative specification is
# non-negative; they have no empirical interpretation in this repository.
NONNEGATIVE_PARAMETERS = (
    "psi_max",
    "a1",
    "a2",
    "a3",
    "b1",
    "rho_s",
    "a_s",
    "nu_s",
    "xi_s",
    "mu_e",
    "omega_e",
    "kappa_e",
    "a_e",
    "P0",
    "alpha_r",
    "a_r",
    "gamma_r",
    "theta_r",
    "eta_r",
    "tau_c",
    "rho_p",
    "a_p",
    "b_p",
    "delta_p",
    "zeta_p",
    "a_w1",
    "b_w1",
    "gamma_12",
    "d_w1",
    "a_w2",
    "b_w2",
    "d_w2",
    "h_w2",
)


@dataclass(frozen=True)
class Diagnostics:
    """Scalar scenario quantities from Equations (16)--(17)."""

    pressure_proxy: float
    exposure_proxy: float


def _as_floats(values: Mapping[str, float], names: Sequence[str], label: str) -> dict[str, float]:
    """Return a complete float mapping, rejecting absent or unexpected keys."""

    expected = set(names)
    received = set(values)
    missing = expected - received
    unexpected = received - expected
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if unexpected:
            details.append(f"unexpected {sorted(unexpected)}")
        raise ValueError(f"{label} keys are invalid: {', '.join(details)}")
    return {name: float(values[name]) for name in names}


def validate_parameters(parameters: Mapping[str, float]) -> dict[str, float]:
    """Validate the explicit sign restrictions stated in Appendix A."""

    names = UNIT_INTERVAL_PARAMETERS + NONNEGATIVE_PARAMETERS
    result = _as_floats(parameters, names, "parameter")
    for name in UNIT_INTERVAL_PARAMETERS:
        if not 0.0 <= result[name] <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1], received {result[name]!r}")
    for name in NONNEGATIVE_PARAMETERS:
        if result[name] < 0.0:
            raise ValueError(f"{name} must be non-negative, received {result[name]!r}")
    if result["psi_max"] <= 0.0:
        raise ValueError("psi_max must be strictly positive")
    return result


def validate_state(state: Sequence[float]) -> tuple[float, ...]:
    """Validate a normalized state vector in the declared mathematical domain."""

    if len(state) != len(STATE_NAMES):
        raise ValueError(f"state must contain {len(STATE_NAMES)} values in order {STATE_NAMES}")
    checked = tuple(float(value) for value in state)
    for name, value in zip(STATE_NAMES, checked):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"state coordinate {name} must lie in [0, 1], received {value!r}")
    return checked


def validate_inputs(inputs: Mapping[str, float]) -> dict[str, float]:
    """Validate the three normalized candidate inputs from Equation (9)."""

    result = _as_floats(inputs, INPUT_NAMES, "input")
    for name, value in result.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"input {name} must lie in [0, 1], received {value!r}")
    return result


def positive_part(value: float) -> float:
    """The positive-part operator [z]_+ from Equation (15)."""

    return max(0.0, value)


def diagnostics(
    state: Sequence[float], inputs: Mapping[str, float], parameters: Mapping[str, float]
) -> Diagnostics:
    """Evaluate the explicitly labelled scenario proxies in Equations (16)--(17)."""

    s, e, _r, c, p, _w1, w2 = validate_state(state)
    u = validate_inputs(inputs)
    par = validate_parameters(parameters)
    permeability_excess = positive_part(c - par["c_star"])
    raw_pressure = (
        par["a1"] * w2 * (1.0 - par["q1"] * u["delay"])
        + par["a2"] * (1.0 - s * p)
        + par["a3"] * (e + permeability_excess)
    )
    pressure = min(par["psi_max"], max(0.0, raw_pressure))
    exposure = (
        w2
        * (1.0 + par["b1"] * permeability_excess)
        * (1.0 - par["b2"] * s * p)
        * (1.0 - par["b3"] * u["boundary"])
    )
    return Diagnostics(pressure_proxy=pressure, exposure_proxy=exposure)


def derivative(
    state: Sequence[float], inputs: Mapping[str, float], parameters: Mapping[str, float]
) -> tuple[float, ...]:
    """Evaluate the seven right-hand sides in Equation (18) of Appendix A."""

    s, e, r, c, p, w1, w2 = validate_state(state)
    u = validate_inputs(inputs)
    par = validate_parameters(parameters)
    proxy = diagnostics(state, inputs, parameters)
    psi = proxy.pressure_proxy
    exposure = proxy.exposure_proxy

    ds = (
        (par["rho_s"] * r * u["review"] + par["a_s"] * p) * (1.0 - s)
        - (par["nu_s"] * e + par["xi_s"] * c * psi * (1.0 - par["q_s"] * u["delay"])) * s
    )
    de = (
        (par["mu_e"] * (1.0 - s) * (1.0 - p) * psi + par["omega_e"] * w2 * (1.0 - par["q_e"] * u["delay"]))
        * (1.0 - e)
        - (par["kappa_e"] * s * p + par["a_e"] * u["review"]) * e
    )
    dr = (
        (par["P0"] + par["alpha_r"] * s * p + par["a_r"] * w1) * (1.0 - r)
        - (par["gamma_r"] * e**2 + par["theta_r"] * c * exposure + par["eta_r"] * psi) * r
    )
    dc = par["tau_c"] * (par["c_target"] - c)
    dp = (
        par["rho_p"] * s * (par["a_p"] + par["b_p"] * w1 * u["review"]) * (1.0 - p)
        - (par["delta_p"] * e**2 + par["zeta_p"] * w2 * (1.0 - s) * (1.0 - u["delay"])) * p
    )
    dw1 = (
        (par["a_w1"] * r * p + par["b_w1"] * s * (1.0 - e)) * (1.0 - w1)
        - (par["gamma_12"] * w2 * psi * (1.0 - par["q_g"] * u["boundary"]) + par["d_w1"] * e * (1.0 - s)) * w1
    )
    dw2 = (
        (par["a_w2"] * c * (1.0 - par["q_h"] * u["boundary"]) + par["b_w2"] * e) * (1.0 - w2)
        - (par["d_w2"] * s * p * u["boundary"] + par["h_w2"] * u["delay"]) * w2
    )
    return (ds, de, dr, dc, dp, dw1, dw2)


def rk4_step(
    state: Sequence[float], dt: float, inputs: Mapping[str, float], parameters: Mapping[str, float]
) -> tuple[float, ...]:
    """Advance the system one classical fourth-order Runge--Kutta step."""

    if dt <= 0.0:
        raise ValueError("dt must be positive")
    start = validate_state(state)

    def shifted(base: Sequence[float], slope: Sequence[float], scale: float) -> tuple[float, ...]:
        return tuple(value + scale * increment for value, increment in zip(base, slope))

    k1 = derivative(start, inputs, parameters)
    k2 = derivative(shifted(start, k1, dt / 2.0), inputs, parameters)
    k3 = derivative(shifted(start, k2, dt / 2.0), inputs, parameters)
    k4 = derivative(shifted(start, k3, dt), inputs, parameters)
    advanced = tuple(
        value + dt * (slope1 + 2.0 * slope2 + 2.0 * slope3 + slope4) / 6.0
        for value, slope1, slope2, slope3, slope4 in zip(start, k1, k2, k3, k4)
    )

    # The continuous system is bounded by Proposition 1.  This branch handles
    # floating-point round-off only; a material excursion signals an unsuitable
    # numerical step or invalid configuration rather than being silently clipped.
    tolerance = 1e-12
    if any(value < -tolerance or value > 1.0 + tolerance for value in advanced):
        raise RuntimeError("RK4 step left [0, 1]^7; reduce dt or inspect the scenario")
    return tuple(0.0 if isclose(value, 0.0, abs_tol=tolerance) else 1.0 if isclose(value, 1.0, abs_tol=tolerance) else value for value in advanced)


def inputs_at_time(input_specification: Mapping[str, object], time: float) -> dict[str, float]:
    """Apply declared piecewise-constant input overrides at a simulation time."""

    baseline = validate_inputs(input_specification["baseline"])
    schedule = input_specification.get("schedule", [])
    if not isinstance(schedule, list):
        raise ValueError("inputs.schedule must be a list")
    active = dict(baseline)
    for interval in schedule:
        if not isinstance(interval, Mapping):
            raise ValueError("each input schedule item must be an object")
        start = float(interval["start"])
        end = float(interval["end"])
        if end <= start:
            raise ValueError("input schedule intervals must have end > start")
        if start <= time < end:
            overrides = interval.get("values", {})
            if not isinstance(overrides, Mapping):
                raise ValueError("input schedule values must be an object")
            invalid = set(overrides) - set(INPUT_NAMES)
            if invalid:
                raise ValueError(f"invalid scheduled input names: {sorted(invalid)}")
            active.update({name: float(value) for name, value in overrides.items()})
    return validate_inputs(active)


def simulate(configuration: Mapping[str, object]) -> list[dict[str, float]]:
    """Run a fully declared illustrative scenario and return a tidy time series."""

    try:
        solver = configuration["solver"]
        initial_state = configuration["initial_state"]
        input_specification = configuration["inputs"]
        parameters = configuration["parameters"]
    except KeyError as error:
        raise ValueError(f"scenario is missing required key {error.args[0]!r}") from error
    if not isinstance(solver, Mapping) or not isinstance(input_specification, Mapping):
        raise ValueError("solver and inputs must be objects")
    dt = float(solver["dt"])
    horizon = float(solver["time_horizon"])
    if dt <= 0.0 or horizon <= 0.0:
        raise ValueError("solver dt and time_horizon must be positive")
    steps = round(horizon / dt)
    if not isclose(steps * dt, horizon, abs_tol=1e-10):
        raise ValueError("time_horizon must be an integer multiple of dt")

    state = validate_state(initial_state)
    par = validate_parameters(parameters)
    rows: list[dict[str, float]] = []
    for step in range(steps + 1):
        time = step * dt
        inputs = inputs_at_time(input_specification, time)
        proxy = diagnostics(state, inputs, par)
        row = {"time": time, **dict(zip(STATE_NAMES, state)), **inputs}
        row["pressure_proxy"] = proxy.pressure_proxy
        row["exposure_proxy"] = proxy.exposure_proxy
        rows.append(row)
        if step < steps:
            state = rk4_step(state, dt, inputs, par)
    return rows
