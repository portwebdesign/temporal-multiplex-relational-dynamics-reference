"""Reference implementation of Appendix A of the canonical manuscript.

The module implements one *illustrative* seven-coordinate ODE instance.  It
does not contain fitted parameters, empirical observations, or a predictive
model of any person.  Every numerical value is supplied in a scenario file so
that it can be inspected, replaced, or rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite, nextafter
from typing import Callable, Mapping, Sequence

from network import NetworkSignals, aggregate_network, interpolate_network, network_at_time


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
    "a_conflict",
    "lambda_d",
    "lambda_h",
    "k_visibility",
    "k_access",
    "k_input",
    "k_isolation",
)


@dataclass(frozen=True)
class Diagnostics:
    """Scalar pressure and exposure quantities from the Appendix mechanism."""

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
    result = {name: float(values[name]) for name in names}
    if any(not isfinite(value) for value in result.values()):
        raise ValueError(f"{label} values must be finite")
    return result


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
    """Validate the three normalized candidate inputs."""

    result = _as_floats(inputs, INPUT_NAMES, "input")
    for name, value in result.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"input {name} must lie in [0, 1], received {value!r}")
    return result


def positive_part(value: float) -> float:
    """The positive-part operator [z]_+."""

    return max(0.0, value)


def diagnostics(
    state: Sequence[float], inputs: Mapping[str, float], parameters: Mapping[str, float],
    signals: NetworkSignals | None = None,
) -> Diagnostics:
    """Evaluate the declared pressure and exposure proxies."""

    s, e, _r, c, p, _w1, w2 = validate_state(state)
    u = validate_inputs(inputs)
    par = validate_parameters(parameters)
    signals = signals or NetworkSignals()
    validate_signals(signals)
    permeability_excess = positive_part(c - par["c_star"])
    raw_pressure = (
        par["a1"] * w2 * (1.0 - par["q1"] * u["delay"])
        + par["a2"] * (1.0 - s * p)
        + par["a3"] * (e + permeability_excess)
        + par["a_conflict"] * signals.conflict
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
    state: Sequence[float], inputs: Mapping[str, float], parameters: Mapping[str, float],
    signals: NetworkSignals | None = None,
    permeability_target: float | None = None,
) -> tuple[float, ...]:
    """Evaluate the seven right-hand sides of the Appendix mechanism."""

    s, e, r, c, p, w1, w2 = validate_state(state)
    u = validate_inputs(inputs)
    par = validate_parameters(parameters)
    signals = signals or NetworkSignals()
    proxy = diagnostics(state, inputs, parameters, signals)
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
        (par["P0"] + par["alpha_r"] * s * p + par["a_r"] * w1
         + par["k_visibility"] * e + par["k_access"] * c * signals.support) * (1.0 - r)
        - (par["gamma_r"] * e**2 + par["theta_r"] * c * exposure + par["eta_r"] * psi
           + par["k_input"] * sum(value**2 for value in u.values())) * r
    )
    target=validate_target(par["c_target"] if permeability_target is None else permeability_target)
    dc = par["tau_c"] * (target - c)
    dp = (
        par["rho_p"] * s * (par["a_p"] + par["b_p"] * w1 * u["review"]) * (1.0 - p)
        - (par["delta_p"] * e**2 + par["zeta_p"] * w2 * (1.0 - s) * (1.0 - u["delay"])) * p
    )
    dw1 = (
        (par["a_w1"] * r * p + par["b_w1"] * s * (1.0 - e)
         + par["lambda_h"] * signals.support) * (1.0 - w1)
        - (par["gamma_12"] * w2 * psi * (1.0 - par["q_g"] * u["boundary"])
           + par["d_w1"] * e * (1.0 - s)
           + par["k_isolation"] * (1.0 - c) * signals.support) * w1
    )
    dw2 = (
        (par["lambda_d"] * signals.external_demand
         + par["a_w2"] * c * (1.0 - par["q_h"] * u["boundary"]) + par["b_w2"] * e) * (1.0 - w2)
        - (par["d_w2"] * s * p * u["boundary"] + par["h_w2"] * u["delay"]) * w2
    )
    return (ds, de, dr, dc, dp, dw1, dw2)


def rk4_step(
    state: Sequence[float], dt: float, inputs: Mapping[str, float], parameters: Mapping[str, float],
    signals: NetworkSignals | None = None, active_coordinates: Sequence[str] = STATE_NAMES,
    permeability_target: float | None = None,
) -> tuple[float, ...]:
    """Advance one RK4 step under constant exogenous forcing."""
    return rk4_time_step(state,0.0,dt,lambda _time:inputs,parameters,
                         lambda _time:signals,lambda _time:permeability_target,
                         active_coordinates)


def rk4_time_step(
    state: Sequence[float], time: float, dt: float,
    inputs_at: Callable[[float],Mapping[str,float]], parameters: Mapping[str,float],
    signals_at: Callable[[float],NetworkSignals | None] | None=None,
    permeability_target_at: Callable[[float],float | None] | None=None,
    active_coordinates: Sequence[str]=STATE_NAMES, end_left_limit: bool=False,
) -> tuple[float,...]:
    """Evaluate time-varying forcing at every RK4 stage within one event segment."""

    if not isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive")
    if not isfinite(time):
        raise ValueError("time must be finite")
    if not isfinite(time+dt) or time+dt<=time:
        raise ValueError("RK4 time step must advance finite time")
    if not active_coordinates or len(set(active_coordinates))!=len(active_coordinates) or set(active_coordinates)-set(STATE_NAMES):
        raise ValueError("active_coordinates must be a nonempty unique subset of state names")
    start = validate_state(state)

    def shifted(base: Sequence[float], slope: Sequence[float], scale: float) -> tuple[float, ...]:
        return tuple(value + scale * increment for value, increment in zip(base, slope))

    def rate(values,stage_time):
        signals=signals_at(stage_time) if signals_at else None
        target=permeability_target_at(stage_time) if permeability_target_at else None
        rates = derivative(values,inputs_at(stage_time),parameters,signals,target)
        return tuple(value if name in active_coordinates else 0.0 for name, value in zip(STATE_NAMES, rates))

    k1 = rate(start,time)
    k2 = rate(shifted(start,k1,dt/2.0),time+dt/2.0)
    k3 = rate(shifted(start,k2,dt/2.0),time+dt/2.0)
    stage_end=nextafter(time+dt,time) if end_left_limit else time+dt
    k4 = rate(shifted(start,k3,dt),stage_end)
    advanced = tuple(
        value + dt * (slope1 + 2.0 * slope2 + 2.0 * slope3 + slope4) / 6.0
        for value, slope1, slope2, slope3, slope4 in zip(start, k1, k2, k3, k4)
    )

    # The continuous system has the stated gain-loss domain property.  This branch handles
    # floating-point round-off only; a material excursion signals an unsuitable
    # numerical step or invalid configuration rather than being silently clipped.
    tolerance = 1e-12
    if any(value < -tolerance or value > 1.0 + tolerance for value in advanced):
        raise RuntimeError("RK4 step left [0, 1]^7; reduce dt or inspect the scenario")
    return tuple(0.0 if isclose(value, 0.0, abs_tol=tolerance, rel_tol=0.0) else 1.0 if isclose(value, 1.0, abs_tol=tolerance, rel_tol=0.0) else value for value in advanced)


def validate_target(value: float) -> float:
    target=float(value)
    if not isfinite(target) or not 0<=target<=1:
        raise ValueError("permeability target must be finite and lie in [0,1]")
    return target


def target_at_time(specification: Mapping[str,object] | None, time: float, fallback: float) -> float:
    """Evaluate an independently declared piecewise-constant permeability target."""
    if specification is None:
        return validate_target(fallback)
    if set(specification)!={"baseline","schedule"} or not isinstance(specification["schedule"],list):
        raise ValueError("permeability_target must contain baseline and schedule")
    target=validate_target(specification["baseline"])
    previous_end=-float("inf")
    for interval in sorted(specification["schedule"],key=lambda item:item["start"]):
        if set(interval)!={"start","end","value"}:
            raise ValueError("target intervals must contain start, end and value")
        start,end=float(interval["start"]),float(interval["end"])
        value=validate_target(interval["value"])
        if not isfinite(start) or not isfinite(end) or start<0 or end<=start or start<previous_end:
            raise ValueError("target intervals must be finite, nonoverlapping and have end > start")
        previous_end=end
        if start<=time<end:
            target=value
    return target


def inputs_at_time(input_specification: Mapping[str, object], time: float) -> dict[str, float]:
    """Apply declared piecewise-constant input overrides at a simulation time."""

    baseline = validate_inputs(input_specification["baseline"])
    schedule = input_specification.get("schedule", [])
    if not isinstance(schedule, list):
        raise ValueError("inputs.schedule must be a list")
    active = dict(baseline)
    previous_end = -float("inf")
    for interval in sorted(schedule, key=lambda item: item["start"]):
        if not isinstance(interval, Mapping):
            raise ValueError("each input schedule item must be an object")
        start = float(interval["start"])
        end = float(interval["end"])
        if not isfinite(start) or not isfinite(end) or start < 0 or end <= start or start < previous_end:
            raise ValueError("input schedule intervals must have end > start")
        previous_end = end
        overrides = interval.get("values", {})
        if not isinstance(overrides, Mapping):
            raise ValueError("input schedule values must be an object")
        invalid = set(overrides) - set(INPUT_NAMES)
        if invalid:
            raise ValueError(f"invalid scheduled input names: {sorted(invalid)}")
        validate_inputs({**baseline, **overrides})
        if start <= time < end:
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
    if not isfinite(dt) or not isfinite(horizon) or dt <= 0.0 or horizon <= 0.0:
        raise ValueError("solver dt and time_horizon must be positive")
    steps = round(horizon / dt)
    if not isclose(steps * dt, horizon, abs_tol=1e-10):
        raise ValueError("time_horizon must be an integer multiple of dt")

    state = validate_state(initial_state)
    par = validate_parameters(parameters)
    if solver.get("method") != "classical_runge_kutta_4":
        raise ValueError("only classical_runge_kutta_4 is implemented")
    active_coordinates = configuration.get("active_coordinates", STATE_NAMES)
    if not active_coordinates or len(set(active_coordinates)) != len(active_coordinates) or set(active_coordinates) - set(STATE_NAMES):
        raise ValueError("active_coordinates must be a nonempty unique subset of state names")
    network = configuration.get("network")
    network_at_time(network, 0.0)
    # Validate once, then reuse the constant signals on each declared segment.
    # Recomputing aggregation is unnecessary because edges are exogenous here.
    base_signals = aggregate_network(network["baseline"]) if network else NetworkSignals()
    network_intervals = [(float(item["start"]),float(item["end"]),aggregate_network(item["network"]),item)
                         for item in network["schedule"]] if network else []
    inputs_at_time(input_specification,0.0)
    base_inputs = validate_inputs(input_specification["baseline"])
    input_intervals = [(float(item["start"]),float(item["end"]),dict(item.get("values",{})))
                       for item in input_specification.get("schedule",[])]
    target_specification=configuration.get("permeability_target")
    target_at_time(target_specification,0.0,par["c_target"])
    base_target=validate_target(target_specification["baseline"] if target_specification else par["c_target"])
    target_intervals=[(float(item["start"]),float(item["end"]),float(item["value"]))
                      for item in target_specification["schedule"]] if target_specification else []

    def get_signals(time):
        for start,end,value,item in network_intervals:
            if start <= time < end:
                if "end_network" in item:
                    return interpolate_network(item["network"],item["end_network"],(time-start)/(end-start),validated=True)
                return value
        return base_signals

    def get_inputs(time):
        for start,end,values in input_intervals:
            if start <= time < end:
                return {**base_inputs, **values}
        return base_inputs

    def get_target(time):
        for start,end,value in target_intervals:
            if start<=time<end:
                return value
        return base_target

    events = set()
    for schedule in (input_specification.get("schedule", []),network.get("schedule", []) if network else [],target_specification["schedule"] if target_specification else []):
        for interval in schedule:
            for name in ("start", "end"):
                value = float(interval[name])
                if not isfinite(value):
                    raise ValueError("schedule times must be finite")
                if 0 <= value <= horizon:
                    events.add(value)
    rows: list[dict[str, float]] = []
    for step in range(steps + 1):
        time = step * dt
        inputs = get_inputs(time)
        signals = get_signals(time)
        proxy = diagnostics(state, inputs, par, signals)
        row = {"time": time, **dict(zip(STATE_NAMES, state)), **inputs}
        row["pressure_proxy"] = proxy.pressure_proxy
        row["exposure_proxy"] = proxy.exposure_proxy
        row["c_target"]=get_target(time)
        row.update({name: getattr(signals, name) for name in signals.__dataclass_fields__})
        rows.append(row)
        if step < steps:
            # Integrate each constant-input segment separately. The right side
            # of a jump belongs to the next segment; no RK stage crosses it.
            endpoint = (step + 1) * dt
            breaks = sorted(value for value in events if time < value < endpoint) + [endpoint]
            segment_start = time
            for segment_end in breaks:
                last_inside=nextafter(segment_end,segment_start)
                state = rk4_time_step(state,segment_start,segment_end-segment_start,
                                      lambda at:get_inputs(min(at,last_inside)),par,
                                      lambda at:get_signals(min(at,last_inside)),
                                      lambda at:get_target(min(at,last_inside)),
                                      active_coordinates,end_left_limit=True)
                segment_start = segment_end
    return rows


def validate_signals(signals: NetworkSignals) -> None:
    for name in ("external_demand", "support", "conflict"):
        value = getattr(signals, name)
        if not isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"network signal {name} must lie in [0,1]")
    for name in ("raw_conflict", "total_demand", "net_demand"):
        value = getattr(signals, name)
        if not isfinite(value) or value < 0:
            raise ValueError(f"network signal {name} must be finite and nonnegative")
