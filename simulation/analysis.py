"""Stationary-baseline checks and declared trajectory statistics."""

from __future__ import annotations

from bisect import bisect_right
from copy import deepcopy
from dataclasses import asdict, dataclass
from math import isfinite
from typing import Mapping, Sequence

from model import (STATE_NAMES, derivative, rk4_step, target_at_time,
                   validate_inputs, validate_parameters, validate_state)
from network import NetworkSignals, aggregate_network


@dataclass(frozen=True)
class EquilibriumResult:
    state: tuple[float,...]
    residual: float
    full_residual: float
    time: float
    steps: int
    initial_state: tuple[float,...]
    dt: float
    step_reductions: int
    active_coordinates: tuple[str,...]

    def as_dict(self) -> dict:
        return asdict(self)


def stationary_configuration(configuration: Mapping[str,object]) -> dict:
    """Hold every exogenous process at its declared pre-event baseline."""
    baseline=deepcopy(configuration)
    baseline["inputs"]["schedule"]=[]
    if baseline.get("network") is not None:
        baseline["network"]["schedule"]=[]
    if baseline.get("permeability_target") is not None:
        baseline["permeability_target"]["schedule"]=[]
    return baseline


def find_equilibrium(configuration: Mapping[str,object], initial_state: Sequence[float] | None=None,
                     *, dt: float=0.05, tolerance: float=1e-10, max_time: float=5000.0,
                     check_interval: float=1.0, min_dt: float=1e-6) -> EquilibriumResult:
    """Integrate baseline dynamics until the active-coordinate residual is small.

    The returned residual is the infinity norm of the specified vector field.
    A state reached from one seed gives no claim about other basins or global
    stability. Material step failures reduce the step and are recorded; failure
    to meet the finite-time/resolution limits raises an error.
    """
    options=(dt,tolerance,max_time,check_interval,min_dt)
    if any(not isfinite(value) or value<=0 for value in options) or min_dt>dt:
        raise ValueError("equilibrium options must be finite positive values with min_dt <= dt")
    config=stationary_configuration(configuration)
    parameters=validate_parameters(config["parameters"])
    inputs=validate_inputs(config["inputs"]["baseline"])
    network=config.get("network")
    signals=aggregate_network(network["baseline"]) if network else NetworkSignals()
    target=target_at_time(config.get("permeability_target"),0.0,parameters["c_target"])
    active=tuple(config.get("active_coordinates",STATE_NAMES))
    if not active or len(set(active))!=len(active) or set(active)-set(STATE_NAMES):
        raise ValueError("active_coordinates must be a nonempty unique subset of state names")
    selected=[STATE_NAMES.index(name) for name in active]
    seed=validate_state(config["initial_state"] if initial_state is None else initial_state)
    state=seed
    time=0.0
    steps=0
    actual_dt=float(dt)
    reductions=0
    next_check=0.0
    residual=float("inf")
    while True:
        if time>=next_check-1e-12 or time>=max_time:
            rates=derivative(state,inputs,parameters,signals,target)
            residual=max(abs(rates[index]) for index in selected)
            full_residual=max(abs(value) for value in rates)
            if residual<=tolerance:
                return EquilibriumResult(state,residual,full_residual,time,steps,seed,
                                         actual_dt,reductions,active)
            if time>=max_time:
                raise RuntimeError(f"baseline residual {residual:.6g} exceeds {tolerance:.6g} at max_time={max_time:g}")
            next_check=min(time+check_interval,max_time)
        step=min(actual_dt,next_check-time,max_time-time)
        if step<=0:
            raise RuntimeError("stationary integration made no time progress")
        try:
            advanced=rk4_step(state,step,inputs,parameters,signals,active,target)
        except (ValueError,RuntimeError) as error:
            actual_dt*=0.5
            reductions+=1
            if actual_dt<min_dt:
                raise RuntimeError("stationary integration exhausted its minimum step") from error
            continue
        state=advanced
        time+=step
        steps+=1


def find_equilibria(configuration: Mapping[str,object], seeds: Sequence[Sequence[float]],
                    **options) -> list[EquilibriumResult]:
    """Record independent seed probes without merging distinct limiting states."""
    if not seeds:
        raise ValueError("provide at least one stationary-baseline seed")
    return [find_equilibrium(configuration,seed,**options) for seed in seeds]


def deviation_index(state: Sequence[float], reference: Sequence[float]) -> float:
    """Equal-weight V(x)=one half of squared distance from a declared reference."""
    state=validate_state(state)
    reference=validate_state(reference)
    return 0.5*sum((value-base)**2 for value,base in zip(state,reference))


def deviation_trace(rows: Sequence[Mapping[str,float]], reference: Sequence[float]) -> list[float]:
    return [deviation_index([row[name] for name in STATE_NAMES],reference) for row in rows]


def _finite_series(values: Sequence[float], label: str) -> list[float]:
    series=[float(value) for value in values]
    if any(not isfinite(value) for value in series):
        raise ValueError(f"{label} must contain finite values")
    return series


def _interpolate(axis: Sequence[float], values: Sequence[float], point: float) -> float:
    if point<=axis[0]:
        return values[0]
    if point>=axis[-1]:
        return values[-1]
    index=bisect_right(axis,point)-1
    fraction=(point-axis[index])/(axis[index+1]-axis[index])
    return values[index]+fraction*(values[index+1]-values[index])


def delta_deviation(rows: Sequence[Mapping[str,float]], reference: Sequence[float],
                    lag: float=1.0) -> list[float | None]:
    """Forward V(t+lag)-V(t), interpolating only within the observed time range."""
    if not isfinite(lag) or lag<=0 or not rows:
        raise ValueError("delta deviation requires a positive finite lag and nonempty rows")
    times=_finite_series([row["time"] for row in rows],"time")
    if any(right<=left for left,right in zip(times,times[1:])):
        raise ValueError("time must increase strictly")
    values=deviation_trace(rows,reference)
    return [None if time+lag>times[-1]+1e-12 else
            _interpolate(times,values,min(time+lag,times[-1]))-value
            for time,value in zip(times,values)]


def cycle_statistics(pressure: Sequence[float], output: Sequence[float]) -> dict:
    """Integrate matched ascending/descending branches of a triangular input cycle.

    Both branches are interpolated to the union of their pressure knots.
    Signed area follows the ascending-then-descending orientation. Absolute
    separation splits intervals at zero crossings of the branch difference.
    Input endpoints must agree; an output endpoint gap is recorded explicitly.
    """
    pressure=_finite_series(pressure,"pressure")
    output=_finite_series(output,"output")
    if len(pressure)!=len(output) or len(pressure)<3:
        raise ValueError("cycle input/output must have equal length of at least three")
    peak=max(range(len(pressure)),key=pressure.__getitem__)
    if peak==0 or peak==len(pressure)-1:
        raise ValueError("a triangular cycle needs an interior maximum")
    if any(right<=left for left,right in zip(pressure[:peak],pressure[1:peak+1])) or any(right>=left for left,right in zip(pressure[peak:-1],pressure[peak+1:])):
        raise ValueError("cycle branches must increase then decrease strictly; plateaus and steps are unsupported")
    if abs(pressure[0]-pressure[-1])>1e-12*max(1.0,abs(pressure[0]),abs(pressure[-1])):
        raise ValueError("cycle branches must cover the same input range")
    up_axis=pressure[:peak+1]
    up_values=output[:peak+1]
    down_axis=list(reversed(pressure[peak:]))
    down_values=list(reversed(output[peak:]))
    low=max(up_axis[0],down_axis[0])
    high=min(up_axis[-1],down_axis[-1])
    grid=sorted({low,high,*[value for value in up_axis+down_axis if low<value<high]})
    difference=[_interpolate(up_axis,up_values,value)-_interpolate(down_axis,down_values,value) for value in grid]
    signed=0.0
    absolute=0.0
    for left,right,d0,d1 in zip(grid,grid[1:],difference,difference[1:]):
        width=right-left
        signed+=0.5*(d0+d1)*width
        if d0*d1>=0:
            absolute+=0.5*(abs(d0)+abs(d1))*width
        else:
            crossing=width*abs(d0)/(abs(d0)+abs(d1))
            absolute+=0.5*(abs(d0)*crossing+abs(d1)*(width-crossing))
    return {"signed":signed,"absolute":absolute,"grid_points":len(grid),
            "turning_index":peak,"input_min":low,"input_max":high,
            "output_gap":output[-1]-output[0],
            "estimator":"piecewise_linear_common_input_grid"}
