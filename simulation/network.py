"""Source-preserving three-layer focal-network aggregation (0.3.0).

Demand and support are separate nonnegative channels. Directions are declared
unit vectors in a common action space; no observations or inferred motives are
provided by this module. Layer indices are zero based.
"""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from math import isclose, isfinite, sqrt
from typing import Mapping


@dataclass(frozen=True)
class NetworkSignals:
    external_demand: float = 0.0
    support: float = 0.0
    conflict: float = 0.0
    raw_conflict: float = 0.0
    total_demand: float = 0.0
    net_demand: float = 0.0


def _finite(value: object, label: str, positive: bool = False) -> float:
    number = float(value)
    if not isfinite(number) or number < 0 or (positive and number == 0):
        raise ValueError(f"{label} must be finite and {'positive' if positive else 'nonnegative'}")
    return number


def aggregate_network(specification: Mapping[str, object]) -> NetworkSignals:
    """Map an explicitly declared focal network to bounded exogenous signals.

    Q_l=sum_j weight_j*magnitude_j/K_l; Z_l is its directed counterpart.
    Moderated coupling C=I+(I-diag(sigma))Gamma gives v=C Q.
    Each source retains its direction: F_k=(sum_l C_lk) Z_k.
    d_ext=sum(v)/(1+sum(v)); h_net=H/(1+H), where H is a separate
    close-support sum. Psi=(sum ||F_l||)^2-||sum F_l||^2 and
    chi=Psi/(psi_scale+Psi), with a fixed positive psi_scale.
    """
    required = {"dimension", "layer_scales", "support_scale", "psi_scale", "coupling", "demand_edges", "support_edges"}
    if set(specification) - {"moderation"} != required:
        raise ValueError(f"network keys must equal {sorted(required)} with optional moderation")
    dimension = specification["dimension"]
    if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 1:
        raise ValueError("network dimension must be a positive integer")
    scales = specification["layer_scales"]
    coupling = specification["coupling"]
    if len(scales) != 3 or len(coupling) != 3 or any(len(row) != 3 for row in coupling):
        raise ValueError("three scales and a 3 by 3 coupling matrix are required")
    scales = [_finite(value, "layer scale", True) for value in scales]
    support_scale = _finite(specification["support_scale"], "support scale", True)
    psi_scale = _finite(specification["psi_scale"], "psi scale", True)
    moderation=specification.get("moderation",[0.0]*3)
    if len(moderation)!=3:
        raise ValueError("moderation must contain three unit-interval values")
    moderation=[_finite(value,"moderation") for value in moderation]
    if any(value>1 for value in moderation):
        raise ValueError("moderation must lie in [0,1]")
    matrix = [[(1.0-moderation[i])*_finite(value, "coupling") + float(i == j) for j, value in enumerate(row)] for i, row in enumerate(coupling)]
    q = [0.0] * 3
    z = [[0.0] * dimension for _ in range(3)]
    for edge in specification["demand_edges"]:
        if set(edge) != {"layer", "weight", "magnitude", "direction"}:
            raise ValueError("demand edge keys must be layer, weight, magnitude, direction")
        layer = edge["layer"]
        if not isinstance(layer, int) or isinstance(layer, bool) or not 0 <= layer < 3:
            raise ValueError("demand layer must be 0, 1, or 2")
        weight = _finite(edge["weight"], "edge weight")
        magnitude = _finite(edge["magnitude"], "edge magnitude")
        direction = [float(value) for value in edge["direction"]]
        if len(direction) != dimension or any(not isfinite(value) for value in direction):
            raise ValueError("direction must contain dimension finite components")
        if not isclose(sum(value * value for value in direction), 1.0, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError("each declared direction must be a unit vector")
        amount = weight * magnitude / scales[layer]
        q[layer] += amount
        for axis in range(dimension):
            z[layer][axis] += amount * direction[axis]
    v = [sum(matrix[i][j] * q[j] for j in range(3)) for i in range(3)]
    amplifiers=[sum(matrix[i][j] for i in range(3)) for j in range(3)]
    forces = [[amplifiers[k]*value for value in z[k]] for k in range(3)]
    total = sum(sqrt(sum(value * value for value in force)) for force in forces)
    net = sqrt(sum(sum(force[axis] for force in forces) ** 2 for axis in range(dimension)))
    # Round-off at aligned vectors must not produce a negative diagnostic.
    raw_conflict = max(0.0, total * total - net * net)
    demand = sum(v)
    support = 0.0
    for edge in specification["support_edges"]:
        if set(edge) != {"weight", "support"}:
            raise ValueError("support edge keys must be weight and support")
        support += _finite(edge["weight"], "support weight") * _finite(edge["support"], "support mark") / support_scale
    if any(not isfinite(value) for value in (raw_conflict, demand, support, total, net)):
        raise ValueError("network aggregation overflowed; rescale the declared units")
    return NetworkSignals(demand / (1.0 + demand), support / (1.0 + support),
                          raw_conflict / (psi_scale + raw_conflict), raw_conflict, total, net)


def validate_network_interpolation(start: Mapping[str,object], end: Mapping[str,object]) -> None:
    """Require fixed source topology and directions during a continuous ramp."""
    aggregate_network(start)
    aggregate_network(end)
    fixed=("dimension","layer_scales","support_scale","psi_scale","coupling","support_edges")
    if any(start[name]!=end[name] for name in fixed) or start.get("moderation",[0.0]*3)!=end.get("moderation",[0.0]*3):
        raise ValueError("linear network ramps require fixed scales, coupling, moderation and support")
    if len(start["demand_edges"])!=len(end["demand_edges"]):
        raise ValueError("linear network ramps require the same ordered demand edges")
    for left,right in zip(start["demand_edges"],end["demand_edges"]):
        if left["layer"]!=right["layer"] or left["direction"]!=right["direction"]:
            raise ValueError("linear network ramps require fixed edge layers and directions")


def interpolate_network(start: Mapping[str,object], end: Mapping[str,object], fraction: float,
                        *, validated: bool=False) -> NetworkSignals:
    """Linearly vary demand weights and magnitudes on a fixed network topology."""
    if not isfinite(fraction) or not 0<=fraction<=1:
        raise ValueError("network interpolation fraction must lie in [0,1]")
    if not validated:
        validate_network_interpolation(start,end)
    interpolated=deepcopy(start)
    for edge,left,right in zip(interpolated["demand_edges"],start["demand_edges"],end["demand_edges"]):
        for name in ("weight","magnitude"):
            edge[name]=(1.0-fraction)*float(left[name])+fraction*float(right[name])
    return aggregate_network(interpolated)


def network_at_time(specification: Mapping[str, object] | None, time: float) -> NetworkSignals:
    """Evaluate a constant or linearly ramped network on half-open intervals."""
    if specification is None:
        return NetworkSignals()
    if set(specification) != {"baseline", "schedule"}:
        raise ValueError("network schedule must contain baseline and schedule")
    aggregate_network(specification["baseline"])
    selected = specification["baseline"]
    selected_signals=None
    previous_end = -float("inf")
    for interval in sorted(specification["schedule"], key=lambda item: item["start"]):
        if set(interval)-{"end_network"} != {"start", "end", "network"}:
            raise ValueError("network interval must contain start, end, network and optional end_network")
        start = _finite(interval["start"], "interval start")
        end = _finite(interval["end"], "interval end")
        if end <= start or start < previous_end:
            raise ValueError("network intervals must be ordered, nonoverlapping and have end > start")
        previous_end = end
        # Validate inactive intervals too: invalid assumptions never hide in a schedule.
        aggregate_network(interval["network"])
        if "end_network" in interval:
            validate_network_interpolation(interval["network"],interval["end_network"])
        if start <= time < end:
            if "end_network" in interval:
                selected_signals=interpolate_network(interval["network"],interval["end_network"],(time-start)/(end-start),validated=True)
            else:
                selected = interval["network"]
                selected_signals=None
    if selected_signals is not None:
        return selected_signals
    return aggregate_network(selected)
