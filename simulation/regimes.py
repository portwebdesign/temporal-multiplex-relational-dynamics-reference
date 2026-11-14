"""Balance-constructed stationary regimes and local linear diagnostics.

The construction solves six specified equilibrium equations at a supplied
interior reference. It uses no pulse trajectory or outcome objective. A solved
balance need not be stable or unique; branch initialization and local checks
are separate parts of the experiment. All calculations use the standard
library. The Routh table is a floating-point diagnostic, not an interval proof.
"""

from __future__ import annotations

from math import isfinite
from typing import Mapping, Sequence

from model import (STATE_NAMES, derivative, diagnostics, validate_inputs,
                   validate_parameters, validate_signals, validate_state)
from network import NetworkSignals, aggregate_network


SOLVED_COEFFICIENTS = ("a_s", "a_e", "eta_r", "rho_p", "d_w1", "h_w2")


def _signals(value: Mapping[str, object] | NetworkSignals | None) -> NetworkSignals:
    signals = (value if isinstance(value, NetworkSignals) else
               NetworkSignals() if value is None else aggregate_network(value))
    validate_signals(signals)
    return signals


def construct_interior_parameters(
    base_parameters: Mapping[str, float], inputs: Mapping[str, float],
    network_or_signals: Mapping[str, object] | NetworkSignals,
    target: Sequence[float],
) -> dict:
    """Solve for a_s, a_e, eta_r, rho_p, d_w1 and h_w2 at ``target``.

    Every other coefficient is held fixed. The reference must be strictly
    interior, its c coordinate must equal the existing permeability target,
    and baseline delay/review inputs and the needed denominators must be
    positive. Negative solved coefficients reject an infeasible construction.
    No clipping, parameter search, or response-based selection is performed.
    """
    par = validate_parameters(base_parameters)
    u = validate_inputs(inputs)
    x = validate_state(target)
    if any(value <= 0.0 or value >= 1.0 for value in x):
        raise ValueError("the constructed reference must be strictly interior")
    s, e, r, c, p, w1, w2 = x
    if abs(c - par["c_target"]) > 1e-14:
        raise ValueError("reference c must equal the unchanged baseline c_target")
    if u["review"] <= 0.0 or u["delay"] <= 0.0:
        raise ValueError("positive baseline review and delay inputs are required")
    signals = _signals(network_or_signals)
    proxy = diagnostics(x, u, par, signals)
    psi, exposure = proxy.pressure_proxy, proxy.exposure_proxy
    purpose_basis = s * (par["a_p"] + par["b_p"] * w1 * u["review"]) * (1.0 - p)
    if psi <= 0.0 or purpose_basis <= 0.0:
        raise ValueError("positive pressure and purpose-gain basis are required")

    solved = {}
    s_loss = par["nu_s"] * e + par["xi_s"] * c * psi * (1.0 - par["q_s"] * u["delay"])
    solved["a_s"] = (s_loss * s / (1.0 - s) - par["rho_s"] * r * u["review"]) / p
    e_gain = (par["mu_e"] * (1.0 - s) * (1.0 - p) * psi
              + par["omega_e"] * w2 * (1.0 - par["q_e"] * u["delay"]))
    solved["a_e"] = (e_gain * (1.0 - e) / e - par["kappa_e"] * s * p) / u["review"]
    r_gain = (par["P0"] + par["alpha_r"] * s * p + par["a_r"] * w1
              + par["k_visibility"] * e + par["k_access"] * c * signals.support)
    r_other_loss = (par["gamma_r"] * e * e + par["theta_r"] * c * exposure
                    + par["k_input"] * sum(value * value for value in u.values()))
    solved["eta_r"] = (r_gain * (1.0 - r) / r - r_other_loss) / psi
    p_loss = par["delta_p"] * e * e + par["zeta_p"] * w2 * (1.0 - s) * (1.0 - u["delay"])
    solved["rho_p"] = p_loss * p / purpose_basis
    close_gain = (par["a_w1"] * r * p + par["b_w1"] * s * (1.0 - e)
                  + par["lambda_h"] * signals.support)
    close_other_loss = (par["gamma_12"] * w2 * psi * (1.0 - par["q_g"] * u["boundary"])
                        + par["k_isolation"] * (1.0 - c) * signals.support)
    solved["d_w1"] = (close_gain * (1.0 - w1) / w1 - close_other_loss) / (e * (1.0 - s))
    load_gain = (par["lambda_d"] * signals.external_demand
                 + par["a_w2"] * c * (1.0 - par["q_h"] * u["boundary"])
                 + par["b_w2"] * e)
    solved["h_w2"] = (load_gain * (1.0 - w2) / w2
                       - par["d_w2"] * s * p * u["boundary"]) / u["delay"]
    if any(not isfinite(value) or value < 0.0 for value in solved.values()):
        raise ValueError("the specified reference requires an inadmissible solved coefficient")
    par.update(solved)
    par = validate_parameters(par)
    residual = max(abs(value) for value in derivative(x, u, par, signals))
    if residual > 1e-10:
        raise ValueError("constructed balance exceeds the numerical residual tolerance")
    return {
        "parameters": par, "target": list(x), "solved_coefficients": solved,
        "residual": residual,
        "unchanged_parameters": {key: value for key, value in par.items() if key not in solved},
        "method": "six_explicit_stationary_balance_equations",
        "construction_uses_trajectory_outputs": False,
        "interpretation": "Specified model-coordinate reference; stability and other stationary branches are checked separately.",
    }


def analytic_jacobian(
    state: Sequence[float], inputs: Mapping[str, float], parameters: Mapping[str, float],
    signals: NetworkSignals | None = None, permeability_target: float | None = None,
) -> list[list[float]]:
    """Differentiate the Appendix gain/loss equations on a smooth branch.

    Exogenous inputs, network signals and permeability target are fixed.
    A nontrivial positive-part or clipping kink is rejected rather than
    assigned an arbitrary classical derivative.
    """
    s, e, r, c, p, w1, w2 = x = validate_state(state)
    u, par, signals = validate_inputs(inputs), validate_parameters(parameters), _signals(signals)
    # Also validate an optional target override through the vector field.
    derivative(x, u, par, signals, permeability_target)
    excess = max(0.0, c - par["c_star"])
    kink_distance = abs(c - par["c_star"])
    if kink_distance <= 1e-12 and (par["a3"] != 0.0 or par["b1"] != 0.0):
        raise ValueError("classical Jacobian is undefined at the permeability positive-part kink")
    active_excess = float(c > par["c_star"])
    raw_pressure = (par["a1"] * w2 * (1.0 - par["q1"] * u["delay"])
                    + par["a2"] * (1.0 - s * p) + par["a3"] * (e + excess)
                    + par["a_conflict"] * signals.conflict)
    dp = [-par["a2"] * p, par["a3"], 0.0, par["a3"] * active_excess,
          -par["a2"] * s, 0.0, par["a1"] * (1.0 - par["q1"] * u["delay"])]
    if ((abs(raw_pressure) <= 1e-12 or abs(raw_pressure - par["psi_max"]) <= 1e-12)
            and any(value != 0.0 for value in dp)):
        raise ValueError("classical Jacobian is undefined at a pressure-clipping kink")
    if raw_pressure < 0.0 or raw_pressure > par["psi_max"]:
        dp = [0.0] * 7
    psi = min(par["psi_max"], max(0.0, raw_pressure))
    access, purpose, boundary = 1.0 + par["b1"] * excess, 1.0 - par["b2"] * s * p, 1.0 - par["b3"] * u["boundary"]
    exposure = w2 * access * purpose * boundary
    dexp = [-w2 * access * par["b2"] * p * boundary, 0.0, 0.0,
            w2 * par["b1"] * active_excess * purpose * boundary,
            -w2 * access * par["b2"] * s * boundary, 0.0, access * purpose * boundary]
    gains = [0.0] * 7
    losses = [0.0] * 7
    dg = [[0.0] * 7 for _ in range(7)]
    dl = [[0.0] * 7 for _ in range(7)]
    gains[0] = par["rho_s"] * r * u["review"] + par["a_s"] * p
    losses[0] = par["nu_s"] * e + par["xi_s"] * c * psi * (1.0 - par["q_s"] * u["delay"])
    dg[0][2], dg[0][4] = par["rho_s"] * u["review"], par["a_s"]
    dl[0] = [par["xi_s"] * (1.0 - par["q_s"] * u["delay"]) * c * value for value in dp]
    dl[0][1] += par["nu_s"]
    dl[0][3] += par["xi_s"] * (1.0 - par["q_s"] * u["delay"]) * psi

    gains[1] = par["mu_e"] * (1.0 - s) * (1.0 - p) * psi + par["omega_e"] * w2 * (1.0 - par["q_e"] * u["delay"])
    losses[1] = par["kappa_e"] * s * p + par["a_e"] * u["review"]
    dg[1] = [par["mu_e"] * (1.0 - s) * (1.0 - p) * value for value in dp]
    dg[1][0] -= par["mu_e"] * (1.0 - p) * psi
    dg[1][4] -= par["mu_e"] * (1.0 - s) * psi
    dg[1][6] += par["omega_e"] * (1.0 - par["q_e"] * u["delay"])
    dl[1][0], dl[1][4] = par["kappa_e"] * p, par["kappa_e"] * s

    gains[2] = (par["P0"] + par["alpha_r"] * s * p + par["a_r"] * w1
                 + par["k_visibility"] * e + par["k_access"] * c * signals.support)
    losses[2] = (par["gamma_r"] * e * e + par["theta_r"] * c * exposure + par["eta_r"] * psi
                  + par["k_input"] * sum(value * value for value in u.values()))
    dg[2] = [par["alpha_r"] * p, par["k_visibility"], 0.0, par["k_access"] * signals.support,
             par["alpha_r"] * s, par["a_r"], 0.0]
    dl[2] = [par["theta_r"] * c * value + par["eta_r"] * gradient for value, gradient in zip(dexp, dp)]
    dl[2][1] += 2.0 * par["gamma_r"] * e
    dl[2][3] += par["theta_r"] * exposure

    target = par["c_target"] if permeability_target is None else float(permeability_target)
    gains[3], losses[3] = par["tau_c"] * target, par["tau_c"] * (1.0 - target)

    gains[4] = par["rho_p"] * s * (par["a_p"] + par["b_p"] * w1 * u["review"])
    losses[4] = par["delta_p"] * e * e + par["zeta_p"] * w2 * (1.0 - s) * (1.0 - u["delay"])
    dg[4][0] = par["rho_p"] * (par["a_p"] + par["b_p"] * w1 * u["review"])
    dg[4][5] = par["rho_p"] * s * par["b_p"] * u["review"]
    dl[4][0], dl[4][1], dl[4][6] = (-par["zeta_p"] * w2 * (1.0 - u["delay"]),
                                    2.0 * par["delta_p"] * e,
                                    par["zeta_p"] * (1.0 - s) * (1.0 - u["delay"]))

    gains[5] = par["a_w1"] * r * p + par["b_w1"] * s * (1.0 - e) + par["lambda_h"] * signals.support
    losses[5] = (par["gamma_12"] * w2 * psi * (1.0 - par["q_g"] * u["boundary"])
                  + par["d_w1"] * e * (1.0 - s) + par["k_isolation"] * (1.0 - c) * signals.support)
    dg[5] = [par["b_w1"] * (1.0 - e), -par["b_w1"] * s, par["a_w1"] * p,
             0.0, par["a_w1"] * r, 0.0, 0.0]
    dl[5] = [par["gamma_12"] * w2 * (1.0 - par["q_g"] * u["boundary"]) * value for value in dp]
    dl[5][0] -= par["d_w1"] * e
    dl[5][1] += par["d_w1"] * (1.0 - s)
    dl[5][3] -= par["k_isolation"] * signals.support
    dl[5][6] += par["gamma_12"] * psi * (1.0 - par["q_g"] * u["boundary"])

    gains[6] = (par["lambda_d"] * signals.external_demand
                 + par["a_w2"] * c * (1.0 - par["q_h"] * u["boundary"]) + par["b_w2"] * e)
    losses[6] = par["d_w2"] * s * p * u["boundary"] + par["h_w2"] * u["delay"]
    dg[6][1], dg[6][3] = par["b_w2"], par["a_w2"] * (1.0 - par["q_h"] * u["boundary"])
    dl[6][0], dl[6][4] = par["d_w2"] * p * u["boundary"], par["d_w2"] * s * u["boundary"]
    return [[(1.0 - x[i]) * dg[i][j] - x[i] * dl[i][j]
             - (gains[i] + losses[i] if i == j else 0.0)
             for j in range(7)] for i in range(7)]


def finite_difference_jacobian(
    state: Sequence[float], inputs: Mapping[str, float], parameters: Mapping[str, float],
    signals: NetworkSignals | None = None, permeability_target: float | None = None,
    *, step: float = 1e-5,
) -> list[list[float]]:
    """Central difference inside the state cube, without boundary clipping."""
    x = validate_state(state)
    if not isfinite(step) or step <= 0.0 or any(value - step < 0.0 or value + step > 1.0 for value in x):
        raise ValueError("central-difference step must be positive and remain inside the state cube")
    columns = []
    for j in range(7):
        lower, upper = list(x), list(x)
        lower[j] -= step
        upper[j] += step
        low = derivative(lower, inputs, parameters, signals, permeability_target)
        high = derivative(upper, inputs, parameters, signals, permeability_target)
        columns.append([(b - a) / (2.0 * step) for a, b in zip(low, high)])
    return [[columns[j][i] for j in range(7)] for i in range(7)]


def characteristic_coefficients(matrix: Sequence[Sequence[float]]) -> list[float]:
    """Monic characteristic polynomial by the Faddeev--LeVerrier recurrence."""
    n = len(matrix)
    if not n or any(len(row) != n for row in matrix):
        raise ValueError("a nonempty square matrix is required")
    a = [[float(value) for value in row] for row in matrix]
    if any(not isfinite(value) for row in a for value in row):
        raise ValueError("matrix entries must be finite")
    b = [[float(i == j) for j in range(n)] for i in range(n)]
    coefficients = [1.0]
    for k in range(1, n + 1):
        product = [[sum(a[i][h] * b[h][j] for h in range(n)) for j in range(n)] for i in range(n)]
        coefficient = -sum(product[i][i] for i in range(n)) / k
        coefficients.append(coefficient)
        b = [[product[i][j] + (coefficient if i == j else 0.0) for j in range(n)] for i in range(n)]
    if any(not isfinite(value) for value in coefficients):
        raise ValueError("characteristic recurrence overflowed")
    return coefficients


def routh_hurwitz_record(coefficients: Sequence[float], *, tolerance: float = 1e-14) -> dict:
    """Classify nondegenerate real polynomials using their Routh first column.

    Zero rows/pivots are returned as indeterminate; no epsilon substitution or
    imaginary-axis classification is hidden in the computation. The supplied
    tolerance is a relative floating-point resolution check.
    """
    values = [float(value) for value in coefficients]
    if len(values) < 2 or any(not isfinite(value) for value in values) or values[0] == 0.0:
        raise ValueError("a finite nonconstant polynomial with nonzero leading coefficient is required")
    if not isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("Routh tolerance must be finite and positive")
    values = [value / values[0] for value in values]
    order, width = len(values) - 1, (len(values) + 1) // 2
    table = [values[::2] + [0.0] * (width - len(values[::2])),
             values[1::2] + [0.0] * (width - len(values[1::2]))]
    threshold = tolerance * max(1.0, *(abs(value) for value in values))
    while len(table) < order + 1:
        previous2, previous = table[-2:]
        if abs(previous[0]) <= threshold:
            break
        table.append([(previous[0] * previous2[j + 1] - previous2[0] * previous[j + 1]) / previous[0]
                      for j in range(width - 1)] + [0.0])
    first = [row[0] for row in table]
    if any(not isfinite(value) for row in table for value in row):
        raise ValueError("Routh recurrence overflowed")
    unresolved = len(table) != order + 1 or any(abs(value) <= threshold for value in first)
    changes = None if unresolved else sum(left * right < 0.0 for left, right in zip(first, first[1:]))
    status = "indeterminate_zero_pivot" if unresolved else "strictly_hurwitz" if changes == 0 else "not_hurwitz"
    return {"status": status, "first_column": first, "sign_changes": changes,
            "table": table, "tolerance": tolerance, "effective_threshold": threshold,
            "method": "floating_point_routh_first_column_without_zero_pivot_substitution"}


def local_stability_record(
    state: Sequence[float], inputs: Mapping[str, float], parameters: Mapping[str, float],
    signals: NetworkSignals | None = None, permeability_target: float | None = None,
) -> dict:
    """Record a smooth-branch Jacobian, independent differences and Routh test."""
    jacobian = analytic_jacobian(state, inputs, parameters, signals, permeability_target)
    coarse = finite_difference_jacobian(state, inputs, parameters, signals, permeability_target, step=1e-5)
    fine = finite_difference_jacobian(state, inputs, parameters, signals, permeability_target, step=5e-6)
    error = max(abs(a - b) for row, other in zip(jacobian, fine) for a, b in zip(row, other))
    refinement = max(abs(a - b) for row, other in zip(coarse, fine) for a, b in zip(row, other))
    coefficients = characteristic_coefficients(jacobian)
    routh = routh_hurwitz_record(coefficients)
    residual = max(abs(value) for value in derivative(state, inputs, parameters, signals, permeability_target))
    status = "local_stability_not_established"
    if routh["status"] == "strictly_hurwitz" and error <= 1e-6:
        status = ("locally_asymptotically_stable_numerical" if residual <= 1e-8 else
                  "jacobian_hurwitz_at_nonstationary_state")
    return {"status": status, "state": list(state), "jacobian": jacobian,
            "finite_difference_steps": [1e-5, 5e-6], "finite_difference_max_error": error,
            "finite_difference_refinement_error": refinement,
            "characteristic_coefficients": coefficients, "routh": routh,
            "residual": residual, "stationarity_classification_tolerance": 1e-8,
            "jacobian_crosscheck_tolerance": 1e-6,
            "interpretation": "Floating-point local linearization at the supplied smooth-branch state; no global stability or exhaustive branch count is established."}
