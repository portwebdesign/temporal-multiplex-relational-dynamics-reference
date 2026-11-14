"""Independent algebraic and numerical checks for the network-driven candidate.

These tests check declared mathematical consequences, not empirical validity.
"""

from __future__ import annotations

from copy import deepcopy
from itertools import product
import json
from math import exp, inf, nan
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import derivative, rk4_step, simulate, validate_parameters  # noqa: E402
from network import aggregate_network  # noqa: E402


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "illustrative_baseline.json"
NEW_PARAMETERS = (
    "a_conflict", "lambda_d", "lambda_h", "k_visibility", "k_access",
    "k_input", "k_isolation",
)


def empty_network() -> dict:
    return {
        "dimension": 1,
        "layer_scales": [1.0, 1.0, 1.0],
        "support_scale": 1.0,
        "psi_scale": 1.0,
        "coupling": [[0.0, 0.0, 0.0] for _ in range(3)],
        "demand_edges": [],
        "support_edges": [],
    }


def matched_network(opposed: bool) -> dict:
    spec = empty_network()
    spec["demand_edges"] = [
        {"layer": 1, "weight": 1.0, "magnitude": 1.0, "direction": [1.0]},
        {"layer": 2, "weight": 1.0, "magnitude": 1.0,
         "direction": [-1.0 if opposed else 1.0]},
    ]
    return spec


class NetworkRevisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.configuration = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        for name in NEW_PARAMETERS:
            self.configuration["parameters"].setdefault(name, 0.0)
        self.parameters = self.configuration["parameters"]
        self.state = self.configuration["initial_state"]
        self.inputs = self.configuration["inputs"]["baseline"]

    def test_matched_demand_changes_conflict_without_changing_external_signal(self) -> None:
        aligned = aggregate_network(matched_network(False))
        opposed = aggregate_network(matched_network(True))
        # Two unit demands: A=2. Aligned B=2; opposed B=0.
        self.assertAlmostEqual(aligned.total_demand, 2.0)
        self.assertAlmostEqual(opposed.total_demand, 2.0)
        self.assertAlmostEqual(aligned.external_demand, 2.0 / 3.0)
        self.assertAlmostEqual(opposed.external_demand, 2.0 / 3.0)
        self.assertAlmostEqual(aligned.net_demand, 2.0)
        self.assertAlmostEqual(opposed.net_demand, 0.0)
        self.assertAlmostEqual(aligned.raw_conflict, 0.0)
        self.assertAlmostEqual(opposed.raw_conflict, 4.0)
        self.assertAlmostEqual(opposed.conflict, 4.0 / 5.0)

    def test_pairwise_identity_in_two_dimensions(self) -> None:
        spec = empty_network()
        spec["dimension"] = 2
        spec["demand_edges"] = [
            {"layer": 0, "weight": 1.0, "magnitude": 2.0, "direction": [1.0, 0.0]},
            {"layer": 1, "weight": 1.0, "magnitude": 3.0, "direction": [0.0, 1.0]},
            {"layer": 2, "weight": 1.0, "magnitude": 1.0, "direction": [-1.0, 0.0]},
        ]
        signals = aggregate_network(spec)
        # Pairwise 2*[2*3*(1-0)+2*1*(1-(-1))+3*1*(1-0)] = 26.
        self.assertAlmostEqual(signals.raw_conflict, 26.0)
        self.assertAlmostEqual(signals.conflict, 26.0 / 27.0)

    def test_cross_layer_coupling_orientation_and_fixed_scales(self) -> None:
        spec = empty_network()
        spec["layer_scales"] = [1.0, 2.0, 1.0]
        spec["coupling"][0][1] = 0.5  # source layer 1 -> recipient layer 0
        spec["demand_edges"] = [
            {"layer": 1, "weight": 2.0, "magnitude": 3.0, "direction": [1.0]},
        ]
        signals = aggregate_network(spec)
        # Q=(0,3,0), (I+Gamma)Q=(1.5,3,0), sum=4.5.
        self.assertAlmostEqual(signals.external_demand, 4.5 / 5.5)
        self.assertAlmostEqual(signals.total_demand, 4.5)
        self.assertAlmostEqual(signals.raw_conflict, 0.0)

    def test_zero_network_is_well_defined(self) -> None:
        signals = aggregate_network(empty_network())
        for name in ("external_demand", "support", "conflict", "raw_conflict", "total_demand", "net_demand"):
            self.assertEqual(getattr(signals, name), 0.0)

    def test_layer_aggregation_keeps_external_magnitude_but_can_erase_within_layer_conflict(self) -> None:
        spec = matched_network(True)
        spec["demand_edges"][1]["layer"] = 1
        signals = aggregate_network(spec)
        # This intentionally records the resolution limit: a layer-level
        # resultant cannot recover directional disagreement within that layer.
        self.assertAlmostEqual(signals.external_demand, 2.0 / 3.0)
        self.assertAlmostEqual(signals.net_demand, 0.0)
        self.assertAlmostEqual(signals.raw_conflict, 0.0)

    def test_external_demand_remains_a_drive_when_permeability_is_zero(self) -> None:
        state = [0.0] * 7
        par = {name: 0.0 for name in self.parameters}
        par.update({"psi_max": 1.0, "lambda_d": 1.0})
        signals = aggregate_network(matched_network(False))
        rates = derivative(state, self.inputs, par, signals=signals)
        self.assertAlmostEqual(rates[6], 2.0 / 3.0)
        self.assertEqual(state[3], 0.0)

    def test_full_load_trace_respects_external_demand_lower_envelope(self) -> None:
        config = deepcopy(self.configuration)
        config["parameters"]["lambda_d"] = 0.2
        config["initial_state"][1] = 0.0
        config["initial_state"][3] = 0.0
        spec = empty_network()
        spec["demand_edges"] = [
            {"layer": 1, "weight": 1.0, "magnitude": 1.0, "direction": [1.0]},
        ]
        config["network"] = {"baseline": spec, "schedule": []}
        # d_ext=.5 throughout. Gain >= lambda_d*d_ext; loss <= d_w2+h_w2.
        a0 = 0.2 * 0.5
        maximum_loss = config["parameters"]["d_w2"] + config["parameters"]["h_w2"]
        rate = a0 + maximum_loss
        equilibrium = a0 / rate
        initial_load = config["initial_state"][6]
        for row in simulate(config):
            envelope = equilibrium + (initial_load - equilibrium) * exp(-rate * row["time"])
            with self.subTest(time=row["time"]):
                self.assertGreaterEqual(row["w2"] + 2e-10, envelope)

    def test_external_demand_lower_envelope_is_tight_for_scalar_specialization(self) -> None:
        config = deepcopy(self.configuration)
        config["parameters"] = {name: 0.0 for name in self.parameters}
        config["parameters"].update({"psi_max": 1.0, "lambda_d": 0.2, "h_w2": 0.1})
        config["initial_state"] = [0.0] * 6 + [0.4]
        config["inputs"] = {
            "baseline": {"delay": 1.0, "boundary": 0.0, "review": 0.0},
            "schedule": [],
        }
        spec = empty_network()
        spec["demand_edges"] = [
            {"layer": 1, "weight": 1.0, "magnitude": 1.0, "direction": [1.0]},
        ]
        config["network"] = {"baseline": spec, "schedule": []}
        # Exactly w2'=.1(1-w2)-.1*w2, so the envelope is the solution.
        for row in simulate(config):
            expected = 0.5 + (0.4 - 0.5) * exp(-0.2 * row["time"])
            with self.subTest(time=row["time"]):
                self.assertAlmostEqual(row["w2"], expected, places=11)

    def test_directional_path_has_exact_local_resource_difference(self) -> None:
        par = dict(self.parameters, a_conflict=0.25)
        aligned = aggregate_network(matched_network(False))
        opposed = aggregate_network(matched_network(True))
        # Baseline values keep the pressure proxy below its clipping ceiling.
        left = derivative(self.state, self.inputs, par, signals=aligned)
        right = derivative(self.state, self.inputs, par, signals=opposed)
        expected = -par["eta_r"] * par["a_conflict"] * 0.8 * self.state[2]
        self.assertAlmostEqual(right[2] - left[2], expected, places=12)

    def test_conflict_ablation_makes_matched_scenarios_identical_locally(self) -> None:
        par = dict(self.parameters, a_conflict=0.0, lambda_d=0.3)
        self.assertEqual(
            derivative(self.state, self.inputs, par, signals=aggregate_network(matched_network(False))),
            derivative(self.state, self.inputs, par, signals=aggregate_network(matched_network(True))),
        )

    def test_zero_new_couplings_recover_legacy_derivative_even_with_network(self) -> None:
        spec = matched_network(True)
        spec["support_edges"] = [{"weight": 1.0, "support": 2.0}]
        par = dict(self.parameters, **{name: 0.0 for name in NEW_PARAMETERS})
        self.assertEqual(
            derivative(self.state, self.inputs, par),
            derivative(self.state, self.inputs, par, signals=aggregate_network(spec)),
        )

    def test_support_access_and_isolation_cost_have_opposite_exact_channels(self) -> None:
        spec = empty_network()
        spec["support_edges"] = [{"weight": 1.0, "support": 1.0}]
        signals = aggregate_network(spec)
        base = derivative(self.state, self.inputs, self.parameters, signals=signals)
        par = dict(self.parameters, k_access=0.4, k_isolation=0.3)
        changed = derivative(self.state, self.inputs, par, signals=signals)
        self.assertAlmostEqual(
            changed[2] - base[2],
            0.4 * self.state[3] * 0.5 * (1.0 - self.state[2]), places=12,
        )
        self.assertAlmostEqual(
            changed[5] - base[5],
            -0.3 * (1.0 - self.state[3]) * 0.5 * self.state[5], places=12,
        )

    def test_all_cube_corners_have_inward_derivatives_with_tradeoffs(self) -> None:
        par = dict(self.parameters)
        par.update({name: 0.3 for name in NEW_PARAMETERS})
        spec = matched_network(True)
        spec["support_edges"] = [{"weight": 1.0, "support": 2.0}]
        signals = aggregate_network(spec)
        for state in product((0.0, 1.0), repeat=7):
            rates = derivative(state, self.inputs, par, signals=signals)
            for coordinate, rate in zip(state, rates):
                with self.subTest(state=state, coordinate=coordinate, rate=rate):
                    if coordinate == 0.0:
                        self.assertGreaterEqual(rate, -1e-14)
                    else:
                        self.assertLessEqual(rate, 1e-14)

    def test_input_cost_has_exact_negative_resource_contribution(self) -> None:
        signals = aggregate_network(empty_network())
        without = derivative(self.state, self.inputs, self.parameters, signals=signals)
        cost = 0.6
        with_cost = derivative(self.state, self.inputs, dict(self.parameters, k_input=cost), signals=signals)
        expected = -cost * sum(value**2 for value in self.inputs.values()) * self.state[2]
        self.assertAlmostEqual(with_cost[2] - without[2], expected, places=12)

    def test_visibility_can_have_opposite_resource_sign_in_countermodel(self) -> None:
        par = dict(self.parameters)
        par.update({"gamma_r": 0.1, "a3": 0.0, "k_visibility": 0.0})
        low = list(self.state)
        high = list(self.state)
        low[1], high[1] = 0.2, 0.3
        self.assertLess(derivative(high, self.inputs, par)[2], derivative(low, self.inputs, par)[2])
        par["k_visibility"] = 1.0
        self.assertGreater(derivative(high, self.inputs, par)[2], derivative(low, self.inputs, par)[2])

    def test_off_grid_network_pulse_matches_exact_scalar_solution(self) -> None:
        config = deepcopy(self.configuration)
        config["solver"] = {"method": "classical_runge_kutta_4", "dt": 0.02, "time_horizon": 0.1}
        config["initial_state"] = [0.0] * 7
        config["parameters"] = {name: 0.0 for name in self.parameters}
        config["parameters"].update({"psi_max": 1.0, "lambda_d": 1.0})
        pulse = empty_network()
        pulse["demand_edges"] = [{"layer": 1, "weight": 1.0, "magnitude": 1.0, "direction": [1.0]}]
        config["network"] = {
            "baseline": empty_network(),
            "schedule": [{"start": 0.015, "end": 0.055, "network": pulse}],
        }
        rows = simulate(config)
        expected = 1.0 - exp(-0.5 * (0.055 - 0.015))
        self.assertAlmostEqual(rows[-1]["w2"], expected, places=10)

    def test_nonfinite_and_out_of_domain_parameters_are_rejected(self) -> None:
        for name, value in (("a1", nan), ("lambda_d", inf), ("rho_s", -inf), ("c_star", 1.01)):
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                validate_parameters(dict(self.parameters, **{name: value}))

    def test_off_grid_input_pulse_matches_exact_scalar_solution(self) -> None:
        config = deepcopy(self.configuration)
        config["solver"] = {"method": "classical_runge_kutta_4", "dt": 0.02, "time_horizon": 0.1}
        config["initial_state"] = [0.0] * 6 + [0.5]
        config["parameters"] = {name: 0.0 for name in self.parameters}
        config["parameters"].update({"psi_max": 1.0, "h_w2": 1.0})
        config["inputs"] = {
            "baseline": {"delay": 0.0, "boundary": 0.0, "review": 0.0},
            "schedule": [{"start": 0.015, "end": 0.055, "values": {"delay": 1.0}}],
        }
        rows = simulate(config)
        self.assertAlmostEqual(rows[-1]["w2"], 0.5 * exp(-0.04), places=10)

    def test_invalid_network_values_are_rejected(self) -> None:
        for change in ("direction", "negative", "nonfinite", "scale", "coupling"):
            spec = matched_network(False)
            if change == "direction":
                spec["demand_edges"][0]["direction"] = [2.0]
            elif change == "negative":
                spec["demand_edges"][0]["weight"] = -1.0
            elif change == "nonfinite":
                spec["demand_edges"][0]["magnitude"] = inf
            elif change == "scale":
                spec["psi_scale"] = 0.0
            else:
                spec["coupling"][0][1] = -0.1
            with self.subTest(change=change), self.assertRaises(ValueError):
                aggregate_network(spec)

    def test_overlapping_network_intervals_are_rejected(self) -> None:
        self.configuration["network"] = {
            "baseline": empty_network(),
            "schedule": [
                {"start": 1.0, "end": 3.0, "network": matched_network(False)},
                {"start": 2.0, "end": 4.0, "network": matched_network(True)},
            ],
        }
        with self.assertRaises(ValueError):
            simulate(self.configuration)

    def test_invalid_baseline_cannot_hide_under_full_horizon_override(self) -> None:
        invalid = empty_network()
        invalid["psi_scale"] = 0.0
        self.configuration["network"] = {
            "baseline": invalid,
            "schedule": [{"start": 0.0, "end": 21.0, "network": matched_network(False)}],
        }
        with self.assertRaises(ValueError):
            simulate(self.configuration)

    def test_roundoff_handling_preserves_a_legitimate_near_boundary_state(self) -> None:
        state = [1.0 - 5e-10] * 7
        # A zero-rate vector field has the exact solution x(t)=x(0).
        par = {name: 0.0 for name in self.parameters}
        par["psi_max"] = 1.0
        self.assertEqual(rk4_step(state, 0.1, self.inputs, par), tuple(state))

    def test_solver_rejects_nonfinite_steps_and_unimplemented_method(self) -> None:
        for dt in (nan, inf, -inf):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                rk4_step(self.state, dt, self.inputs, self.parameters)
        self.configuration["solver"]["method"] = "unimplemented"
        with self.assertRaises(ValueError):
            simulate(self.configuration)


if __name__ == "__main__":
    unittest.main()
