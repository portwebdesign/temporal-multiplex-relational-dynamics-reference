"""Independent balance, local-linearization and equivalence checks."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
from math import exp, inf, nan
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import STATE_NAMES, derivative, simulate  # noqa: E402
from network import NetworkSignals, aggregate_network  # noqa: E402
from referee_experiments import core_load_benchmark, matched_direction_sensitivity  # noqa: E402
from regimes import (analytic_jacobian, characteristic_coefficients,  # noqa: E402
                     construct_interior_parameters, finite_difference_jacobian,
                     local_stability_record, routh_hurwitz_record)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "illustrative_baseline.json"
TARGET = [0.60, 0.35, 0.60, 0.45, 0.60, 0.60, 0.35]
INPUTS = {"delay": 0.20, "boundary": 0.35, "review": 0.30}
SOLVED = {
    "a_s": 0.139118585,
    "a_e": 0.18792511238095244,
    "eta_r": 0.32217109710753783,
    "rho_p": 0.5909420289855072,
    "d_w1": 0.6603856035714287,
    "h_w2": 1.1985502100840337,
}


def parameters() -> dict:
    result = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["parameters"]
    result.update({"a_conflict": 0.30, "lambda_d": 0.20, "lambda_h": 0.15})
    return result


def network(opposed: bool = False, magnitude: float = 0.15) -> dict:
    return {
        "dimension": 1,
        "layer_scales": [1.0, 1.0, 1.0],
        "support_scale": 1.0,
        "psi_scale": 1.0,
        "coupling": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.2], [0.0, 0.2, 0.0]],
        "demand_edges": [
            {"layer": 1, "weight": 1.0, "magnitude": magnitude, "direction": [1.0]},
            {"layer": 2, "weight": 1.0, "magnitude": magnitude,
             "direction": [-1.0] if opposed else [1.0]},
        ],
        "support_edges": [{"weight": 1.0, "support": 1.0}],
    }


class InteriorBalanceTests(unittest.TestCase):
    def test_constructed_coefficients_match_independent_balance_values(self) -> None:
        record = construct_interior_parameters(parameters(), INPUTS, network(), TARGET)
        for name, expected in SOLVED.items():
            self.assertAlmostEqual(record["parameters"][name], expected, places=13)
        rates = derivative(TARGET, INPUTS, record["parameters"], aggregate_network(network()))
        self.assertLess(max(abs(value) for value in rates), 1.0e-14)
        self.assertLess(record["residual"], 1.0e-14)

    def test_balance_construction_preserves_inputs_and_other_coefficients(self) -> None:
        original = parameters()
        par, inputs, spec, target = deepcopy(original), deepcopy(INPUTS), network(), TARGET[:]
        saved = deepcopy((par, inputs, spec, target))
        record = construct_interior_parameters(par, inputs, spec, target)
        self.assertEqual((par, inputs, spec, target), saved)
        for name in set(original) - set(SOLVED):
            self.assertEqual(record["parameters"][name], original[name], name)
        self.assertEqual(set(record["solved_coefficients"]), set(SOLVED))

    def test_network_and_signal_inputs_give_the_same_balance(self) -> None:
        by_network = construct_interior_parameters(parameters(), INPUTS, network(), TARGET)
        by_signal = construct_interior_parameters(parameters(), INPUTS, aggregate_network(network()), TARGET)
        self.assertEqual(by_network["parameters"], by_signal["parameters"])

    def test_targets_must_be_finite_and_strictly_interior(self) -> None:
        for value in (0.0, 1.0, -0.01, 1.01, nan, inf):
            target = TARGET[:]
            target[0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                construct_interior_parameters(parameters(), INPUTS, network(), target)

    def test_zero_identifying_inputs_and_infeasible_balances_are_rejected(self) -> None:
        wrong_target = TARGET[:]
        wrong_target[3] = 0.46
        with self.assertRaises(ValueError):
            construct_interior_parameters(parameters(), INPUTS, network(), wrong_target)
        for name in ("delay", "review"):
            inputs = dict(INPUTS, **{name: 0.0})
            with self.subTest(input=name), self.assertRaises(ValueError):
                construct_interior_parameters(parameters(), inputs, network(), TARGET)
        excessive_gain = parameters()
        excessive_gain["rho_s"] = 100.0
        with self.assertRaises(ValueError):
            construct_interior_parameters(excessive_gain, INPUTS, network(), TARGET)
        zero_pressure = parameters()
        zero_pressure.update({name: 0.0 for name in ("a1", "a2", "a3", "a_conflict")})
        with self.assertRaises(ValueError):
            construct_interior_parameters(zero_pressure, INPUTS, network(), TARGET)


class LocalLinearizationTests(unittest.TestCase):
    def test_exact_constant_resource_gain_and_permeability_jacobian(self) -> None:
        par = {name: 0.0 for name in parameters()}
        par.update({"psi_max": 1.0, "P0": 0.7, "tau_c": 0.2, "c_target": 0.45})
        matrix = analytic_jacobian(TARGET, INPUTS, par)
        for row in range(7):
            for column in range(7):
                expected = -0.7 if row == column == 2 else -0.2 if row == column == 3 else 0.0
                self.assertAlmostEqual(matrix[row][column], expected, places=14)

    def test_analytic_jacobian_matches_finite_differences_in_both_smooth_regions(self) -> None:
        generator = random.Random(20261114)
        signals = aggregate_network(network(opposed=True, magnitude=1.0))
        par = parameters()
        for index in range(12):
            if index == 6:
                par.update({"k_visibility": 0.10, "k_access": 0.08,
                            "k_input": 0.20, "k_isolation": 0.20})
            state = [generator.uniform(0.15, 0.85) for _ in STATE_NAMES]
            state[3] = 0.35 if index % 2 == 0 else 0.65
            analytical = analytic_jacobian(state, INPUTS, par, signals)
            numerical = finite_difference_jacobian(state, INPUTS, par, signals, step=1.0e-5)
            with self.subTest(index=index):
                self.assertLess(max(abs(a - b) for row_a, row_b in zip(analytical, numerical)
                                    for a, b in zip(row_a, row_b)), 2.0e-9)

    def test_saturated_pressure_has_zero_local_pressure_gradient(self) -> None:
        par = parameters()
        par["psi_max"] = 0.10
        state = TARGET[:]
        state[3] = 0.65
        signals = aggregate_network(network(opposed=True, magnitude=1.0))
        analytical = analytic_jacobian(state, INPUTS, par, signals)
        numerical = finite_difference_jacobian(state, INPUTS, par, signals, step=1.0e-5)
        self.assertLess(max(abs(a - b) for row_a, row_b in zip(analytical, numerical)
                            for a, b in zip(row_a, row_b)), 2.0e-9)

    def test_classical_jacobian_is_not_reported_at_declared_kinks(self) -> None:
        par = parameters()
        state = TARGET[:]
        state[3] = par["c_star"]
        with self.assertRaises(ValueError):
            analytic_jacobian(state, INPUTS, par)
        par.update({"a1": 1.0, "a2": 0.0, "a3": 0.0, "a_conflict": 0.0,
                    "q1": 0.0, "psi_max": 0.5})
        state = TARGET[:]
        state[6] = 0.5
        with self.assertRaises(ValueError):
            analytic_jacobian(state, INPUTS, par)

    def test_known_characteristic_polynomials(self) -> None:
        diagonal = [[-0.5, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -2.0]]
        rotation = [[-1.0, -2.0], [2.0, -1.0]]
        for matrix, expected in ((diagonal, [1.0, 3.5, 3.5, 1.0]),
                                 (rotation, [1.0, 2.0, 5.0])):
            actual_coefficients = characteristic_coefficients(matrix)
            self.assertEqual(len(actual_coefficients), len(expected))
            for actual, value in zip(actual_coefficients, expected):
                self.assertAlmostEqual(actual, value, places=13)

    def test_routh_distinguishes_stability_instability_and_imaginary_axis(self) -> None:
        stable = routh_hurwitz_record([1.0, 3.5, 3.5, 1.0])
        self.assertEqual(stable["status"], "strictly_hurwitz")
        self.assertEqual(stable["sign_changes"], 0)
        self.assertTrue(all(value > 0.0 for value in stable["first_column"]))
        unstable = routh_hurwitz_record([1.0, -1.0, -2.0])
        self.assertEqual(unstable["status"], "not_hurwitz")
        self.assertEqual(unstable["sign_changes"], 1)
        imaginary = routh_hurwitz_record([1.0, 0.0, 1.0])
        self.assertEqual(imaginary["status"], "indeterminate_zero_pivot")
        self.assertIsNone(imaginary["sign_changes"])

    def test_constructed_reference_has_checked_local_stability(self) -> None:
        record = construct_interior_parameters(parameters(), INPUTS, network(), TARGET)
        stability = local_stability_record(TARGET, INPUTS, record["parameters"], aggregate_network(network()))
        self.assertEqual(stability["status"], "locally_asymptotically_stable_numerical")
        self.assertLess(stability["residual"], 1.0e-14)
        self.assertLess(stability["finite_difference_max_error"], 2.0e-9)
        self.assertEqual(stability["routh"]["sign_changes"], 0)


class ObservationalEquivalenceTests(unittest.TestCase):
    def test_pressure_unit_rescaling_preserves_the_complete_vector_field(self) -> None:
        generator = random.Random(20261115)
        signals = aggregate_network(network(opposed=True, magnitude=1.0))
        for cap in (1.0, 0.10):
            base = parameters()
            base["psi_max"] = cap
            for scale in (0.25, 2.0, 7.0):
                transformed = deepcopy(base)
                for name in ("a1", "a2", "a3", "a_conflict", "psi_max"):
                    transformed[name] *= scale
                for name in ("xi_s", "mu_e", "eta_r", "gamma_12"):
                    transformed[name] /= scale
                for _ in range(5):
                    state = [generator.uniform(0.05, 0.95) for _ in STATE_NAMES]
                    original = derivative(state, INPUTS, base, signals)
                    changed = derivative(state, INPUTS, transformed, signals)
                    self.assertLess(max(abs(a - b) for a, b in zip(original, changed)), 1.0e-13)

    def test_different_recipients_with_equal_weighted_column_sums_are_equivalent(self) -> None:
        for moderation, alternate_coupling in (
            ([0.0, 0.0, 0.0], [[0.0, 0.2, 0.2], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            ([0.0, 0.5, 0.25], [[0.0, 0.15, 0.10], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        ):
            first = network(opposed=True, magnitude=1.0)
            first["moderation"] = moderation
            second = deepcopy(first)
            second["coupling"] = alternate_coupling
            signals_a, signals_b = aggregate_network(first), aggregate_network(second)
            for name, value in asdict(signals_a).items():
                self.assertAlmostEqual(value, getattr(signals_b, name), places=13)
            rates_a = derivative(TARGET, INPUTS, parameters(), signals_a)
            rates_b = derivative(TARGET, INPUTS, parameters(), signals_b)
            self.assertLess(max(abs(a - b) for a, b in zip(rates_a, rates_b)), 1.0e-13)

    def test_topology_equivalence_requires_the_filter_to_remain_fixed(self) -> None:
        first = network(opposed=True, magnitude=1.0)
        second = deepcopy(first)
        second["coupling"] = [[0.0, 0.2, 0.2], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        first["moderation"] = second["moderation"] = [0.0, 0.5, 0.25]
        self.assertNotAlmostEqual(aggregate_network(first).external_demand,
                                  aggregate_network(second).external_demand)

    def test_external_demand_slope_is_exactly_load_headroom_times_gain(self) -> None:
        for load in (0.35, 0.9042582812432383):
            state = TARGET[:]
            state[6] = load
            low = derivative(state, INPUTS, parameters(), NetworkSignals(external_demand=0.2))
            high = derivative(state, INPUTS, parameters(), NetworkSignals(external_demand=0.7))
            slope = (high[6] - low[6]) / (0.7 - 0.2)
            self.assertAlmostEqual(slope, 0.2 * (1.0 - load), places=14)

    def test_prescribed_constant_target_keeps_permeability_constant_during_demand(self) -> None:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        config["parameters"] = construct_interior_parameters(parameters(), INPUTS, network(), TARGET)["parameters"]
        config["initial_state"] = TARGET[:]
        config["solver"] = {"method": "classical_runge_kutta_4", "dt": 0.1, "time_horizon": 3.0}
        config["inputs"] = {"baseline": dict(INPUTS), "schedule": []}
        config["network"] = {"baseline": network(), "schedule": [
            {"start": 1.0, "end": 2.0, "network": network(opposed=True, magnitude=1.0)}]}
        config["permeability_target"] = {"baseline": 0.45, "schedule": []}
        rows = simulate(config)
        self.assertTrue(all(row["c"] == 0.45 for row in rows))
        self.assertGreater(max(abs(row["r"] - TARGET[2]) for row in rows), 0.001)


class ExactCoreLoadTests(unittest.TestCase):
    def configuration(self) -> dict:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        config["parameters"] = construct_interior_parameters(parameters(), INPUTS, network(), TARGET)["parameters"]
        config["initial_state"] = TARGET[:]
        config["initial_state"][6] = 0.20
        config["active_coordinates"] = ["r", "w2"]
        config["solver"] = {"method": "classical_runge_kutta_4", "dt": 0.02, "time_horizon": 0.10}
        config["inputs"] = {"baseline": dict(INPUTS), "schedule": []}
        config["network"] = {"baseline": network(), "schedule": []}
        config["permeability_target"] = {"baseline": 0.45, "schedule": []}
        return config

    def scalar_rates(self, config: dict, external_demand: float) -> tuple[float, float]:
        s, e, _r, c, p, _w1, _w2 = config["initial_state"]
        par = config["parameters"]
        gain = (par["lambda_d"] * external_demand
                + par["a_w2"] * c * (1.0 - par["q_h"] * INPUTS["boundary"])
                + par["b_w2"] * e)
        loss = par["d_w2"] * s * p * INPUTS["boundary"] + par["h_w2"] * INPUTS["delay"]
        return gain, loss

    @staticmethod
    def exact_segment(initial: float, gain: float, loss: float, duration: float) -> float:
        equilibrium = gain / (gain + loss)
        return equilibrium + (initial - equilibrium) * exp(-(gain + loss) * duration)

    def test_constant_core_load_matches_the_exact_scalar_solution(self) -> None:
        config = self.configuration()
        rows = simulate(config)
        gain, loss = self.scalar_rates(config, 9.0 / 34.0)
        errors = [abs(row["w2"] - self.exact_segment(0.20, gain, loss, row["time"])) for row in rows]
        self.assertLess(max(errors), 1.0e-12)
        benchmark = core_load_benchmark(config, rows)
        self.assertAlmostEqual(benchmark["max_absolute_w2_error"], max(errors), delta=1.0e-15)
        self.assertEqual(benchmark["grid_points"], len(rows))

    def test_off_grid_core_pulse_uses_exact_segments_and_detects_a_corrupted_trace(self) -> None:
        config = self.configuration()
        config["network"]["schedule"] = [{"start": 0.015, "end": 0.055,
                                            "network": network(opposed=True, magnitude=1.0)}]
        rows = simulate(config)
        gain0, loss = self.scalar_rates(config, 9.0 / 34.0)
        gain1, _ = self.scalar_rates(config, 12.0 / 17.0)

        def exact_at(time: float) -> float:
            value = self.exact_segment(0.20, gain0, loss, min(time, 0.015))
            if time > 0.015:
                value = self.exact_segment(value, gain1, loss, min(time, 0.055) - 0.015)
            if time > 0.055:
                value = self.exact_segment(value, gain0, loss, time - 0.055)
            return value

        errors = [abs(row["w2"] - exact_at(row["time"])) for row in rows]
        self.assertLess(max(errors), 1.0e-12)
        benchmark = core_load_benchmark(config, rows)
        self.assertAlmostEqual(benchmark["max_absolute_w2_error"], max(errors), delta=1.0e-15)
        corrupted = deepcopy(rows)
        corrupted[3]["w2"] += 0.01
        detected = core_load_benchmark(config, corrupted)
        self.assertAlmostEqual(detected["max_absolute_w2_error"], 0.01, delta=1.0e-12)

    def test_exact_core_benchmark_rejects_evolving_background_or_target(self) -> None:
        config = self.configuration()
        rows = simulate(config)
        evolving = deepcopy(config)
        evolving["active_coordinates"].append("c")
        with self.assertRaises(ValueError):
            core_load_benchmark(evolving, rows)
        shifted = deepcopy(config)
        shifted["permeability_target"]["schedule"] = [{"start": 0.02, "end": 0.06, "value": 0.35}]
        with self.assertRaises(ValueError):
            core_load_benchmark(shifted, rows)


class DirectionalSensitivityFailureTests(unittest.TestCase):
    def test_uncompleted_direction_conditions_are_retained_without_endpoint_values(self) -> None:
        failures = []
        for parameter, values in (("a_conflict", (0.0, 0.15, 0.30, 0.60)),
                                  ("psi_scale", (0.5, 1.0, 2.0))):
            for index, value in enumerate(values):
                failures.append({"scenario": f"uncompleted_{parameter}_{index}",
                                 "parameter": parameter, "value": value,
                                 "status": "not_completed",
                                 "reason": f"declared stationarity limit for {parameter}={value}"})
        configuration = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        configuration["parameters"] = parameters()
        configuration["network"] = {"baseline": network(), "schedule": []}
        report = {"scenarios": {"aligned": {"r_t10": 0.60, "r_t20": 0.60}},
                  "sensitivity": failures}
        # Failed conditions have no initialized scenario and no endpoint fields.
        rows = matched_direction_sensitivity(report, {}, configuration)
        self.assertEqual(len(rows), 7)
        actual = {row["scenario"]: row for row in rows}
        self.assertEqual(set(actual), {row["scenario"] for row in failures})
        for failure in failures:
            retained = actual[failure["scenario"]]
            for key, expected in failure.items():
                self.assertEqual(retained[key], expected)
            self.assertFalse(any(key.startswith("r_") or key.startswith("opposed_minus_aligned")
                                 for key in retained))


if __name__ == "__main__":
    unittest.main()
