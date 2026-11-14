"""Mathematical regression checks for source-preserving forcing and dynamics."""

from __future__ import annotations

from copy import deepcopy
import json
from math import e, exp, inf, nan
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import derivative, simulate  # noqa: E402
from network import aggregate_network, network_at_time  # noqa: E402


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "illustrative_baseline.json"


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


def opposed_network(gamma: float) -> dict:
    spec = empty_network()
    spec["coupling"][1][2] = gamma
    spec["coupling"][2][1] = gamma
    spec["demand_edges"] = [
        {"layer": 1, "weight": 1.0, "magnitude": 1.0, "direction": [1.0]},
        {"layer": 2, "weight": 1.0, "magnitude": 1.0, "direction": [-1.0]},
    ]
    return spec


class SourcePreservingAggregationTests(unittest.TestCase):
    def test_opposed_source_amplification_has_exact_conflict_values(self) -> None:
        cases = ((0.0, 4.0, 4.0 / 5.0), (0.2, 5.76, 144.0 / 169.0), (1.0, 16.0, 16.0 / 17.0))
        for gamma, raw, bounded in cases:
            with self.subTest(gamma=gamma):
                signals = aggregate_network(opposed_network(gamma))
                self.assertAlmostEqual(signals.raw_conflict, raw, places=12)
                self.assertAlmostEqual(signals.conflict, bounded, places=12)
                self.assertAlmostEqual(signals.total_demand, 2.0 * (1.0 + gamma))
                self.assertAlmostEqual(signals.net_demand, 0.0)

    def test_one_directed_coupling_preserves_its_source_direction(self) -> None:
        spec = empty_network()
        spec["dimension"] = 2
        spec["demand_edges"] = [
            {"layer": 0, "weight": 1.0, "magnitude": 2.0, "direction": [1.0, 0.0]},
            {"layer": 1, "weight": 1.0, "magnitude": 1.0, "direction": [0.0, 1.0]},
        ]
        spec["coupling"][0][1] = 0.5
        signals = aggregate_network(spec)
        # h_0=1,h_1=1.5: F_0=(2,0), F_1=(0,1.5).
        self.assertAlmostEqual(signals.total_demand, 3.5)
        self.assertAlmostEqual(signals.net_demand, 2.5)
        self.assertAlmostEqual(signals.raw_conflict, 6.0)
        self.assertAlmostEqual(signals.external_demand, 3.5 / 4.5)

    def test_scalar_transmission_mass_equals_source_weighted_mass(self) -> None:
        spec = empty_network()
        spec["coupling"] = [[0.0, 0.2, 0.1], [0.4, 0.0, 0.3], [0.1, 0.5, 0.0]]
        spec["moderation"] = [0.1, 0.6, 0.25]
        q = [2.0, 1.0, 3.0]
        spec["demand_edges"] = [
            {"layer": layer, "weight": 1.0, "magnitude": amount, "direction": [1.0]}
            for layer, amount in enumerate(q)
        ]
        # Independently sum recipient-level transmitted masses.
        recipient_mass = sum(
            q[recipient] + (1.0 - spec["moderation"][recipient]) * sum(
                spec["coupling"][recipient][source] * q[source] for source in range(3)
            ) for recipient in range(3)
        )
        source_mass = sum(
            q[source] * (1.0 + sum(
                (1.0 - spec["moderation"][recipient]) * spec["coupling"][recipient][source]
                for recipient in range(3)
            )) for source in range(3)
        )
        self.assertAlmostEqual(recipient_mass, source_mass)
        signals = aggregate_network(spec)
        self.assertAlmostEqual(signals.external_demand, recipient_mass / (1.0 + recipient_mass))
        self.assertAlmostEqual(signals.total_demand, source_mass)
        self.assertAlmostEqual(signals.raw_conflict, 0.0, places=12)

    def test_nonnegative_coupling_increases_opposed_conflict_and_moderation_reduces_it(self) -> None:
        coupling_results = [aggregate_network(opposed_network(value)) for value in (0.0, 0.1, 0.2, 0.5, 1.0)]
        for left, right in zip(coupling_results, coupling_results[1:]):
            self.assertGreater(right.raw_conflict, left.raw_conflict)
            self.assertGreater(right.external_demand, left.external_demand)
        moderated_results = []
        for value in (0.0, 0.25, 0.5, 0.75, 1.0):
            spec = opposed_network(1.0)
            spec["moderation"] = [value] * 3
            moderated_results.append(aggregate_network(spec))
        for left, right in zip(moderated_results, moderated_results[1:]):
            self.assertLess(right.raw_conflict, left.raw_conflict)
            self.assertLess(right.external_demand, left.external_demand)
        self.assertAlmostEqual(moderated_results[-1].raw_conflict, 4.0)

    def test_upstream_moderation_cannot_filter_away_direct_demand(self) -> None:
        spec = opposed_network(4.0)
        spec["moderation"] = [1.0, 1.0, 1.0]
        signals = aggregate_network(spec)
        self.assertAlmostEqual(signals.external_demand, 2.0 / 3.0)
        self.assertAlmostEqual(signals.raw_conflict, 4.0)

    def test_asymmetric_moderation_acts_on_recipient_rows(self) -> None:
        spec = opposed_network(0.0)
        spec["coupling"][0][1] = 3.0
        spec["moderation"] = [1.0, 0.0, 0.0]
        signals = aggregate_network(spec)
        self.assertAlmostEqual(signals.raw_conflict, 4.0)
        spec["moderation"] = [0.0, 1.0, 0.0]
        signals = aggregate_network(spec)
        # Recipient 0 is unfiltered: source 1 has h=4, source 2 has h=1.
        self.assertAlmostEqual(signals.total_demand, 5.0)
        self.assertAlmostEqual(signals.net_demand, 3.0)
        self.assertAlmostEqual(signals.raw_conflict, 16.0)

    def test_within_layer_cancellation_remains_an_explicit_resolution_limit(self) -> None:
        spec = opposed_network(0.5)
        spec["demand_edges"][1]["layer"] = 1
        signals = aggregate_network(spec)
        self.assertGreater(signals.external_demand, 0.0)
        self.assertEqual(signals.total_demand, 0.0)
        self.assertEqual(signals.raw_conflict, 0.0)

    def test_invalid_moderation_vectors_are_rejected(self) -> None:
        for moderation in ([0.0, 0.0], [0.0, 0.0, 1.01], [-0.1, 0.0, 0.0], [0.0, nan, 0.0], [0.0, inf, 0.0]):
            spec = opposed_network(0.2)
            spec["moderation"] = moderation
            with self.subTest(moderation=moderation), self.assertRaises(ValueError):
                aggregate_network(spec)


class ContinuousForcingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.configuration = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def scalar_configuration(self, active: str, dt: float = 0.02, horizon: float = 1.0) -> dict:
        config = deepcopy(self.configuration)
        config["parameters"] = {name: 0.0 for name in config["parameters"]}
        config["parameters"]["psi_max"] = 1.0
        config["initial_state"] = [0.0] * 7
        config["solver"] = {"method": "classical_runge_kutta_4", "dt": dt, "time_horizon": horizon}
        config["inputs"] = {"baseline": {"delay": 0.0, "boundary": 0.0, "review": 0.0}, "schedule": []}
        config["active_coordinates"] = [active]
        return config

    def test_permeability_target_override_has_exact_derivative(self) -> None:
        config = self.scalar_configuration("c")
        config["parameters"].update({"tau_c": 0.2, "c_target": 0.45})
        config["initial_state"][3] = 0.3
        result = derivative(config["initial_state"], config["inputs"]["baseline"], config["parameters"], permeability_target=0.9)
        self.assertAlmostEqual(result[3], 0.12)

    def test_off_grid_target_pulse_matches_two_exact_relaxation_segments(self) -> None:
        config = self.scalar_configuration("c", horizon=0.1)
        config["initial_state"][3] = 0.45
        config["parameters"].update({"tau_c": 0.2, "c_target": 0.45})
        config["permeability_target"] = {
            "baseline": 0.45,
            "schedule": [{"start": 0.015, "end": 0.055, "value": 0.1}],
        }
        pulse_end = 0.1 + (0.45 - 0.1) * exp(-0.2 * (0.055 - 0.015))
        expected = 0.45 + (pulse_end - 0.45) * exp(-0.2 * (0.1 - 0.055))
        self.assertAlmostEqual(simulate(config)[-1]["c"], expected, places=12)

    def test_continuous_network_interpolates_edge_marks_before_aggregation(self) -> None:
        start = empty_network()
        start["demand_edges"] = [{"layer": 1, "weight": 1.0, "magnitude": 0.0, "direction": [1.0]}]
        end = deepcopy(start)
        end["demand_edges"][0].update({"weight": 2.0, "magnitude": 1.0})
        schedule = {"baseline": empty_network(), "schedule": [{"start": 0.0, "end": 1.0, "network": start, "end_network": end}]}
        # At t=.5: weight=1.5, magnitude=.5, so amount=.75 and d_ext=3/7.
        self.assertAlmostEqual(network_at_time(schedule, 0.5).external_demand, 3.0 / 7.0)

    def test_ramp_forcing_matches_exact_solution_with_fourth_order_refinement(self) -> None:
        start = empty_network()
        start["demand_edges"] = [{"layer": 1, "weight": 1.0, "magnitude": 0.0, "direction": [1.0]}]
        end = deepcopy(start)
        end["demand_edges"][0]["magnitude"] = 1.0
        errors = []
        for dt in (0.2, 0.1, 0.05):
            config = self.scalar_configuration("w2", dt=dt)
            config["parameters"]["lambda_d"] = 1.0
            config["network"] = {
                "baseline": empty_network(),
                "schedule": [{"start": 0.0, "end": 1.0, "network": start, "end_network": end}],
            }
            # w'=[t/(1+t)](1-w), w(0)=0. Integrating the hazard gives 1-2/e.
            error = abs(simulate(config)[-1]["w2"] - (1.0 - 2.0 / e))
            errors.append(error)
        self.assertLess(errors[-1], 1.0e-8)
        for coarse, fine in zip(errors, errors[1:]):
            self.assertGreater(coarse / fine, 10.0)
            self.assertLess(coarse / fine, 22.0)

    def test_ramp_cannot_change_directions_or_coupling_silently(self) -> None:
        start = opposed_network(0.2)
        for field in ("direction", "coupling", "moderation"):
            end = deepcopy(start)
            if field == "direction":
                end["demand_edges"][0]["direction"] = [-1.0]
            elif field == "coupling":
                end["coupling"][0][1] = 0.1
            else:
                end["moderation"] = [0.1, 0.0, 0.0]
            schedule = {"baseline": empty_network(), "schedule": [{"start": 0.0, "end": 1.0, "network": start, "end_network": end}]}
            with self.subTest(field=field), self.assertRaises(ValueError):
                network_at_time(schedule, 0.5)

    def test_invalid_permeability_target_is_rejected(self) -> None:
        for target in (-0.1, 1.1, nan, inf):
            config = self.scalar_configuration("c")
            config["permeability_target"] = {"baseline": target, "schedule": []}
            with self.subTest(target=target), self.assertRaises(ValueError):
                simulate(config)


class AnalysisAgainstExactResultsTests(unittest.TestCase):
    def scalar_equilibrium_configuration(self, seed: float = 0.1) -> dict:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        config["parameters"] = {name: 0.0 for name in config["parameters"]}
        config["parameters"].update({"psi_max": 1.0, "lambda_d": 0.2, "h_w2": 0.1})
        config["initial_state"] = [0.0] * 6 + [seed]
        config["inputs"] = {"baseline": {"delay": 1.0, "boundary": 0.0, "review": 0.0}, "schedule": []}
        spec = empty_network()
        spec["demand_edges"] = [{"layer": 1, "weight": 1.0, "magnitude": 1.0, "direction": [1.0]}]
        config["network"] = {"baseline": spec, "schedule": []}
        config["active_coordinates"] = ["w2"]
        return config

    def test_equilibrium_solver_recovers_exact_balance_from_both_sides(self) -> None:
        from analysis import find_equilibrium

        for seed in (0.1, 0.9):
            with self.subTest(seed=seed):
                result = find_equilibrium(self.scalar_equilibrium_configuration(seed), dt=0.05, tolerance=1e-10)
                self.assertAlmostEqual(result.state[6], 0.5, places=8)
                self.assertLessEqual(result.residual, 1e-10)
                self.assertGreater(result.time, 0.0)
                self.assertGreater(result.steps, 0)

    def test_frozen_coordinate_residual_is_not_a_full_system_equilibrium(self) -> None:
        from analysis import find_equilibrium

        config = self.scalar_equilibrium_configuration()
        config["initial_state"][2] = 0.5
        config["inputs"]["baseline"]["review"] = 1.0
        config["parameters"]["rho_s"] = 1.0
        result = find_equilibrium(config, tolerance=1e-10)
        self.assertAlmostEqual(result.state[6], 0.5, places=8)
        self.assertEqual(result.state[0], 0.0)
        self.assertEqual(result.state[2], 0.5)
        self.assertLessEqual(result.residual, 1e-10)
        self.assertAlmostEqual(result.full_residual, 0.5)

    def test_unconverged_equilibrium_is_reported_as_failure(self) -> None:
        from analysis import find_equilibrium

        with self.assertRaises(RuntimeError):
            find_equilibrium(self.scalar_equilibrium_configuration(), dt=0.05, max_time=0.05, tolerance=1e-10)

    def test_equilibrium_solver_records_step_reduction_for_a_stiff_exact_relaxation(self) -> None:
        from analysis import find_equilibrium

        config = self.scalar_equilibrium_configuration()
        config["parameters"] = {name: 0.0 for name in config["parameters"]}
        config["parameters"].update({"psi_max": 1.0, "tau_c": 100.0, "c_target": 0.5})
        config["initial_state"] = [0.0] * 7
        config["active_coordinates"] = ["c"]
        # c'=100(.5-c) has equilibrium .5. A unit RK4 step leaves the
        # declared domain at an intermediate stage and must be reduced.
        result = find_equilibrium(config, dt=1.0, max_time=5.0, tolerance=1e-10)
        self.assertAlmostEqual(result.state[3], 0.5, places=11)
        self.assertLessEqual(result.residual, 1e-10)
        self.assertGreater(result.step_reductions, 0)
        self.assertLess(result.dt, 1.0)

    def test_deviation_index_is_half_squared_distance_from_declared_reference(self) -> None:
        from analysis import deviation_index

        reference = [0.1] * 7
        state = [0.3, 0.5] + [0.1] * 5
        self.assertAlmostEqual(deviation_index(state, reference), 0.1)
        self.assertEqual(deviation_index(reference, reference), 0.0)

    def test_forward_deviation_uses_declared_lag_and_interpolation_on_irregular_times(self) -> None:
        from analysis import delta_deviation, deviation_trace

        names = ("s", "e", "r", "c", "p", "w1", "w2")
        rows = [
            {"time": time, **dict(zip(names, [value] + [0.0] * 6))}
            for time, value in ((0.0, 0.0), (0.5, 0.5), (2.0, 1.0))
        ]
        self.assertEqual(deviation_trace(rows, [0.0] * 7), [0.0, 0.125, 0.5])
        changes = delta_deviation(rows, [0.0] * 7, lag=0.75)
        self.assertAlmostEqual(changes[0], 0.1875)
        self.assertAlmostEqual(changes[1], 0.1875)
        self.assertIsNone(changes[2])

    def test_forward_deviation_rejects_ambiguous_or_nonfinite_time_grids(self) -> None:
        from analysis import delta_deviation

        names = ("s", "e", "r", "c", "p", "w1", "w2")
        for times in ((0.0, 0.0, 1.0), (0.0, 1.0, 0.5), (0.0, nan, 1.0)):
            rows = [{"time": time, **dict.fromkeys(names, 0.0)} for time in times]
            with self.subTest(times=times), self.assertRaises(ValueError):
                delta_deviation(rows, [0.0] * 7, lag=0.5)

    def test_cycle_signed_lobes_can_cancel_while_absolute_separation_is_positive(self) -> None:
        from analysis import cycle_statistics

        pressure = [0.0, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25, 0.0]
        output = [0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 0.5, 0.0, 0.5]
        result = cycle_statistics(pressure, output)
        self.assertAlmostEqual(result["signed"], 0.0, places=12)
        self.assertAlmostEqual(result["absolute"], 0.25, places=12)
        self.assertEqual(result["turning_index"], 4)
        self.assertAlmostEqual(result["input_min"], 0.0)
        self.assertAlmostEqual(result["input_max"], 1.0)
        self.assertAlmostEqual(result["output_gap"], 0.0)

    def test_cycle_orientation_and_area_agree_with_a_triangular_loop(self) -> None:
        from analysis import cycle_statistics

        result = cycle_statistics([0.0, 0.5, 1.0, 0.5, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0])
        self.assertAlmostEqual(result["signed"], -0.5, places=12)
        self.assertAlmostEqual(result["absolute"], 0.5, places=12)

    def test_absolute_branch_difference_integrates_an_internal_zero_crossing(self) -> None:
        from analysis import cycle_statistics

        result = cycle_statistics([0.0, 0.5, 1.0, 0.5, 0.0], [0.25, 0.25, 0.25, 0.75, 0.0])
        # Branch differences at u=0,.5,1 are .25,-.5,0. Their first
        # segment crosses at u=1/6; integrating |difference| gives 11/48.
        self.assertAlmostEqual(result["signed"], -3.0 / 16.0, places=12)
        self.assertAlmostEqual(result["absolute"], 11.0 / 48.0, places=12)

    def test_cycle_estimator_rejects_steps_unmatched_cycles_and_extra_turns(self) -> None:
        from analysis import cycle_statistics

        invalid = (
            [0.0, 0.0, 1.0, 1.0, 0.0],
            [0.0, 0.5, 1.0, 0.5, 0.1],
            [0.0, 1.0, 0.5, 0.8, 0.0],
            [0.0, 0.5, nan, 0.5, 0.0],
        )
        for pressure in invalid:
            with self.subTest(pressure=pressure), self.assertRaises(ValueError):
                cycle_statistics(pressure, [0.0] * len(pressure))


if __name__ == "__main__":
    unittest.main()
