"""Independent design checks for the additional regulation comparisons."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import simulate
from network import aggregate_network
from regulation_research import construct_conditions, context_configuration, factorial_subsets


class RegulationDesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads((ROOT / "configs/regulation_research_protocol.json").read_text())
        cls.conditions, cls.references, cls.baselines = construct_conditions(cls.protocol)

    def test_all_factorial_subsets_and_own_coefficient_references(self):
        subsets = factorial_subsets(("delay", "boundary", "review"))
        self.assertEqual(len(subsets), 8)
        self.assertEqual(len(set(subsets)), 8)
        self.assertIn((), subsets)
        self.assertIn(("delay", "boundary", "review"), subsets)
        self.assertNotEqual(self.references["0.0"], self.references["0.2"])
        for cost in self.protocol["input_costs"]:
            arms = [item for item in self.conditions if item["family"] == "input_factorial" and item["cost"] == cost]
            self.assertEqual(len(arms), 8)
            self.assertEqual(len({tuple(item["configuration"]["initial_state"]) for item in arms}), 1)

    def test_boundary_support_common_start_and_measured_signal_levels(self):
        arms = [item for item in self.conditions if item["family"] == "boundary_support"]
        self.assertEqual(len(arms), 9)
        self.assertEqual(len({tuple(item["configuration"]["initial_state"]) for item in arms}), 1)
        for item in arms:
            cfg = item["configuration"]
            before = aggregate_network(cfg["network"]["baseline"])
            during = aggregate_network(cfg["network"]["schedule"][0]["network"])
            self.assertAlmostEqual(before.support, .5)
            expected = {0.5: 1/3, 1.0: .5, 2.0: 2/3}[item["support_mark"]]
            self.assertAlmostEqual(during.support, expected)
            # Boundary/support changes do not erase prescribed direct demand.
            self.assertAlmostEqual(during.external_demand, 2.4/3.4)
            self.assertGreater(during.conflict, 0)

    def test_context_retains_all_nondemand_interventions(self):
        for item in self.conditions:
            cfg, context = item["configuration"], item["context"]
            self.assertEqual(cfg["initial_state"], context["initial_state"])
            self.assertEqual(cfg["parameters"], context["parameters"])
            self.assertEqual(cfg["inputs"], context["inputs"])
            self.assertEqual(cfg["permeability_target"], context["permeability_target"])
            baseline = aggregate_network(context["network"]["baseline"])
            context_signal = aggregate_network(context["network"]["schedule"][0]["network"])
            pulse = aggregate_network(cfg["network"]["schedule"][0]["network"])
            self.assertAlmostEqual(context_signal.external_demand, baseline.external_demand)
            self.assertEqual(context_signal.conflict, 0)
            self.assertAlmostEqual(context_signal.support, pulse.support)

    def test_no_demand_difference_returns_identical_trajectory(self):
        # The control operation must be idempotent when demand already equals
        # baseline, even with nontrivial support and target interventions.
        arm = next(item for item in self.conditions if item["family"] == "boundary_support"
                   and item["target"] == .35 and item["support_mark"] == 2)
        cfg = copy.deepcopy(arm["context"])
        cfg["solver"] = {"method": "classical_runge_kutta_4", "dt": .1, "time_horizon": 12}
        self.assertEqual(simulate(cfg), simulate(context_configuration(cfg)))


if __name__ == "__main__":
    unittest.main()
