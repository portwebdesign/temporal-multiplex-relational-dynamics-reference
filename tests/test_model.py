"""Regression tests for the reference Appendix A implementation."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import derivative, diagnostics, simulate  # noqa: E402
from sensitivity import parse_values  # noqa: E402


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "illustrative_baseline.json"


class AppendixAModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.configuration = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_proxy_respects_declared_pressure_bound(self) -> None:
        result = diagnostics(
            self.configuration["initial_state"],
            self.configuration["inputs"]["baseline"],
            self.configuration["parameters"],
        )
        self.assertGreaterEqual(result.pressure_proxy, 0.0)
        self.assertLessEqual(result.pressure_proxy, self.configuration["parameters"]["psi_max"])
        self.assertGreaterEqual(result.exposure_proxy, 0.0)

    def test_baseline_trajectory_stays_in_declared_domain(self) -> None:
        rows = simulate(self.configuration)
        self.assertEqual(len(rows), 1001)
        for row in rows:
            for coordinate in ("s", "e", "r", "c", "p", "w1", "w2"):
                self.assertGreaterEqual(row[coordinate], 0.0)
                self.assertLessEqual(row[coordinate], 1.0)

    def test_unknown_parameter_is_rejected(self) -> None:
        invalid = json.loads(json.dumps(self.configuration))
        invalid["parameters"]["unjustified_extra_term"] = 0.1
        with self.assertRaises(ValueError):
            simulate(invalid)

    def test_threshold_parameters_respect_declared_unit_interval(self) -> None:
        invalid = json.loads(json.dumps(self.configuration))
        invalid["parameters"]["c_star"] = 1.01
        with self.assertRaises(ValueError):
            simulate(invalid)

    def test_zero_reflection_and_purpose_need_review_input_in_this_candidate(self) -> None:
        state = [0.0, 0.35, 0.60, 0.45, 0.0, 0.65, 0.40]
        inputs = dict(self.configuration["inputs"]["baseline"])
        inputs["review"] = 0.0
        rates = derivative(state, inputs, self.configuration["parameters"])
        self.assertAlmostEqual(rates[0], 0.0)
        self.assertAlmostEqual(rates[4], 0.0)

    def test_sensitivity_values_need_a_range(self) -> None:
        self.assertEqual(parse_values("0.1, 0.2,0.3"), [0.1, 0.2, 0.3])
        with self.assertRaises(ValueError):
            parse_values("0.1")


if __name__ == "__main__":
    unittest.main()
