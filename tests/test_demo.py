"""Tests for interactive demo simulation and input validation."""

import unittest
from src.demo import validate_demo_inputs, simulate_demo, render_topology_base64
from src.topologies import get_all_topologies


class TestDemo(unittest.TestCase):
    def setUp(self):
        self.topologies = get_all_topologies()

    def test_validate_demo_inputs_valid(self):
        """Verify valid configurations across all 3 topologies pass validation."""
        valid_cases = [
            ("5Q Star", 0, 3, "Ideal", 20000),
            ("5Q Star", 0, 2, "Noisy", 10000),
            ("Heavy-Hex Inspired", 0, 6, "Noisy + Protection", 20000),
            ("Heavy-Hex Inspired", 0, 4, "Ideal", 5000),
            ("Hyperbolic Inspired", 6, 11, "Noisy", 20000),
            ("Hyperbolic Inspired", 1, 11, "Noisy + Protection", 15000),
        ]
        for topo, s, d, cond, shots in valid_cases:
            is_valid, err_msg, custom_t, valid_shots = validate_demo_inputs(
                topology_name=topo,
                source_val=s,
                target_val=d,
                condition=cond,
                shots_val=shots,
            )
            self.assertTrue(is_valid, f"Expected {topo} (Q{s}->Q{d}) to be valid. Got error: {err_msg}")
            self.assertIsNone(err_msg)
            self.assertIsNotNone(custom_t)
            self.assertEqual(custom_t.source, s)
            self.assertEqual(custom_t.destination, d)
            self.assertEqual(valid_shots, shots)

    def test_validate_demo_inputs_invalid_topology(self):
        """Verify unknown topology is rejected."""
        is_valid, err, _, _ = validate_demo_inputs("NonExistent Topology", 0, 3, "Ideal", 20000)
        self.assertFalse(is_valid)
        self.assertIn("Invalid topology", err)

    def test_validate_demo_inputs_identical_endpoints(self):
        """Verify identical source and target are rejected."""
        is_valid, err, _, _ = validate_demo_inputs("5Q Star", 2, 2, "Ideal", 20000)
        self.assertFalse(is_valid)
        self.assertIn("cannot be identical", err)

    def test_validate_demo_inputs_adjacent_endpoints(self):
        """Verify directly adjacent endpoints (1 hop) are rejected for remote routing demo."""
        is_valid, err, _, _ = validate_demo_inputs("5Q Star", 0, 1, "Ideal", 20000)
        self.assertFalse(is_valid)
        self.assertIn("directly adjacent", err)

    def test_validate_demo_inputs_out_of_range(self):
        """Verify out-of-range qubit IDs are rejected."""
        # 5Q Star has qubits 0..4
        is_valid, err, _, _ = validate_demo_inputs("5Q Star", 0, 5, "Ideal", 20000)
        self.assertFalse(is_valid)
        self.assertIn("out of range", err)

        is_valid, err, _, _ = validate_demo_inputs("5Q Star", -1, 3, "Ideal", 20000)
        self.assertFalse(is_valid)
        self.assertIn("out of range", err)

    def test_validate_demo_inputs_invalid_shots(self):
        """Verify zero, negative, and non-integer shots are rejected."""
        for invalid_shots in [0, -100, "abc", None]:
            is_valid, err, _, _ = validate_demo_inputs("5Q Star", 0, 3, "Ideal", invalid_shots)
            self.assertFalse(is_valid)
            self.assertIn("Shots", err)

    def test_simulate_demo_ideal_star(self):
        """Verify real simulation on 5Q Star in Ideal mode produces expected Bell metrics."""
        res = simulate_demo(
            topology_name="5Q Star",
            source=0,
            target=3,
            condition="Ideal",
            shots=2000,
            seed=42,
        )
        self.assertEqual(res.topology_name, "5Q Star")
        self.assertEqual(res.source, 0)
        self.assertEqual(res.target, 3)
        self.assertEqual(res.path, [0, 1, 2, 3])
        self.assertEqual(res.hops, 3)
        self.assertEqual(res.swaps, 4)
        self.assertEqual(res.two_qubit_gates, 5)
        self.assertEqual(res.circuit_depth, 6)
        self.assertGreaterEqual(res.fidelity, 0.99)
        self.assertIsNone(res.survival_yield)
        self.assertIn("q_", res.circuit_diagram)

    def test_simulate_demo_noisy_and_protected(self):
        """Verify real simulation in Noisy and Noisy + Protection modes on Hyperbolic topology."""
        res_noisy = simulate_demo(
            topology_name="Hyperbolic Inspired",
            source=6,
            target=11,
            condition="Noisy",
            shots=2000,
            seed=42,
        )
        self.assertAlmostEqual(res_noisy.hops, 4)
        self.assertEqual(res_noisy.swaps, 6)
        self.assertEqual(res_noisy.two_qubit_gates, 7)
        self.assertEqual(res_noisy.circuit_depth, 8)
        self.assertGreater(res_noisy.fidelity, 0.80)
        self.assertLess(res_noisy.fidelity, 0.98)
        self.assertIsNone(res_noisy.survival_yield)

        res_prot = simulate_demo(
            topology_name="Hyperbolic Inspired",
            source=6,
            target=11,
            condition="Noisy + Protection",
            shots=2000,
            seed=42,
        )
        self.assertIsNotNone(res_prot.survival_yield)
        self.assertGreater(res_prot.survival_yield, 0.70)
        self.assertLessEqual(res_prot.survival_yield, 1.0)
        self.assertGreater(res_prot.fidelity, res_noisy.fidelity)

    def test_render_topology_base64(self):
        """Verify topology figure generates valid base64 image string."""
        topo = self.topologies["5Q Star"]
        b64 = render_topology_base64(topo)
        self.assertIsInstance(b64, str)
        self.assertGreater(len(b64), 500)

    def test_run_official_benchmark_demo(self):
        """Verify that demo benchmark mode executes the 9-case matrix and returns valid DataFrame."""
        import os
        from src.demo import run_official_benchmark_demo
        test_csv = "results/test_demo_benchmark.csv"
        df, geom, comp = run_official_benchmark_demo(shots=2000, seed=42, output_csv_path=test_csv)
        self.assertEqual(len(df), 9)
        self.assertIn("Fidelity", df.columns)
        self.assertIn("Survival Yield", df.columns)
        self.assertTrue(os.path.exists(test_csv))
        os.remove(test_csv)


if __name__ == "__main__":
    unittest.main()
