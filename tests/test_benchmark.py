"""Tests for benchmark simulation, fidelity, protection post-selection, and CSV matrix."""

import os
import unittest
import pandas as pd
from src.topologies import get_all_topologies
from src.benchmark import cross_check_statevector_fidelity, run_full_benchmark
from src.noise import NoiseConfig


class TestBenchmark(unittest.TestCase):
    def setUp(self):
        self.topologies = get_all_topologies()

    def test_ideal_bell_fidelity_is_high(self):
        """Verify exact statevector fidelity is mathematically 1.0 for all topologies."""
        for name, topo in self.topologies.items():
            fid = cross_check_statevector_fidelity(topo)
            self.assertGreater(
                fid,
                0.999,
                f"{name}: Ideal Bell state preparation fidelity should be > 0.999 (got {fid})"
            )

    def test_nine_cases_and_csv_generation(self):
        """Verify all 9 benchmark cases are generated, have realistic values, and CSV exists with 9 rows."""
        test_csv = "results/test_benchmark_results.csv"
        df, geom_metrics, comparisons = run_full_benchmark(
            noise_config=NoiseConfig(p1_gate_error=0.001, p2_gate_error=0.015, readout_error=0.02),
            shots=5000,
            seed=42,
            output_csv_path=test_csv,
        )

        # 1. Row count and columns
        self.assertEqual(len(df), 9, f"Expected 9 rows in benchmark matrix, got {len(df)}")
        expected_cols = ["Topology", "Condition", "Fidelity", "XX", "YY", "ZZ", "SWAPs", "2Q Gates", "Depth", "Survival Yield"]
        for col in expected_cols:
            self.assertIn(col, df.columns)

        # 2. Check each topology has Ideal, Noisy, Noisy + Protection
        for name in ["5Q Star", "Heavy-Hex Inspired", "Hyperbolic Inspired"]:
            topo_df = df[df["Topology"] == name]
            self.assertEqual(len(topo_df), 3)

            f_ideal = topo_df[topo_df["Condition"] == "Ideal"]["Fidelity"].values[0]
            f_noisy = topo_df[topo_df["Condition"] == "Noisy"]["Fidelity"].values[0]
            f_prot = topo_df[topo_df["Condition"] == "Noisy + Protection"]["Fidelity"].values[0]
            eta_prot = topo_df[topo_df["Condition"] == "Noisy + Protection"]["Survival Yield"].values[0]

            # 3. Ideal fidelity is near 1.0
            self.assertGreaterEqual(f_ideal, 0.98, f"{name}: Ideal fidelity should be >= 0.98")

            # 4. Noisy fidelity degrades (does not magically become perfect)
            self.assertLess(
                f_noisy,
                f_ideal,
                f"{name}: Noisy fidelity should be strictly less than ideal fidelity"
            )
            self.assertGreater(f_noisy, 0.70, f"{name}: Noisy fidelity should remain physical")

            # 5. Protected fidelity improves over noisy baseline
            self.assertGreater(
                f_prot,
                f_noisy,
                f"{name}: Protected fidelity should exceed noisy baseline"
            )

            # 6. Survival yield is a real physical probability: 0 < eta <= 1.0
            self.assertGreater(eta_prot, 0.5, f"{name}: Survival yield must be > 0.5")
            self.assertLessEqual(eta_prot, 1.0, f"{name}: Survival yield must be <= 1.0")

        # 7. Check CSV file was written and clean up test file
        self.assertTrue(os.path.exists(test_csv))
        os.remove(test_csv)


if __name__ == "__main__":
    unittest.main()
