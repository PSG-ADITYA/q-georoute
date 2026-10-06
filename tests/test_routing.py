"""Tests for shortest-path routing, SWAP insertion, and circuit construction."""

import unittest
from src.topologies import get_all_topologies
from src.routing import build_routed_bell_circuit


class TestRouting(unittest.TestCase):
    def setUp(self):
        self.topologies = get_all_topologies()

    def test_routing_paths_are_valid(self):
        for name, topo in self.topologies.items():
            path = topo.shortest_path()
            self.assertEqual(path[0], topo.source, f"{name}: Path should start at source")
            self.assertEqual(path[-1], topo.destination, f"{name}: Path should end at destination")
            self.assertGreater(len(path), 2, f"{name}: Non-adjacent path length must be >= 3 nodes")

            # Check every consecutive pair of nodes in path has an edge in the graph
            for u, v in zip(path[:-1], path[1:]):
                self.assertTrue(
                    topo.graph.has_edge(u, v),
                    f"{name}: Edge ({u}, {v}) in routing path does not exist in graph!"
                )

    def test_swap_and_two_qubit_metrics(self):
        for name, topo in self.topologies.items():
            _, metrics = build_routed_bell_circuit(topo, measurement_basis="Z")
            hops = len(metrics.path) - 1

            self.assertGreaterEqual(metrics.num_swaps, 0, f"{name}: SWAPs must be non-negative")
            self.assertEqual(
                metrics.num_swaps,
                2 * (hops - 1),
                f"{name}: Expected 2*(hops-1) swaps for bidirectional traversal"
            )
            self.assertEqual(
                metrics.total_two_qubit_gates,
                metrics.num_swaps + metrics.num_cx,
                f"{name}: Total 2Q gates must equal SWAPs + CX"
            )
            self.assertEqual(metrics.num_cx, 1)
            self.assertGreater(metrics.circuit_depth, 0)

    def test_measurement_bases_circuit_structure(self):
        topo = self.topologies["5Q Star"]
        for basis in ["X", "Y", "Z"]:
            qc, _ = build_routed_bell_circuit(topo, measurement_basis=basis, with_protection_checks=True)
            self.assertEqual(qc.num_qubits, topo.num_qubits)
            # 2 target clbits + 2 intermediate qubits = 4 clbits
            self.assertEqual(qc.num_clbits, 4)


    def test_exact_paths_and_routed_depths(self):
        """Verify exact paths and primary routed DAG depth across topologies."""
        expected_paths = {
            "5Q Star": [0, 1, 2, 3],
            "Heavy-Hex Inspired": [0, 1, 2, 3, 4, 5, 6],
            "Hyperbolic Inspired": [6, 1, 0, 3, 11],
        }
        expected_depths = {
            "5Q Star": 6,
            "Heavy-Hex Inspired": 12,
            "Hyperbolic Inspired": 8,
        }
        for name, topo in self.topologies.items():
            _, metrics = build_routed_bell_circuit(topo, measurement_basis="Z")
            self.assertEqual(metrics.path, expected_paths[name], f"{name}: Path mismatch")
            self.assertEqual(metrics.circuit_depth, expected_depths[name], f"{name}: Depth mismatch")


if __name__ == "__main__":
    unittest.main()
