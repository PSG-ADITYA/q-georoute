"""Tests for processor topology graphs and endpoints."""

import unittest
import networkx as nx
from src.topologies import get_all_topologies, five_qubit_star, heavy_hex_inspired, hyperbolic_inspired


class TestTopologies(unittest.TestCase):
    def setUp(self):
        self.topologies = get_all_topologies()

    def test_three_topologies_exist(self):
        self.assertEqual(len(self.topologies), 3)
        self.assertIn("5Q Star", self.topologies)
        self.assertIn("Heavy-Hex Inspired", self.topologies)
        self.assertIn("Hyperbolic Inspired", self.topologies)

    def test_topology_graphs_are_connected(self):
        for name, topo in self.topologies.items():
            self.assertTrue(
                nx.is_connected(topo.graph),
                f"Topology {name} graph must be fully connected."
            )

    def test_endpoints_are_valid_and_non_adjacent(self):
        for name, topo in self.topologies.items():
            self.assertIn(topo.source, topo.graph, f"{name}: source not in graph")
            self.assertIn(topo.destination, topo.graph, f"{name}: destination not in graph")
            self.assertNotEqual(topo.source, topo.destination, f"{name}: endpoints must be distinct")
            self.assertFalse(
                topo.graph.has_edge(topo.source, topo.destination),
                f"{name}: endpoints {topo.source} and {topo.destination} must NOT be directly adjacent!"
            )

    def test_heavy_hex_sparsity(self):
        hh = heavy_hex_inspired()
        degrees = dict(hh.graph.degree())
        for node, deg in degrees.items():
            self.assertLessEqual(deg, 3, f"Heavy-hex node {node} has degree {deg} > 3")

    def test_hyperbolic_connectivity_and_diameter(self):
        hyp = hyperbolic_inspired()
        self.assertEqual(hyp.num_qubits, 16)
        self.assertLessEqual(hyp.diameter(), 4, "Hyperbolic graph should maintain small diameter")
        self.assertLess(hyp.average_shortest_path_length(), 2.5)


if __name__ == "__main__":
    unittest.main()
