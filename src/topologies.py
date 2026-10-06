"""
Topology Module for Q-GeoRoute.

Defines three representative quantum processor coupling topologies:
1. 2016-inspired 5-qubit star/T-shaped topology
2. Contemporary IBM-style heavy-hex-inspired sparse benchmark graph
3. Hyperbolic-inspired finite graph (abstract architecture inspired by negative curvature)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import networkx as nx
import numpy as np


@dataclass
class Topology:
    """Represents a quantum processor coupling topology and benchmark endpoints."""
    name: str
    graph: nx.Graph
    num_qubits: int
    edge_list: List[Tuple[int, int]]
    source: int
    destination: int
    pos: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    description: str = ""

    def validate(self) -> None:
        """Validate topological integrity and benchmark constraints."""
        if not nx.is_connected(self.graph):
            raise ValueError(f"Topology '{self.name}' must be connected.")
        if self.source not in self.graph or self.destination not in self.graph:
            raise ValueError(f"Endpoints ({self.source}, {self.destination}) must belong to the graph.")
        if self.source == self.destination:
            raise ValueError(f"Source and destination must be distinct physical qubits.")
        if self.graph.has_edge(self.source, self.destination):
            raise ValueError(
                f"Benchmark requires non-adjacent endpoints. Edge ({self.source}, {self.destination}) exists."
            )

    def shortest_path(self) -> List[int]:
        """Return the shortest routing path between source and destination."""
        return nx.shortest_path(self.graph, source=self.source, target=self.destination)

    def shortest_path_length(self) -> int:
        """Return the shortest path hop distance between source and destination."""
        return len(self.shortest_path()) - 1

    def diameter(self) -> int:
        """Return the graph diameter (longest shortest path)."""
        return nx.diameter(self.graph)

    def average_shortest_path_length(self) -> float:
        """Return the average shortest path length across all pairs."""
        return float(nx.average_shortest_path_length(self.graph))


def five_qubit_star() -> Topology:
    """
    Constructs a 2016-era inspired 5-qubit star/T-shaped topology.

    Inspired by early IBM quantum devices (e.g., IBM Q 5 Yorktown / Tenerife / bow-tie/T),
    featuring a central junction qubit and linear routing paths.

    Nodes:
        0, 1, 2, 3, 4
    Edges:
        (0, 1), (1, 2), (2, 3), (1, 4)
    Selected Endpoints:
        Source: 0, Destination: 3 (Path: 0 -> 1 -> 2 -> 3, distance = 3 hops).
    """
    g = nx.Graph()
    g.add_nodes_from(range(5))
    edges = [(0, 1), (1, 2), (2, 3), (1, 4)]
    g.add_edges_from(edges)

    pos = {
        0: (0.0, 1.0),
        1: (1.0, 1.0),
        2: (2.0, 1.0),
        3: (3.0, 1.0),
        4: (1.0, 0.0),
    }

    topo = Topology(
        name="5Q Star",
        graph=g,
        num_qubits=5,
        edge_list=edges,
        source=0,
        destination=3,
        pos=pos,
        description="2016-era inspired 5-qubit star/T-shaped coupling graph with central routing hub.",
    )
    topo.validate()
    return topo


def heavy_hex_inspired() -> Topology:
    """
    Constructs a representative contemporary IBM-style heavy-hex-inspired sparse benchmark graph.

    Heavy-hex topologies arrange qubits on vertices and edges of a hexagonal lattice
    to suppress spectator errors and frequency collisions. Vertex qubits have degree <= 3,
    and edge/bridge qubits have degree 2.

    NOTE: This is a representative heavy-hex-inspired benchmark graph of 14 qubits,
    not a claim of reproducing a full 127-qubit IBM Eagle/Heron processor.

    Nodes:
        Vertices: 0, 2, 4, 6, 8, 10, 13
        Bridge qubits: 1, 3, 5, 7, 9, 11, 12
    Selected Endpoints:
        Source: 0, Destination: 6 (Opposite vertices across hexagon, distance = 6 hops).
    """
    g = nx.Graph()
    g.add_nodes_from(range(14))

    # Hexagonal ring: alternating vertex (even) and bridge (odd)
    hex_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
        (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 0)
    ]
    # Heavy-hex interconnect branch from vertex 4 -> bridge 12 -> vertex 13
    branch_edges = [(4, 12), (12, 13)]
    edges = hex_edges + branch_edges
    g.add_edges_from(edges)

    # Compute positions along a regular hexagon plus external branch
    pos = {}
    r = 1.0
    # 12 nodes along the hexagon perimeter
    for i in range(12):
        angle = np.pi / 2 - i * (2 * np.pi / 12)
        pos[i] = (float(r * np.cos(angle)), float(r * np.sin(angle)))
    # Branch nodes from node 4
    x4, y4 = pos[4]
    pos[12] = (x4 + 0.7, y4 - 0.4)
    pos[13] = (x4 + 1.4, y4 - 0.8)

    topo = Topology(
        name="Heavy-Hex Inspired",
        graph=g,
        num_qubits=14,
        edge_list=edges,
        source=0,
        destination=6,
        pos=pos,
        description="Contemporary IBM-style heavy-hex sparse benchmark graph with vertex and bridge qubits (max degree 3).",
    )
    topo.validate()
    return topo


def hyperbolic_inspired() -> Topology:
    """
    Constructs a finite sparse graph inspired by hyperbolic tessellation and negative curvature connectivity.

    In hyperbolic space, area and circumference grow exponentially with radius.
    This finite benchmark architecture features a central core surrounded by concentric shells
    in the Poincaré disk model, providing logarithmic diameter and low-latency radial routing shortcuts.

    NOTE: This is a research simulation model of a finite hyperbolic-inspired abstract architecture,
    not a claim that a physical hyperbolic QPU exists in commercial hardware.

    Nodes:
        Core: 0 (r=0)
        Shell 1 (inner): 1, 2, 3, 4, 5 (r=0.45)
        Shell 2 (outer): 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 (r=0.85)
    Selected Endpoints:
        Source: 6, Destination: 11 (Opposite outer boundary qubits, distance = 4 hops via core).
    """
    g = nx.Graph()
    g.add_nodes_from(range(16))
    edges = []

    # Radial edges from core (0) to Shell 1 (1..5)
    for i in range(1, 6):
        edges.append((0, i))

    # Shell 1 cycle
    for i in range(1, 6):
        nxt = 1 if i == 5 else i + 1
        edges.append((i, nxt))

    # Radial edges from Shell 1 to Shell 2 (2 children per Shell 1 node)
    for i in range(1, 6):
        c1 = 5 + (2 * (i - 1) + 1)
        c2 = 5 + (2 * (i - 1) + 2)
        edges.append((i, c1))
        edges.append((i, c2))

    # Shell 2 perimeter cycle (boundary connectivity)
    for j in range(6, 16):
        nxt = 6 if j == 15 else j + 1
        edges.append((j, nxt))

    g.add_edges_from(edges)

    # Poincaré disk coordinates
    pos = {0: (0.0, 0.0)}
    r1 = 0.45
    for i in range(1, 6):
        angle = np.pi / 2 + (i - 1) * (2 * np.pi / 5)
        pos[i] = (float(r1 * np.cos(angle)), float(r1 * np.sin(angle)))

    r2 = 0.85
    for j in range(6, 16):
        angle = np.pi / 2 + (j - 6) * (2 * np.pi / 10)
        pos[j] = (float(r2 * np.cos(angle)), float(r2 * np.sin(angle)))

    topo = Topology(
        name="Hyperbolic Inspired",
        graph=g,
        num_qubits=16,
        edge_list=edges,
        source=6,
        destination=11,
        pos=pos,
        description="Finite hyperbolic-inspired sparse graph with negative-curvature radial routing shortcuts.",
    )
    topo.validate()
    return topo


def get_all_topologies() -> Dict[str, Topology]:
    """Return an ordered dictionary of all 3 benchmark topologies."""
    return {
        "5Q Star": five_qubit_star(),
        "Heavy-Hex Inspired": heavy_hex_inspired(),
        "Hyperbolic Inspired": hyperbolic_inspired(),
    }
