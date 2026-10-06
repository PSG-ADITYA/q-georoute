"""
Metrics Module for Q-GeoRoute.

Computes geometry-aware graph metrics, routing overheads, fidelity degradations,
and composite architectural recommendation scores:
    Score = Fidelity / (1 + normalized_routing_cost)
"""

from dataclasses import dataclass
from typing import Dict, List
import networkx as nx
from src.topologies import Topology


@dataclass
class GeometryMetrics:
    """Graph and routing metrics for a specific processor topology."""
    topology_name: str
    num_qubits: int
    num_edges: int
    shortest_path_length: int
    average_shortest_path_length: float
    graph_diameter: int
    routing_overhead: int            # Extra hops beyond direct connection (hops - 1)
    swap_overhead: int               # Number of SWAP operations required
    total_two_qubit_gates: int       # SWAPs + 1 CX
    circuit_depth: int               # Primary metric: Transpiled/routed circuit depth from DAG (2 * hops)
    estimated_serialized_depth: int  # Secondary estimate: Serialized time-slices (2 * SWAPs + 1)


@dataclass
class BenchmarkComparison:
    """Fidelity degradation and composite architectural score."""
    topology_name: str
    ideal_fidelity: float
    noisy_fidelity: float
    protected_fidelity: float
    survival_yield: float
    fidelity_degradation: float       # F_ideal - F_noisy
    protection_fidelity_gain: float   # F_protected - F_noisy
    normalized_routing_cost: float
    noisy_composite_score: float      # F_noisy / (1 + cost)
    protected_composite_score: float  # F_protected / (1 + cost)


def compute_geometry_metrics(topology: Topology) -> GeometryMetrics:
    """Compute topology graph and routing overhead metrics."""
    sp = topology.shortest_path()
    sp_length = len(sp) - 1
    num_swaps = 2 * (sp_length - 1)
    total_2q = num_swaps + 1
    
    # Primary metric: Actual compiled DAG circuit depth (2 * hops)
    routed_depth = 2 * sp_length
    # Secondary estimate: Serialized gate-level time-slices (2 * SWAPs + 1)
    serialized_depth = 2 * num_swaps + 1

    return GeometryMetrics(
        topology_name=topology.name,
        num_qubits=topology.num_qubits,
        num_edges=len(topology.edge_list),
        shortest_path_length=sp_length,
        average_shortest_path_length=float(nx.average_shortest_path_length(topology.graph)),
        graph_diameter=nx.diameter(topology.graph),
        routing_overhead=max(0, sp_length - 1),
        swap_overhead=num_swaps,
        total_two_qubit_gates=total_2q,
        circuit_depth=routed_depth,
        estimated_serialized_depth=serialized_depth,
    )


def compute_composite_scores(
    geom_metrics: Dict[str, GeometryMetrics],
    benchmark_fidelities: Dict[str, Dict[str, float]],
    yields: Dict[str, float],
) -> Dict[str, BenchmarkComparison]:
    """
    Computes comparative metrics, fidelity degradation, and composite architectural scores:
        Score = Fidelity / (1 + normalized_routing_cost)

    Args:
        geom_metrics: Mapping of topology name to GeometryMetrics.
        benchmark_fidelities: Mapping of topology name to {'Ideal': f_i, 'Noisy': f_n, 'Noisy + Protection': f_p}.
        yields: Mapping of topology name to post-selection survival yield eta.

    Returns:
        Dictionary of BenchmarkComparison instances.
    """
    # Baseline for normalization: minimum 2Q gates among compared topologies
    min_2q = min(m.total_two_qubit_gates for m in geom_metrics.values())

    comparisons = {}
    for name, gm in geom_metrics.items():
        f_ideal = benchmark_fidelities[name]["Ideal"]
        f_noisy = benchmark_fidelities[name]["Noisy"]
        f_prot = benchmark_fidelities[name]["Noisy + Protection"]
        eta = yields.get(name, 1.0)

        # Normalized routing cost relative to minimum required overhead
        # normalized_routing_cost = (total_2q - min_2q) / min_2q
        norm_cost = (gm.total_two_qubit_gates - min_2q) / float(min_2q)

        noisy_score = f_noisy / (1.0 + norm_cost)
        prot_score = f_prot / (1.0 + norm_cost)

        comparisons[name] = BenchmarkComparison(
            topology_name=name,
            ideal_fidelity=f_ideal,
            noisy_fidelity=f_noisy,
            protected_fidelity=f_prot,
            survival_yield=eta,
            fidelity_degradation=f_ideal - f_noisy,
            protection_fidelity_gain=f_prot - f_noisy,
            normalized_routing_cost=norm_cost,
            noisy_composite_score=noisy_score,
            protected_composite_score=prot_score,
        )

    return comparisons
