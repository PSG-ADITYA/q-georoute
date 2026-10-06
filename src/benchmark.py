"""
Benchmark Module for Q-GeoRoute.

Orchestrates the 9-case benchmark matrix:
- 3 Topologies: 5Q Star, Heavy-Hex Inspired, Hyperbolic Inspired
- 3 Conditions: Ideal, Noisy, Noisy + Protection
Calculates:
- Fidelity F(|Phi+>)
- Pauli expectation values: <XX>, <YY>, <ZZ>
- Routing metrics: SWAP count, Two-qubit gate count, Circuit depth
- Post-selection survival yield eta
Cross-checks ideal fidelity with exact Statevector simulation.
Exports results to results/benchmark_results.csv.
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from qiskit.quantum_info import Statevector, partial_trace, state_fidelity
from qiskit_aer import AerSimulator

from src.topologies import Topology, get_all_topologies
from src.routing import build_routed_bell_circuit, RoutingMetrics
from src.noise import NoiseConfig, build_noise_model
from src.protection import process_measurement_counts, compute_bell_fidelity_from_paulis
from src.metrics import compute_geometry_metrics, compute_composite_scores, GeometryMetrics, BenchmarkComparison


def cross_check_statevector_fidelity(topology: Topology) -> float:
    """
    Computes exact mathematical statevector fidelity of the unmeasured routed Bell circuit
    against the ideal |Phi+> statevector.
    """
    qc, _ = build_routed_bell_circuit(topology, measurement_basis=None)
    sv = Statevector.from_instruction(qc)

    # Trace out all qubits except source and destination
    traced_qubits = [
        q for q in range(topology.num_qubits)
        if q not in (topology.source, topology.destination)
    ]
    reduced_state = partial_trace(sv, traced_qubits)

    ideal_bell = (Statevector.from_label("00") + Statevector.from_label("11")) / np.sqrt(2)
    fid = state_fidelity(reduced_state, ideal_bell)
    return float(np.real(fid))


def run_single_condition(
    topology: Topology,
    condition: str,
    sim: AerSimulator,
    shots: int = 20000,
    seed: int = 42,
) -> Tuple[float, float, float, float, float, RoutingMetrics]:
    """
    Runs tomographic evaluation in X, Y, and Z bases for a specific topology and condition.

    Returns:
        (fidelity, xx, yy, zz, mean_yield, routing_metrics)
    """
    with_protection = (condition == "Noisy + Protection")
    paulis = {}
    yields = []
    routing_metrics = None

    for basis in ["X", "Y", "Z"]:
        qc, metrics = build_routed_bell_circuit(
            topology,
            measurement_basis=basis,
            with_protection_checks=with_protection,
        )
        routing_metrics = metrics

        # Execute on Aer simulator
        job = sim.run(qc, shots=shots, seed_simulator=seed)
        result = job.result()
        counts = result.get_counts()

        post_res = process_measurement_counts(counts, with_protection=with_protection)
        paulis[basis] = post_res.expectation_val
        yields.append(post_res.survival_yield)

    xx = paulis["X"]
    yy = paulis["Y"]
    zz = paulis["Z"]
    fidelity = compute_bell_fidelity_from_paulis(xx, yy, zz)
    mean_yield = sum(yields) / len(yields)

    return fidelity, xx, yy, zz, mean_yield, routing_metrics


def run_full_benchmark(
    noise_config: Optional[NoiseConfig] = None,
    shots: int = 20000,
    seed: int = 42,
    output_csv_path: str = "results/benchmark_results.csv",
) -> Tuple[pd.DataFrame, Dict[str, GeometryMetrics], Dict[str, BenchmarkComparison]]:
    """
    Executes the complete 9-case benchmark across all 3 topologies and 3 conditions.

    Saves results to output_csv_path and returns the results DataFrame,
    geometry metrics, and composite comparison scores.
    """
    if noise_config is None:
        noise_config = NoiseConfig()

    topologies = get_all_topologies()
    noise_model = build_noise_model(noise_config)

    sim_ideal = AerSimulator()
    sim_noisy = AerSimulator(noise_model=noise_model)

    conditions = ["Ideal", "Noisy", "Noisy + Protection"]
    records = []

    geom_metrics: Dict[str, GeometryMetrics] = {}
    fidelities_by_topo: Dict[str, Dict[str, float]] = {}
    yields_by_topo: Dict[str, float] = {}

    for name, topo in topologies.items():
        gm = compute_geometry_metrics(topo)
        geom_metrics[name] = gm
        fidelities_by_topo[name] = {}

        # Exact statevector cross check for ideal baseline
        sv_fidelity = cross_check_statevector_fidelity(topo)

        for cond in conditions:
            sim = sim_ideal if cond == "Ideal" else sim_noisy
            fid, xx, yy, zz, eta, rm = run_single_condition(
                topology=topo,
                condition=cond,
                sim=sim,
                shots=shots,
                seed=seed,
            )

            fidelities_by_topo[name][cond] = fid
            if cond == "Noisy + Protection":
                yields_by_topo[name] = eta

            records.append({
                "Topology": name,
                "Condition": cond,
                "Fidelity": round(fid, 4),
                "XX": round(xx, 4),
                "YY": round(yy, 4),
                "ZZ": round(zz, 4),
                "SWAPs": rm.num_swaps,
                "2Q Gates": rm.total_two_qubit_gates,
                "Depth": rm.circuit_depth,
                "Survival Yield": round(eta, 4),
            })

    df_results = pd.DataFrame(records)

    # Compute composite scores and comparisons
    comparisons = compute_composite_scores(
        geom_metrics=geom_metrics,
        benchmark_fidelities=fidelities_by_topo,
        yields=yields_by_topo,
    )

    # Ensure output directory exists and write CSV
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df_results.to_csv(output_csv_path, index=False)

    return df_results, geom_metrics, comparisons
