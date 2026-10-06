"""
Routing Module for Q-GeoRoute.

Constructs geometry-aware shortest-path routed quantum circuits for remote Bell state preparation:
    |Phi+> = (|00> + |11>) / sqrt(2)
between non-adjacent physical qubits across arbitrary coupling graphs.
Tracks SWAPs, CX gates, two-qubit gate counts, and circuit depth.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
from qiskit import QuantumCircuit
from src.topologies import Topology


@dataclass
class RoutingMetrics:
    """Detailed structural and gate metrics for a routed circuit."""
    topology_name: str
    path: List[int]
    path_length: int
    num_swaps: int
    num_cx: int
    total_two_qubit_gates: int
    circuit_depth: int


def build_routed_bell_circuit(
    topology: Topology,
    measurement_basis: Optional[str] = None,
    with_protection_checks: bool = False,
) -> Tuple[QuantumCircuit, RoutingMetrics]:
    """
    Builds a routed Bell-state preparation circuit on the specified topology.

    Algorithm:
    1. Find shortest path P = [p_0, p_1, ..., p_k] from topology.source to topology.destination.
    2. Apply H(p_0) to create superposition.
    3. SWAP p_0 sequentially forward to p_{k-1} (adjacent to p_k).
    4. Apply CX(p_{k-1}, p_k) to entangle the qubits.
    5. Reverse SWAPs from p_{k-1} back to p_0.
       Physical qubit p_0 now holds the control qubit, physical qubit p_k holds the target qubit,
       and intermediate qubits p_1, ..., p_{k-1} return to state |0>.

    Args:
        topology: The target processor topology.
        measurement_basis: None (statevector), 'Z', 'X', or 'Y'.
        with_protection_checks: If True, measures intermediate routing qubits and ancilla
                                into syndrome bits for post-selection verification.

    Returns:
        circuit: Transpiled-free native routed QuantumCircuit.
        metrics: RoutingMetrics tracking routing path, SWAPs, CX, 2Q gates, and depth.
    """
    path = topology.shortest_path()
    k = len(path) - 1  # path length in hops
    num_qubits = topology.num_qubits
    src = topology.source
    dst = topology.destination

    # Intermediate qubits along the path
    intermediate_qubits = path[1:-1]

    # Base circuit
    # If with_protection_checks and measurement_basis is set:
    # We allocate 2 classical bits for target qubits (src, dst), plus len(intermediate_qubits) syndrome bits
    num_clbits = 0
    if measurement_basis is not None:
        num_clbits = 2 + (len(intermediate_qubits) if with_protection_checks else 0)
        qc = QuantumCircuit(num_qubits, num_clbits)
    else:
        qc = QuantumCircuit(num_qubits)

    # 1. State preparation: H on source qubit
    qc.h(src)

    # 2. Forward SWAP chain
    swaps_forward = 0
    for i in range(k - 1):
        qc.swap(path[i], path[i + 1])
        swaps_forward += 1

    # 3. Entangling CX between adjacent qubits (p_{k-1}, p_k)
    qc.cx(path[k - 1], dst)
    cx_count = 1

    # 4. Reverse SWAP chain
    swaps_reverse = 0
    for i in range(k - 2, -1, -1):
        qc.swap(path[i], path[i + 1])
        swaps_reverse += 1

    total_swaps = swaps_forward + swaps_reverse
    total_two_qubit_gates = total_swaps + cx_count

    # Circuit depth before measurement basis rotations
    depth_core = qc.depth()

    # 5. Basis rotation and measurement if requested
    if measurement_basis is not None:
        basis = measurement_basis.upper()
        if basis == "X":
            qc.h(src)
            qc.h(dst)
        elif basis == "Y":
            qc.sdg(src)
            qc.h(src)
            qc.sdg(dst)
            qc.h(dst)
        elif basis == "Z":
            pass  # Standard computational basis
        else:
            raise ValueError(f"Unknown measurement basis: {measurement_basis}. Use 'Z', 'X', or 'Y'.")

        # Measure targets into clbits 0 and 1
        qc.measure(src, 0)
        qc.measure(dst, 1)

        # If protection checks enabled, measure intermediate qubits into syndrome clbits 2, 3, ...
        if with_protection_checks:
            for idx, inter_q in enumerate(intermediate_qubits):
                qc.measure(inter_q, 2 + idx)

    metrics = RoutingMetrics(
        topology_name=topology.name,
        path=path,
        path_length=k,
        num_swaps=total_swaps,
        num_cx=cx_count,
        total_two_qubit_gates=total_two_qubit_gates,
        circuit_depth=depth_core,
    )

    return qc, metrics
