"""
Protection Module for Q-GeoRoute.

Implements stabilizer/parity verification with post-selection for routed remote Bell states:
- Measures intermediate routing qubits along the SWAP transport path as intrinsic syndromes
- Evaluates stabilizer consistency
- Discards shots where any syndrome check fails
- Quantifies the post-selection survival yield:
    eta = accepted_shots / total_shots
- Recomputes Pauli expectation values and Bell-state fidelity on accepted shots.

NOTE: This strategy is explicitly labeled as stabilizer/parity verification with post-selection,
and does not claim to implement full fault-tolerant quantum error correction.
"""

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class PostSelectionResult:
    """Results from shot-based measurement with optional stabilizer post-selection."""
    expectation_val: float
    survival_yield: float
    total_shots: int
    accepted_shots: int
    accepted_distribution: Dict[str, float]


def process_measurement_counts(
    counts: Dict[str, int],
    with_protection: bool = False,
) -> PostSelectionResult:
    """
    Parses Qiskit Aer measurement bitstrings, evaluates syndrome flags,
    and calculates the basis expectation value and survival yield.

    Bitstring convention (little-endian):
    - Target qubits (src, dst) are measured into clbits 0 and 1 (last 2 characters of bitstring).
    - If protection is enabled, syndrome check bits occupy preceding clbit positions (bitstring[:-2]).

    Args:
        counts: Dictionary of {bitstring: count} from Aer execution.
        with_protection: Whether to apply post-selection on syndrome bits == 0.

    Returns:
        PostSelectionResult with expectation value <ZZ>, <XX>, or <YY> and survival yield eta.
    """
    total_shots = sum(counts.values())
    if total_shots == 0:
        return PostSelectionResult(
            expectation_val=0.0,
            survival_yield=0.0,
            total_shots=0,
            accepted_shots=0,
            accepted_distribution={"00": 0.0, "01": 0.0, "10": 0.0, "11": 0.0},
        )

    accepted_shots = 0
    accepted_counts = {"00": 0, "01": 0, "10": 0, "11": 0}

    for bitstr, cnt in counts.items():
        # Clean any whitespace in bitstring if returned as segmented
        clean_bitstr = bitstr.replace(" ", "")

        target_bits = clean_bitstr[-2:]
        syndrome_bits = clean_bitstr[:-2] if len(clean_bitstr) > 2 else ""

        if with_protection and len(syndrome_bits) > 0:
            # All verification syndrome check bits must be '0'
            if all(b == "0" for b in syndrome_bits):
                accepted_shots += cnt
                accepted_counts[target_bits] = accepted_counts.get(target_bits, 0) + cnt
        else:
            accepted_shots += cnt
            accepted_counts[target_bits] = accepted_counts.get(target_bits, 0) + cnt

    survival_yield = accepted_shots / total_shots if total_shots > 0 else 0.0

    if accepted_shots > 0:
        p00 = accepted_counts.get("00", 0) / accepted_shots
        p11 = accepted_counts.get("11", 0) / accepted_shots
        p01 = accepted_counts.get("01", 0) / accepted_shots
        p10 = accepted_counts.get("10", 0) / accepted_shots
        exp_val = (p00 + p11) - (p01 + p10)
        dist = {"00": p00, "01": p01, "10": p10, "11": p11}
    else:
        exp_val = 0.0
        dist = {"00": 0.0, "01": 0.0, "10": 0.0, "11": 0.0}

    return PostSelectionResult(
        expectation_val=float(exp_val),
        survival_yield=float(survival_yield),
        total_shots=total_shots,
        accepted_shots=accepted_shots,
        accepted_distribution=dist,
    )


def compute_bell_fidelity_from_paulis(
    xx: float,
    yy: float,
    zz: float,
) -> float:
    """
    Computes Bell state |Phi+> fidelity from Pauli expectation values:
        F(|Phi+>) = (1 + <XX> - <YY> + <ZZ>) / 4
    """
    fid = (1.0 + xx - yy + zz) / 4.0
    # Numerical clip to physical range [0.0, 1.0]
    return float(max(0.0, min(1.0, fid)))
