"""
Noise Modeling Module for Q-GeoRoute.

Defines realistic, configurable quantum noise models using Qiskit Aer:
- Single-qubit depolarizing gate error
- Two-qubit depolarizing gate error
- Readout (measurement assignment) error
"""

from dataclasses import dataclass
from typing import Optional
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError


@dataclass
class NoiseConfig:
    """Configurable parameters for quantum device noise simulation."""
    p1_gate_error: float = 0.001   # 0.1% 1Q gate depolarizing rate
    p2_gate_error: float = 0.015   # 1.5% 2Q gate depolarizing rate
    readout_error: float = 0.020   # 2.0% readout assignment error
    seed: Optional[int] = 42

    def summary(self) -> str:
        return (
            f"1Q Depolarizing: {self.p1_gate_error*100:.2f}%, "
            f"2Q Depolarizing: {self.p2_gate_error*100:.2f}%, "
            f"Readout Error: {self.readout_error*100:.2f}%"
        )


def build_noise_model(config: Optional[NoiseConfig] = None) -> NoiseModel:
    """
    Constructs a Qiskit Aer NoiseModel based on the provided configuration.

    Args:
        config: NoiseConfig instance. If None, default realistic parameters are used.

    Returns:
        Qiskit Aer NoiseModel object.
    """
    if config is None:
        config = NoiseConfig()

    noise_model = NoiseModel()

    # 1. Single-qubit depolarizing error on rotation/superposition gates
    if config.p1_gate_error > 0:
        err_1q = depolarizing_error(config.p1_gate_error, 1)
        noise_model.add_all_qubit_quantum_error(err_1q, ["h", "s", "sdg"])

    # 2. Two-qubit depolarizing error on CX and SWAP gates
    if config.p2_gate_error > 0:
        err_2q = depolarizing_error(config.p2_gate_error, 2)
        noise_model.add_all_qubit_quantum_error(err_2q, ["cx", "swap"])

    # 3. Symmetric readout error on measurement
    if config.readout_error > 0:
        p_ro = config.readout_error
        ro_matrix = [
            [1.0 - p_ro, p_ro],
            [p_ro, 1.0 - p_ro]
        ]
        readout_err = ReadoutError(ro_matrix)
        noise_model.add_all_qubit_readout_error(readout_err)

    return noise_model
