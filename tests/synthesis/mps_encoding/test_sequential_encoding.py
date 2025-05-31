# Copyright 2023-2025 Qualition Computing LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Qualition/qmprs/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

__all__ = ["TestSequential"]

import numpy as np
from numpy.typing import NDArray
from quick.circuit import QiskitCircuit

from qmprs.primitives import MPS
from qmprs.synthesis.mps_encoding import Sequential

from tests.synthesis.mps_encoding import Template


def generate_random_state(num_qubits: int) -> NDArray[np.complex128]:
    """ Generate a random statevector.

    Parameters
    ----------
    num_qubits : int
        The number of qubits.

    Returns
    -------
    `statevector` : NDArray[np.complex128]
        The random statevector.
    """
    statevector = np.random.rand(2**num_qubits) + 1j * np.random.rand(2**num_qubits)
    statevector /= np.linalg.norm(statevector)
    return statevector


class TestSequential(Template):
    """ `tests.synthesis.mps_encoding.TestSequential` is the tester for `qmprs.synthesis.mps_encoding.Sequential` class.
    """
    def test_prepare_state(self) -> None:
        """ Test the preparation of the MPS from a statevector.
        """
        # Define the number of qubits and generate a random statevector
        num_qubits = 8
        statevector = generate_random_state(num_qubits)

        # Define the Sequential encoder
        encoder = Sequential(circuit_framework=QiskitCircuit)

        # Prepare the MPS from the statevector using the Sequential encoder
        circuit = encoder.prepare_state(statevector=statevector, bond_dimension=32, num_layers=32)

        # Extract the statevector from the circuit
        statevector_from_circuit = circuit.get_statevector()

        # Ensure that the statevector from the circuit is equal to the original statevector
        assert 1 - abs(np.dot(statevector_from_circuit.conj(), statevector)) < 1e-2

    def test_prepare_mps(self) -> None:
        """ Test the preparation of the MPS from a MPS.
        """
        # Define the number of qubits and generate a random statevector
        num_qubits = 8
        statevector = generate_random_state(num_qubits)

        # Prepare the MPS from the statevector
        mps = MPS(statevector=statevector, bond_dimension=32)

        # Define the Sequential encoder
        encoder = Sequential(circuit_framework=QiskitCircuit)

        # Prepare the MPS from the MPS using the Sequential encoder
        circuit = encoder.prepare_mps(mps=mps, num_layers=32)

        # Extract the statevector from the circuit
        statevector_from_circuit = circuit.get_statevector()

        # Ensure that the statevector from the circuit is equal to the original statevector
        assert 1 - abs(np.dot(statevector_from_circuit.conj(), statevector)) < 1e-2

    def test_prepare_circuit_with_partial_entanglement(self) -> None:
        """ Test the preparation of the MPS from a statevector with partial entanglement.
        """
        # Generate a circuit where a qubit is not entnagled, and not between
        # two entangled qubits
        num_qubits = 8
        circuit = QiskitCircuit(num_qubits)
        circuit.H(0)
        circuit.CX(0, 3)
        circuit.H(4)
        circuit.H(5)
        circuit.CX(5, 7)

        statevector = circuit.get_statevector()

        # Define the Sequential encoder
        encoder = Sequential(circuit_framework=QiskitCircuit)

        # Prepare the MPS from the statevector using the Sequential encoder
        circuit = encoder.prepare_state(statevector=statevector, bond_dimension=32, num_layers=1)

        # Extract the statevector from the circuit
        statevector_from_circuit = circuit.get_statevector()

        # Ensure that the statevector from the circuit is equal to the original statevector
        assert 1 - abs(np.dot(statevector_from_circuit.conj(), statevector)) < 1e-2

        # The produced circuit is much shallower compared to a full entangled circuit
        # and we need to check that the circuit depth is less than 20
        # (this is a heuristic, but it should be enough for our purposes)
        assert circuit.get_depth() <= 20

    def test_prepare_circuit_with_partial_entanglement_with_sweep(self) -> None:
        """ Test the preparation of the MPS from a statevector with partial
        entanglement using sweeps.
        """
        # Generate a circuit where a qubit is not entnagled, and not between
        # two entangled qubits
        num_qubits = 4
        circuit = QiskitCircuit(num_qubits)
        circuit.H(0)
        circuit.CX(0, 1)
        circuit.CX(1, 2)
        circuit.H(3)

        statevector = circuit.get_statevector()

        # Define the Sequential encoder
        encoder = Sequential(circuit_framework=QiskitCircuit)

        # Prepare the MPS from the statevector using the Sequential encoder
        circuit = encoder.prepare_state(
            statevector=statevector, bond_dimension=32, num_layers=1, num_sweeps=1
        )

        # Extract the statevector from the circuit
        statevector_from_circuit = circuit.get_statevector()

        # Ensure that the statevector from the circuit is equal to the original statevector
        assert 1 - abs(np.dot(statevector_from_circuit.conj(), statevector)) < 1e-2

        # The produced circuit is much shallower compared to a full entangled circuit
        # and we need to check that the circuit depth is less than 20
        # (this is a heuristic, but it should be enough for our purposes)
        assert circuit.get_depth() <= 7

    def test_layer_improvement(self) -> None:
        """ Test the improvement of the fidelity as the number of layers increases.
        """
        # Define the number of qubits and generate a random statevector
        num_qubits = 8
        statevector = generate_random_state(num_qubits)

        # Define the Sequential encoder
        encoder = Sequential(circuit_framework=QiskitCircuit)

        layer_fidelity = []

        for i in range(1, 10):
            # Prepare the MPS from the statevector using the Sequential encoder
            circuit = encoder.prepare_state(statevector=statevector, bond_dimension=64, num_layers=i)

            # Extract the statevector from the circuit
            statevector_from_circuit = circuit.get_statevector()

            # Compute the fidelity between the statevector from the circuit and the original statevector
            fidelity = abs(np.dot(statevector_from_circuit.conj(), statevector))
            layer_fidelity.append(fidelity)

        # Ensure that the fidelity increases with the number of layers
        assert np.all(np.diff(layer_fidelity) >= 0)

    def test_sweep_improvement(self) -> None:
        """ Test the improvement of the fidelity as the bond dimension increases.
        """
        # Define the number of qubits and generate a random statevector
        num_qubits = 8
        statevector = generate_random_state(num_qubits)

        # Define the Sequential encoder
        encoder = Sequential(circuit_framework=QiskitCircuit)

        bond_fidelity = []

        for i in range(1, 8):
            # Prepare the MPS from the statevector using the Sequential encoder
            circuit = encoder.prepare_state(statevector=statevector, bond_dimension=64, num_layers=6, num_sweeps=i)

            # Extract the statevector from the circuit
            statevector_from_circuit = circuit.get_statevector()

            # Compute the fidelity between the statevector from the circuit and the original statevector
            fidelity = abs(np.dot(statevector_from_circuit.conj(), statevector))
            bond_fidelity.append(fidelity)

        # Ensure that the fidelity increases with the bond dimension
        assert np.all(np.diff(bond_fidelity) >= 0)