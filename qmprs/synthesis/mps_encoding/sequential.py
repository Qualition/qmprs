# Copyright 2023-2024 Qualition Computing LLC.
#
# Licensed under the QUALITION Dual License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Qualition/QMPRS/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

__all__ = ["Sequential"]

import copy
from typing import SupportsIndex
from quick.circuit import Circuit # type: ignore

from qmprs.primitives.mps import MPS, UnitaryBlock, UnitaryLayer
from qmprs.synthesis.mps_encoding import MPSEncoder


class Sequential(MPSEncoder):
    """ `qmprs.synthesis.mps_encoder.Sequential` is the class for preparing MPS
    using Sequential encoding. The circuit is constructed using the disentangling
    algorithm described in the 2019 paper by Shi-Ju Ran.

    We find the sequence of $\chi$ by $\chi$ unitary matrices that optimally disentangle
    the MPS to the product state |00...0>. We compress the max bond dimension to 2, so
    that we would only use one and two qubit gates. We then reverse the sequence to obtain
    the quantum circuit that prepares the MPS from the product state.

    Each layer disentangles the MPS further, and depending on how entangled an MPS is, or
    how large, the number of layers needed to sufficiently disentangle the MPS will differ.

    [1] Ran, Shi-Ju.
    “Encoding of Matrix Product States into Quantum Circuits of One- and Two-Qubit Gates.” (2020).
    https://arxiv.org/abs/1908.07958

    Notes
    -----
    The sequential encoding is a method to prepare a target MPS using a sequence of
    $\chi x \chi$ unitary matrices on each site (aka qubit), where $\chi$ is the bond
    dimension of the MPS.

    Assuming the bond dimension is a power of two, we would require $n = \log_2(\chi)$
    qubits to prepare the unitary matrix. The exact preparation of a general unitary
    matrix scales $O(2^{2n})$, and it is equivalent to $O(\chi^2)$. Furthermore, given
    the linear dependence of the unitary matrices on the number of sites, the overall
    scaling is $O(N\chi^2)$ as stated in [1].

    In the implementation of the sequential encoding, we provide two parameters:
    1) `num_layers` : The number of unitary layers used to prepare the MPS.
    2) `bond_dimension` : The maximum bond dimension of the MPS.

    Unlike [1], the bond dimension does not directly influence the circuit depth.
    Instead, the bond dimension controls the maximum possible fidelity achievable
    through the MPS encoding. The number of layers is the primary parameter that
    influences the circuit depth.

    The algorithm was designed for long-range correlation and follows a unitary-only
    operation approach which limits the efficiency of the encoding. The number of layers
    scale linearly with the circuit depth, and exponentially with the number of sites.
    However, the exponential growth is significantly reduced by the low-rank structure of
    the MPS representation, which allows for increasingly efficient encoding of quantum
    states as we scale the number of sites when compared to exact encoding schema such
    as Mottonen or Shende. Additionally, given the analytical decomposition employed,
    the sequential encoding is computationally more efficient, however, that also means
    that increasing the number of layers will only slightly improve the fidelity of the
    encoding.

    To achieve a higher fidelity within a reasonable circuit depth, we recommend
    using the `qmprs.synthesis.mps_encoding.OptimizedSequential` encoding, which performs
    optimization on the unitary layers to improve the overlap between the MPS encoded by
    the circuit and the target MPS.

    Parameters
    ----------
    `circuit_framework` : type[quick.circuit.Circuit]
        The quantum circuit framework.

    Attributes
    ----------
    `circuit_framework` : type[quick.circuit.Circuit]
        The quantum circuit framework.
    `fidelity_threshold` : float, optional, default=0.99
        The fidelity threshold for the MPS encoding. The encoding stops when the fidelity
        of the MPS with the product state is greater than or equal to the threshold.

    Raises
    ------
    ValueError
        If the number of layers is not a positive integer.

    Usage
    -----
    >>> sequential = Sequential(Circuit)
    """
    def __init__(
            self,
            circuit_framework: type[Circuit]
        ) -> None:

        super().__init__(circuit_framework)

        self.fidelity_threshold = 1 - 1e-6

    def prepare_mps(
            self,
            mps: MPS,
            **kwargs
        ) -> Circuit:

        num_layers = kwargs.get("num_layers", 1)

        if not isinstance(num_layers, SupportsIndex) or num_layers < 1: # type: ignore
            raise ValueError("The number of layers must be a positive integer.")

        # Create a copy of the MPS as the operations are applied inplace
        mps_copy = copy.deepcopy(mps)

        def apply_unitary_layer_to_circuit(
                circuit: Circuit,
                unitary_layer: list[UnitaryBlock]
            ) -> None:
            """ Apply a unitary layer to the quantum circuit.

            Parameters
            ----------
            `circuit` : quick.circuit.Circuit
                The quantum circuit.
            `unitary_layer` : list[qtn.Tensor]
                The unitary layer to be applied to the circuit.
            """
            for start_index, end_index, unitary_blocks in unitary_layer:
                for index in range(start_index, end_index + 1):
                    unitary = unitary_blocks[index - start_index].data

                    # To avoid having to perform a vertical reverse on the circuit for preserving LSB
                    # convention, we will explicitly define the indices after the reverse operation
                    # hence:
                    # - Instead of applying the operation to index, we will apply it to
                    # what index would be after the reverse operation, i.e. abs(index - (circuit.num_qubits - 1)
                    # which is equivalent to abs(index - circuit.num_qubits + 1)
                    # - Instead of applying the operation to index + 1, we will apply it to
                    # what index + 1 would be after the reverse operation, i.e. abs(index + 2 - circuit.num_qubits)
                    if index == end_index:
                        circuit.unitary(unitary, abs(index - circuit.num_qubits + 1))
                    else:
                        circuit.unitary(unitary, [abs(index - circuit.num_qubits + 2), abs(index - circuit.num_qubits + 1)])

        def circuit_from_unitary_layers(
                circuit_framework: type[Circuit],
                unitary_layers: list[list[UnitaryBlock]]
            ) -> Circuit:
            """ Generate a quantum circuit from the MPS unitary layers.

            Parameters
            ----------
            `circuit_framework` : type[quick.circuit.Circuit]
                The quantum circuit framework.
            `unitary_layers` : list[list[qtn.Tensor]]
                A list of unitary layers (list of unitaries) to be applied to the circuit.

            Returns
            -------
            `circuit` : quick.circuit.Circuit
                The quantum circuit with the unitary layers applied.
            """
            circuit = circuit_framework(mps.num_sites)

            for layer in unitary_layers:
                apply_unitary_layer_to_circuit(circuit, layer)

            return circuit

        def sequential_unitary_circuit(mps: MPS) -> Circuit:
            """ Construct the unitary matrix products that optimally disentangle
            the MPS to a product state. These matrix products form the quantum
            circuit that evolves a product state to the targeted MPS.

            These unitary matrix products are referred to as MPDs in [1]. The method
            implements the disentangling algorithm described in section 3 of [1].

            Parameters
            ----------
            `mps` : qmprs.primitives.MPS
                The MPS state to prepare.

            Returns
            -------
            `circuit` : quick.circuit.Circuit
                The quantum circuit preparing the MPS.
            """
            unitary_layers: list[UnitaryLayer] = []

            # Normalize and compress the MPS to the right orthogonal form
            # This improves the fidelity of the encoding
            mps_copy.normalize()
            mps_copy.compress(mode="right")

            # Permute the MPS to left orthogonal form
            mps.permute(shape="lpr")

            # Normalize and convert to canonical form to represent the MPS using
            # isometries
            # This is needed to ensure the unitary layers are indeed unitary, so
            # that we can convert them to quantum gates
            mps.canonicalize("right", normalize=True)

            # The loop will run until either the MPS is sufficiently disentangled
            # or the number of layers is reached
            for _ in range(num_layers):
                # Generate the bond 2 compression of the unitary layer
                # to form one and two qubit gates given Fig. 1 in [1]
                unitary_layer = mps.generate_bond_D_unitary_layer()
                unitary_layers.append(unitary_layer)

                # Given MPS = U|00...0>, we need to apply the inverse of U
                # to disentangle the MPS to the product state |00...0>
                # U^adjoint * MPS = |00...0>
                # This updates the MPS definition for the next layer
                mps.apply_unitary_layer(unitary_layer, inverse=True)

                # If the fidelity of the MPS with the product state is greater
                # than or equal to 0.99, we can break the loop as the MPS is
                # sufficiently disentangled
                if mps.fidelity_with_zero_state() >= self.fidelity_threshold:
                    break

            # Given the unitary layers when applied to the MPS disentangle the MPS
            # to reach the product state |00...0>, we need to reverse the unitary layers
            # to obtain the circuit that prepares the MPS from the product state
            unitary_layers.reverse()

            circuit = circuit_from_unitary_layers(self.circuit_framework, unitary_layers)

            return circuit

        return sequential_unitary_circuit(mps_copy)