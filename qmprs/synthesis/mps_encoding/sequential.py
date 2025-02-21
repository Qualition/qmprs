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

""" Sequential MPS encoding for preparing MPS using unitary layers.
"""

from __future__ import annotations

__all__ = ["Sequential"]

import copy
from quick.circuit import Circuit

from qmprs.primitives.mps import MPS, UnitaryBlock, UnitaryLayer
from qmprs.synthesis.mps_encoding import MPSEncoder


class Sequential(MPSEncoder):
    """ `qmprs.synthesis.mps_encoder.Sequential` is the class for preparing
    MPS using Sequential encoding. The circuit is constructed using the
    disentangling algorithm described in the 2019 paper by Shi-Ju Ran.

    We find the sequence of $\chi$ by $\chi$ unitary matrices that optimally
    disentangle the MPS to the product state |00...0>. We compress the max
    bond dimension to 2, so that we would only use one and two qubit gates.
    We then reverse the sequence to obtain the quantum circuit that prepares
    the MPS from the product state.

    Each layer disentangles the MPS further, and depending on how entangled
    an MPS is, or how large, the number of layers needed to sufficiently
    disentangle the MPS will differ. The pseudo-code for the algorithm is
    available in [2] for developers' reference in Algorithm 1.

    [1] Ran, Shi-Ju.
    Encoding of Matrix Product States into Quantum Circuits of One- and Two-Qubit Gates (2020).
    https://arxiv.org/abs/1908.07958

    [2] Rudolph, Chen, Miller, Acharya, Perdomo-Ortiz.
    Decomposition of Matrix Product States into Shallow Quantum Circuits (2022).
    https://arxiv.org/abs/2209.00595

    Notes
    -----
    The sequential encoding is a method to prepare a target MPS using a sequence
    of $\chi x \chi$ unitary matrices on each site (aka qubit), where $\chi$ is
    the bond dimension of the MPS.

    Assuming the bond dimension is a power of two, we would require $n = \log_2(\chi)$
    qubits to prepare the unitary matrix. The exact preparation of a general unitary
    matrix scales $O(2^{2n})$, and it is equivalent to $O(\chi^2)$. Furthermore,
    given the linear dependence of the unitary matrices on the number of sites,
    the overall scaling is $O(N\chi^2)$ as stated in [1].

    In the implementation of the sequential encoding, we provide two parameters:
    1) `num_layers` : The number of unitary layers used to prepare the MPS.
    2) `bond_dimension` : The maximum bond dimension of the MPS.

    Unlike [1], the bond dimension does not directly influence the circuit depth.
    Instead, the bond dimension controls the maximum possible fidelity achievable
    through the MPS encoding. The number of layers is the primary parameter that
    influences the circuit depth.

    The algorithm was designed for long-range correlation and follows a unitary-only
    operation approach which limits the efficiency of the encoding. The number
    of layers scale linearly with the circuit depth, and exponentially with the
    number of sites. However, the exponential growth is significantly reduced by
    the low-rank structure of the MPS representation, which allows for increasingly
    efficient encoding of quantum states as we scale the number of sites when compared
    to exact encoding schema such as Mottonen, Shende, or SOTA Isometry by Iten et al.
    Additionally, given the analytical decomposition employed, the sequential encoding
    is computationally more efficient, however, that also means that increasing the
    number of layers will only slightly improve the fidelity of the encoding.

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
        The fidelity threshold for the MPS encoding. The encoding stops when the
        fidelity of the MPS with the product state is greater than or equal to the
        threshold.

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

        self._fidelity_threshold = 1 - 1e-6

    @property
    def fidelity_threshold(self) -> float:
        """ The fidelity threshold $\hat{f}$ for the MPS encoding.

        Returns
        -------
        `fidelity_threshold` : float
            The fidelity threshold.
        """
        return self._fidelity_threshold

    @fidelity_threshold.setter
    def fidelity_threshold(
            self,
            threshold: float
        ) -> None:
        """ Set the fidelity threshold $\hat{f}$ for the MPS encoding.

        Parameters
        ----------
        `threshold` : float
            The fidelity threshold for the MPS encoding.

        Raises
        ------
        ValueError
            - If the fidelity threshold is not a float between 0 and 1.
        """
        if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 1:
            raise ValueError("The fidelity threshold must be a float between 0 and 1.")

        self._fidelity_threshold = threshold

    @staticmethod
    def _apply_unitary_layer_to_circuit(
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

    @staticmethod
    def _circuit_from_unitary_layers(
            mps: MPS,
            circuit_framework: type[Circuit],
            unitary_layers: list[list[UnitaryBlock]]
        ) -> Circuit:
        """ Generate a quantum circuit from the MPS unitary layers.

        Parameters
        ----------
        `mps` : qmprs.primitives.MPS
            The MPS state to prepare.
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
            Sequential._apply_unitary_layer_to_circuit(circuit, layer)

        return circuit

    def _sequential_unitary_circuit(
            self,
            mps: MPS,
            num_layers: int
        ) -> Circuit:
        """ Construct the unitary matrix products that optimally disentangle
        the MPS to a product state. These matrix products form the quantum
        circuit that evolves a product state to the targeted MPS.

        These unitary matrix products are referred to as MPDs in [1]. The method
        implements the disentangling algorithm described in section 3 of [1].

        Parameters
        ----------
        `mps` : qmprs.primitives.MPS
            The MPS state to prepare.
        `num_layers` : int
            The number of unitary layers to prepare the MPS.

        Returns
        -------
        `circuit` : quick.circuit.Circuit
            The quantum circuit preparing the MPS.
        """
        unitary_layers: list[UnitaryLayer] = []

        # Normalize and compress the MPS to the right orthogonal form
        # This improves the depth and fidelity of the circuit
        mps.normalize()
        mps.compress(mode="right")

        # Permute the MPS to left orthogonal form
        mps.permute(shape="lpr")

        # Normalize and convert to canonical form to represent the MPS using
        # isometries
        # This is needed to ensure the unitary layers are indeed unitary, so
        # that we can convert them to quantum gates
        mps.canonicalize("right", normalize=True)

        # The loop will run until either the MPS is sufficiently disentangled
        # or the number of layers is reached [2]
        for _ in range(num_layers):
            # Generate the bond 2 truncation of the unitary layer
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

        circuit = Sequential._circuit_from_unitary_layers(mps, self.circuit_framework, unitary_layers)

        return circuit

    def prepare_mps(
            self,
            mps: MPS,
            **kwargs
        ) -> Circuit:

        num_layers = kwargs.get("num_layers", 1)

        if not isinstance(num_layers, int) or num_layers < 1:
            raise ValueError("The number of layers must be a positive integer.")

        # Create a copy of the MPS as the operations are applied inplace
        mps_copy = copy.deepcopy(mps)

        return self._sequential_unitary_circuit(mps_copy, num_layers)