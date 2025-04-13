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
import numpy as np
import quimb.tensor as qtn # type: ignore
from quick.circuit import Circuit

from qmprs.primitives.mps import MPS, UnitaryBlock, UnitaryLayer
from qmprs.synthesis.mps_encoding import MPSEncoder


class Sequential(MPSEncoder):
    r""" `qmprs.synthesis.mps_encoder.Sequential` is the class for preparing
    MPS using Sequential encoding. The circuit is constructed using the
    disentangling algorithm described in the 2019 paper by Shi-Ju Ran.

    We find the sequence of $\chi$ by $\chi$ unitary matrices that optimally
    disentangle the MPS to the product state $\ket{00\cdots 0}$. We compress
    the max bond dimension to 2, so that we would only use one and two qubit gates.
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

    To achieve a higher fidelity within a reasonable circuit depth, we use environment
    tensor updates to reach the optimal gates for the circuit based on [2].

    Parameters
    ----------
    `circuit_framework` : type[quick.circuit.Circuit]
        The quantum circuit framework.

    Attributes
    ----------
    `circuit_framework` : type[quick.circuit.Circuit]
        The quantum circuit framework.
    `fidelity_threshold` : float, optional, default=0.999999
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
        r""" The fidelity threshold $\hat{f}$ for the MPS encoding.

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
        r""" Set the fidelity threshold $\hat{f}$ for the MPS encoding.

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
        for start_index, end_index, unitary_block in unitary_layer:
            for index in range(start_index, end_index + 1):
                unitary = unitary_block[index - start_index].data

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
                    circuit.unitary(
                        unitary,
                        [abs(index - circuit.num_qubits + 2), abs(index - circuit.num_qubits + 1)]
                    )

    def _circuit_from_unitary_layers(
            self,
            mps: MPS,
            unitary_layers: list[UnitaryLayer]
        ) -> Circuit:
        """ Generate a quantum circuit from the MPS unitary layers.

        Parameters
        ----------
        `mps` : qmprs.primitives.MPS
            The MPS state to prepare.
        `unitary_layers` : list[UnitaryLayer]
            A list of unitary layers (list of unitaries) to be applied to the circuit.

        Returns
        -------
        `circuit` : quick.circuit.Circuit
            The quantum circuit with the unitary layers applied.
        """
        circuit = self.circuit_framework(mps.num_sites)

        for layer in unitary_layers:
            Sequential._apply_unitary_layer_to_circuit(circuit, layer)

        return circuit

    @staticmethod
    def _tensor_network_from_unitary_layers(
            mps: MPS,
            unitary_layers: list[UnitaryLayer]
        ) -> qtn.TensorNetwork:
        """ Generate the circuit tensor network from the unitary layers.

        Notes
        -----
        We can use `quick.circuit.QuimbCircuit` and reduce the duplication
        by using `Sequential._circuit_from_unitary_layers()`, however, we
        would have to perform a vertical reverse on the circuit which introduces
        additional runtime so we trade off brevity with performance and rewrite
        this method from scratch using `qtn.Circuit`.

        Parameters
        ----------
        `mps` : qmprs.primitives.MPS
            The MPS state to prepare.
        `unitary_layers` : list[UnitaryLayer]
            A list of unitary layers, each containing the start
            and end index of the qubits and the corresponding
            unitary blocks.

        Returns
        -------
        `qtn.TensorNetwork`
            The tensor network representation of the circuit.
        """
        circuit = qtn.Circuit(N=mps.num_sites)
        gate_tracker: list[str] = []

        for i, unitary_layer in enumerate(unitary_layers):
            for j, (start_index, end_index, unitary_block) in enumerate(unitary_layer):
                for index in range(start_index, end_index + 1):
                    unitary = unitary_block[index].data

                    # MSB convention
                    if index == end_index:
                        circuit.apply_gate_raw(
                            unitary.reshape(2, 2),
                            where=[index]
                        )
                    else:
                        circuit.apply_gate_raw(
                            unitary.reshape(2, 2, 2, 2),
                            where=[index, index + 1]
                        )

                    # Add the gate to the tracker
                    gate_tracker.append(f"{i}_{j}_{index}")

        tensor_network = qtn.TensorNetwork(circuit.psi)

        # We do not want to include the qubits, so we will
        # excplitily control the iteration index
        gate_index = 0

        for gate in tensor_network:
            # We only update the gates, not the qubits
            if "PSI0" in gate.tags:
                continue

            # Remove existing tags from the gate
            gate.drop_tags(tags=gate.tags)

            # Marshal the gate with the gate tracker
            # This is needed to ensure the gates are properly tagged
            # for updating the unitary layers
            gate.add_tag(gate_tracker[gate_index])
            gate_index += 1

        return tensor_network

    def _get_unitary_layer(
            self,
            mps: MPS
        ) -> UnitaryLayer:
        r""" Construct a single layer of matrix products that optimally
        disentangle the MPS to the product state $\ket{00\cdots 0}$.

        Notes
        -----
        These unitary matrix products are referred to as MPDs in [1]. The method
        implements the disentangling algorithm described in section 3 of [1].

        Synonymously, this method implements Di from [2].

        Parameters
        ----------
        `mps` : qmprs.primitives.MPS
            The MPS state to prepare.

        Returns
        -------
        `unitary_layer` : UnitaryLayer
            The unitary layer that prepares the MPS.
        """
        # Generate the bond 2 truncation of the unitary layer
        # to form one and two qubit gates given Fig. 1 in [1]
        unitary_layer = mps.generate_bond_D_unitary_layer()

        # Given MPS ~= U_k|00...0>, we need to apply the inverse of U_k
        # to disentangle the MPS to the product state |00...0>
        # U_k^adjoint * MPS ~= |00...0>
        # This updates the MPS definition for the next layer
        mps.apply_unitary_layer(unitary_layer, inverse=True)

        return unitary_layer

    def _get_unitary_layers(
            self,
            mps: MPS,
            num_layers: int
        ) -> list[UnitaryLayer]:
        r""" Construct the unitary matrix products that optimally disentangle
        the MPS to the product state $\ket{00\cdots 0}$. These matrix products
        form the quantum circuit that evolves a product state to the targeted
        MPS.

        Notes
        -----
        These unitary matrix products are referred to as MPDs in [1]. The method
        implements the disentangling algorithm described in section 3 of [1].

        Synonymously, this method implements Dall from [2].

        Parameters
        ----------
        `mps` : qmprs.primitives.MPS
            The MPS state to prepare.
        `num_layers` : int
            The number of unitary layers to prepare the MPS.

        Returns
        -------
        `unitary_layers` : list[UnitaryLayer]
            The unitary layers that prepare the MPS.
        """
        # The MPS is copied to avoid modifying the original MPS
        mps_copy = copy.deepcopy(mps)

        unitary_layers: list[UnitaryLayer] = []

        # Normalize and compress the MPS to the right orthogonal form
        # This improves the depth and fidelity of the circuit
        mps_copy.normalize()
        mps_copy.compress(mode="right")

        # Permute the MPS to left orthogonal form
        mps_copy.permute(shape="lpr")

        # Normalize and convert to canonical form to represent the MPS using
        # isometries
        # This is needed to ensure the unitary layers are indeed unitary, so
        # that we can convert them to quantum gates
        mps_copy.canonicalize("right", normalize=True)

        # The loop will run until either the MPS is sufficiently disentangled
        # or the number of layers is reached [2]
        # |psi> = U_1 U_2 ... U_k |00...0>
        # where U_k is the last unitary layer
        # and |psi> is the MPS
        # U_k^adjoint ... U_2^adjoint U_1^adjoint |psi> = |00...0>
        for _ in range(num_layers):
            unitary_layers.append(self._get_unitary_layer(mps_copy))

            # If the fidelity of the MPS with the product state is greater
            # than or equal to the fidelity threshold, we can break the loop
            # as the MPS is sufficiently disentangled
            if np.isclose(mps_copy.fidelity_with_zero_state(), 1+0j, atol=1-self.fidelity_threshold):
                break

        # Given the unitary layers when applied to the MPS disentangle the MPS
        # to reach the product state |00...0>, we need to reverse the unitary layers
        # to obtain the circuit that prepares the MPS from the product state |00...0>
        unitary_layers.reverse()

        return unitary_layers

    def _sweep_unitary_layers(
            self,
            mps: MPS,
            circuit_tensor_network: qtn.TensorNetwork,
            unitary_layers: list[UnitaryLayer],
            specific_layer_indices: list[int] = []
        ) -> tuple[list[UnitaryLayer], qtn.TensorNetwork]:
        """ Perform a sweep of the unitary layers to optimize the gates
        using the environment tensor updates to reach the optimal gates
        for the circuit based on [2].

        This method implements Oall and allows for Iter Oi from [2].

        Parameters
        ----------
        `mps` : qmprs.primitives.MPS
            The MPS state to prepare.
        `circuit_tensor_network` : qtn.TensorNetwork
            The circuit tensor network to be optimized.
        `unitary_layers` : list[UnitaryLayer]
            The unitary layers to be optimized.
        `specific_layer_indices` : list[int], optional, default=[]
            The specific layer indices to be optimized. If empty, all layers
            are optimized. This is used to optimize specific layers of the
            circuit tensor network. This is useful for implementing Iter
            Oi from [2].

        Returns
        -------
        `unitary_layers` : list[UnitaryLayer]
            The optimized unitary layers.
        `circuit_tensor_network` : qtn.TensorNetwork
            The optimized circuit tensor network.
        """
        # We do not want to include the qubits, so we will
        # excplitily control the iteration index
        gate_index = 0

        for gate in circuit_tensor_network:
            # We only update the gates, not the qubits
            if "PSI0" in gate.tags:
                continue

            # Update the unitary layer with the new unitary
            gate_tag = list(gate.tags)[0]
            layer_index, block_index, tensor_index = gate_tag.split("_")

            # If the layer index is not in the specific layer indices,
            # we skip the gate
            if specific_layer_indices and int(layer_index) not in specific_layer_indices:
                gate_index += 1
                continue

            # Remove the gate tensor from the circuit
            # and contract the circuit with the conjugate of the MPS
            # to get the environment tensor
            circuit_tensor_network, _ = circuit_tensor_network.partition(gate.tags)
            environment_tensor = mps.mps.conj() @ circuit_tensor_network

            left_inds = gate.inds[:2] if len(gate.inds) == 4 else [gate.inds[0]]
            right_inds = gate.inds[2:] if len(gate.inds) == 4 else [gate.inds[1]]

            # The environment tensor is the optimal tensor to
            # match the target MPS, however, it is not guaranteed
            # to be unitary
            # We need to perform SVD on the environment tensor
            # to approximate the environment tensor with a unitary
            u, _, vh = np.linalg.svd(
                environment_tensor.to_dense( # type: ignore
                    (left_inds), (right_inds)
                )
            )
            u_new = np.dot(u, vh)

            if len(gate.inds) == 2:
                new_tensor = qtn.Tensor(
                    u_new.reshape(2, 2).conj(),
                    inds=gate.inds,
                    tags=gate.tags
                )
            elif len(gate.inds) == 4:
                new_tensor = qtn.Tensor(
                    u_new.reshape(2, 2, 2, 2).conj(),
                    inds=gate.inds,
                    tags=gate.tags
                )

            # Put the new unitary back into the circuit to perform the update
            circuit_tensor_network.add_tensor(new_tensor)

            unitary_layers[int(layer_index)][int(block_index)][2][int(tensor_index)] = qtn.Tensor(
                u_new.conj(),
                inds=["L", "R"],
                tags={"G"}
            )

            gate_index += 1

        return unitary_layers, circuit_tensor_network

    def _optimize_unitary_layers(
            self,
            mps: MPS,
            unitary_layers: list[UnitaryLayer],
            num_sweeps: int
        ) -> list[UnitaryLayer]:
        """ Optimize the unitary layers using the environment tensor updates
        to reach the optimal gates for the circuit based on [2].

        Parameters
        ----------
        `mps` : qmprs.primitives.MPS
            The MPS state to prepare.
        `unitary_layers` : list[UnitaryLayer]
            The unitary layers to be optimized.
        `num_sweeps` : int
            The number of sweeps to perform for the optimization.

        Returns
        -------
        `unitary_layers` : list[UnitaryLayer]
            The optimized unitary layers.
        """
        circuit_tensor_network = Sequential._tensor_network_from_unitary_layers(
            mps, unitary_layers
        )

        # Create a copy of the unitary layers to avoid modifying the original
        # This will be used to create the circuit after the optimization
        updated_unitary_layers = copy.deepcopy(unitary_layers)

        for _ in range(num_sweeps):
            updated_unitary_layers, circuit_tensor_network = self._sweep_unitary_layers(
                mps, circuit_tensor_network, updated_unitary_layers
            )

        return updated_unitary_layers

    def _sequential_unitary_circuit(
            self,
            mps: MPS,
            num_layers: int,
            num_sweeps: int = 0
        ) -> Circuit:
        """ Create the circuit that performs the sequential encoding of
        the MPS. The circuit is constructed using the disentangling algorithm
        described in [1].

        To improve the fidelity of the circuit, we can perform a number
        of sweeps to optimize the unitary layers using the environment
        tensor updates to reach the optimal gates for the circuit based
        on [2]. This method performs DallOall from [2].

        Parameters
        ----------
        `mps` : qmprs.primitives.MPS
            The MPS state to prepare.
        `num_layers` : int
            The number of unitary layers to prepare the MPS.
        `num_sweeps` : int, optional, default=0
            The number of sweeps to perform for the optimization.
            This is used to optimize the unitary layers using the
            environment tensor updates to reach the optimal gates
            for the circuit based on [2]. If 0, no optimization is
            performed.

        Returns
        -------
        `circuit` : quick.circuit.Circuit
            The quantum circuit preparing the MPS.
        """
        unitary_layers = self._get_unitary_layers(mps, num_layers)

        if num_sweeps > 0:
            unitary_layers = self._optimize_unitary_layers(mps, unitary_layers, num_sweeps)

        circuit = self._circuit_from_unitary_layers(
            mps,
            unitary_layers
        )

        return circuit

    def prepare_mps(
            self,
            mps: MPS,
            **kwargs
        ) -> Circuit:

        num_layers = kwargs.get("num_layers", 1)
        num_sweesps = kwargs.get("num_sweeps", 0)

        if not isinstance(num_layers, int) or num_layers < 1:
            raise ValueError("The number of layers must be a positive integer.")

        return self._sequential_unitary_circuit(mps, num_layers, num_sweesps)