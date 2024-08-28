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
from qickit.circuit import Circuit # type: ignore

from qmprs.mps import MPS
from qmprs.synthesis.mps_encoding import MPSEncoder


class Sequential(MPSEncoder):
    """ `qmprs.synthesis.mps_encoder.Sequential` is the class for preparing MPS
    using Sequential encoding. The circuit is constructed using the disentangling
    algorithm described in the 2019 paper by Shi-Ju Ran.

    [1] Ran, Shi-Ju. “Encoding of Matrix Product States into Quantum Circuits of One- and Two-Qubit Gates.”
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
    as Mottonen or Shende.

    Parameters
    ----------
    `circuit_framework` : type[qickit.circuit.Circuit]
        The quantum circuit framework.

    Attributes
    ----------
    `circuit_framework` : type[qickit.circuit.Circuit]
        The quantum circuit framework.

    Raises
    ------
    ValueError
        If the number of layers is not a positive integer.

    Usage
    -----
    >>> sequential = Sequential(Circuit)
    """
    def prepare_mps(self,
                    mps: MPS,
                    **kwargs) -> Circuit:
        num_layers = kwargs.get("num_layers")

        if not isinstance(num_layers, SupportsIndex) or num_layers < 1: # type: ignore
            raise ValueError("The number of layers must be a positive integer.")

        # Create a copy of the MPS as the operations are applied inplace
        mps_copy = copy.deepcopy(mps)

        mps_copy.normalize()
        mps_copy.compress(mode="right")

        def sequential_unitary_circuit(mps: MPS) -> Circuit:
            """ Construct the unitary matrix product operators that optimally disentangle
            the MPS to a product state. These matrix product operators form the quantum
            circuit that evolves a product state to the targeted MPS.

            Parameters
            ----------
            `mps` : qmprs.mps.MPS
                The MPS state.

            Returns
            -------
            `circuit` : qickit.circuit.Circuit
                The quantum circuit preparing the MPS.
            """
            unitary_layers: list = []

            mps.permute(shape="lpr")
            mps.canonicalize("right", normalize=True)

            for _ in range(num_layers):
                unitary_layer = mps.generate_bond_d_unitary()
                unitary_layers.append(unitary_layer)
                mps.apply_unitary_layer(unitary_layer, inverse=True)
                mps.canonicalize(mode="right", normalize=True)
                mps.compress()

            circuit = mps.circuit_from_unitary_layers(self.circuit_framework, unitary_layers)
            circuit.vertical_reverse()

            # Transpile the circuit to merge the sequence of U3 gates into a single U3 gate
            # circuit.transpile()

            return circuit

        return sequential_unitary_circuit(mps_copy)