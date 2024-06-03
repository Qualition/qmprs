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

# Import `qickit.circuit.Circuit`
from qickit.circuit import Circuit # type: ignore

# Import `qmprs.mps.MPS`
from qmprs.mps import MPS

# Import `qmprs.synthesis.mps_encoding.MPSEncoder`
from qmprs.synthesis.mps_encoding import MPSEncoder


class Sequential(MPSEncoder):
    """ `qmprs.synthesis.mps_encoder.Sequential` is the class for preparing MPS
    using Sequential encoding. The circuit is constructed using the disentangling
    algorithm described in the 2019 paper by Shi-Ju Ran.

    ref: https://arxiv.org/abs/1908.07958

    Notes
    -----
    - The circuit depth scales $O(N * \chi^2)$ where N is the number of qubits and $\chi$
    is the bond dimension.
    - The sequential encoding approach allows for encoding of long-range correlated states.

    Parameters
    ----------
    `circuit_framework` : Type[Circuit]
        The quantum circuit framework.
    """
    def prepare_mps(self,
                    mps: MPS,
                    **kwargs) -> Circuit:
        """ Prepare the quantum state using MPS.

        Parameters
        ----------
        `mps` : qmprs.mps.MPS
            The MPS to be prepared.
        `num_layers` : int
            The number of sequential layers. Passed as a kwarg.
        """
        # Define the number of layers
        num_layers = kwargs.get("num_layers")

        # Check if the number of layers is a positive integer
        if not isinstance(num_layers, int) or num_layers < 1:
            raise ValueError("The number of layers must be a positive integer.")

        # Normalize the MPS
        mps.normalize()

        # Compress the MPS to canonical form
        mps.compress(mode="right")

        # Define the circuit to prepare the MPS
        circuit = self.circuit_framework(mps.num_sites, mps.num_sites)

        def sequential_unitary_circuit(mps: MPS,
                                       circuit: Circuit) -> Circuit:
            """ Sequentially apply unitary layers to a MPS to prepare a target state.

            Parameters
            ----------
            `mps` : qmprs.mps.MPS
                The MPS state.
            `circuit` : Circuit
                The quantum circuit.

            Returns
            -------
            `circuit` : Circuit
                The quantum circuit preparing the MPS.
            """
            # Define the unitary layers
            unitary_layers: list = []

            # Permute the arrays to the left-right canonical form
            mps.permute(shape="lpr")
            mps.canonicalize("right", normalize=True)

            # Iterate over the number of layers
            for _ in range(num_layers):
                # Generate the unitary for the bond-d (physical dimension) compression of the MPS.
                unitary_layer = mps.generate_bond_d_unitary()

                # Append the unitary layer to the list of unitary layers
                unitary_layers.append(unitary_layer)

                # Apply the inverse unitary layer on the wavefunction
                mps.apply_unitary_layer(unitary_layer, inverse=True)

                # Normalize the MPS and convert to right canonical form
                mps.canonicalize(mode="right", normalize=True)

                # Left canonicalize and right compress the MPS (default mode)
                mps.compress()

            # Generate the quantum circuit from the unitary layers
            circuit.add(mps.circuit_from_unitary_layers(type(circuit), unitary_layers), range(mps.num_sites))

        # Define the sequential unitary circuit that prepares the target MPS with
        # the specified parameters
        sequential_unitary_circuit(mps, circuit)

        # Apply a vertical reverse
        circuit.vertical_reverse()

        return circuit