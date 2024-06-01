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

__all__ = ["MPSEncoder", "Sequential"]

from abc import ABC, abstractmethod
from typing import Type

# Import `qickit.data.Data`
from qickit.data import Data # type: ignore

# Import `qickit.circuit.Circuit`
from qickit.circuit import Circuit # type: ignore

# Import `qickit.types.collection.NestedCollection`
from qickit.types import NestedCollection # type: ignore

# Import `qmprs.mps.MPS`
from qmprs.mps import MPS

# Import `qmprs.synthesis.mps_utils.check_unitaries`
from qmprs.synthesis.mps_utils import check_unitaries


class MPSEncoder(ABC):
    """ `qmprs.synthesis.mps_encoder.MPSEncoder` is the class for preparing quantum states using Matrix Product
    States (MPS).

    Parameters
    ----------
    `circuit_framework` : Type[Circuit]
        The quantum circuit framework.
    """
    def __init__(self,
                 circuit_framework: Type[Circuit]) -> None:
        """ Initialize a `qickit.MPSEncoder` instance.
        """
        self.circuit_framework = circuit_framework

    def prepare_state(self,
                      statevector: Data | NestedCollection,
                      bond_dimension: int,
                      compression_percentage: float=0.0,
                      index_type: str="row",
                      **kwargs) -> Circuit:
        """ Prepare the quantum state using statevector.

        Parameters
        ----------
        `statevector` : qickit.data.Data | NestedCollection[NumberType]
            The statevector of the quantum system.
        `bond_dimension` : int
            The maximum bond dimension.
        `compression_percentage` : float
            The compression percentage.
        `index_type` : str
            The indexing type.
        `**kwargs`
            Additional keyword arguments.
        """
        # Check if the statevector is a `Data` instance
        if not isinstance(statevector, Data):
            statevector = Data(statevector)

        # Change the indexing (if necessary)
        statevector.change_indexing(index_type)

        # Compress the statevector
        if compression_percentage > 0.0:
            statevector.compress(compression_percentage)

        # Define an `qmprs.mps.MPS` instance
        mps = MPS(statevector, bond_dimension=bond_dimension)

        # Prepare the MPS
        return self.prepare_mps(mps, **kwargs)

    @abstractmethod
    def prepare_mps(self,
                    mps: MPS,
                    **kwargs) -> Circuit:
        """ Prepare the quantum state using MPS.

        Parameters
        ----------
        `mps` : qmprs.mps.MPS
            The matrix product state (MPS).
        `**kwargs`
            Additional keyword arguments.
        """
        pass


class Sequential(MPSEncoder):
    """ `qmprs.synthesis.mps_encoder.Sequential` is the class for preparing MPS
    using Sequential encoding. The circuit is constructed using the disentangling
    algorithm described in the 2019 paper by Shi-Ju Ran.

    ref: https://arxiv.org/abs/1908.07958

    The circuit scales $O(N * \chi^2)$ where N is the number of qubits and $\chi$
    is the bond dimension.

    Parameters
    ----------
    `circuit_framework` : Type[Circuit]
        The quantum circuit framework.
    """
    def __init__(self,
                 circuit_framework: Type[Circuit]) -> None:
        """ Initialize a `qickit.Sequential` instance.
        """
        super().__init__(circuit_framework)

    def prepare_mps(self,
                    mps: MPS,
                    **kwargs) -> Circuit:
        """ Prepare the quantum state using MPS.

        Parameters
        ----------
        `mps` : qmprs.mps.MPS
            The matrix product state (MPS).
        `num_layers` : int
            The number of sequential layers.
        """
        # Define the number of layers
        num_layers = kwargs.get("num_layers")

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
            mps.canonicalize("right")
            mps.normalize()

            # Iterate over the number of layers
            for _ in range(num_layers): # type: ignore
                # Generate the bond dimension unitary
                unitary_layer = mps.generate_bond_d_unitary()

                # Check the unitaries
                check_unitaries(unitary_layer)

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

        # Return the overlap and circuit
        return circuit