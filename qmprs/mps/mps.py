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

__all__ = ['MPS']

import numpy as np
import quimb.tensor as qtn # type: ignore

# Import `qickit.data.Data`
from qickit.data import Data # type: ignore

# Import `qickit.types.Collection` and `qickit.types.NestedCollection`
from qickit.types import Collection, NestedCollection

# Define `NumberType` as an alias for `int | float | complex`
NumberType = int | float | complex


# TODO: Confirm all methods and attributes needed for MPS support
class MPS:
    """ `qmprs.mps.MPS` is the class for creating and manipulating matrix product states (MPS).

    Parameters
    ----------
    `statevector` : qickit.data.Data | NestedCollection[NumberType]
        The statevector of the quantum system.
    `bond_dimension` : int
        The maximum bond dimension of the MPS.

    Attributes
    ----------
    `statevector` : qickit.data.Data
        The statevector of the quantum system.
    `mps` : qtn.MatrixProductState
        The matrix product state (MPS) of the quantum system.
    `bond_dimension` : int
        The maximum bond dimension of the MPS.
    `num_sites` : int
        The number of sites for the MPS.
    `physical_dimension` : int
        The physical dimension of the MPS.

    Raises
    ------
    TypeError
        If `bond_dimension` is not an integer.
    ValueError
        If `bond_dimension` is less than 1.

    Usage
    -----
    >>> statevector = [1, 2, 3, 4]
    >>> bond_dimension = 2
    >>> mps = MPS(statevector, bond_dimension)
    """
    def __init__(self,
                 statevector: Data | NestedCollection[NumberType],
                 bond_dimension: int) -> None:
        """ Initialize a `qmprs.mps.MPS` instance.
        """
        # Ensure `bond_dimension` is an integer greater than 0
        if not isinstance(bond_dimension, int):
            raise TypeError("`bond_dimension` must be an integer.")
        if bond_dimension < 1:
            raise ValueError("`bond_dimension` must be greater than 0.")

        # Ensure `statevector` is a `qickit.data.Data` instance
        if not isinstance(statevector, Data):
            statevector = Data(statevector)

        # Normalize and pad the data if necessary
        if statevector.is_normalized is False:
            statevector.normalize()
        if statevector.is_padded is False:
            statevector.pad()

        # Save the state vector for later use (i.e., re-indexing the MPS)
        self.statevector = statevector
        # Define the maximum bond dimension
        self.bond_dimension = bond_dimension
        # Define the MPS from the statevector
        self.mps = self.define_mps(statevector)
        # Define the number of sites for the MPS
        self.num_sites = self.mps.L
        # Define the physical dimension of the MPS
        if self.mps.phys_dim() != 2:
            raise ValueError("Only supports MPS with physical dimesnion=2.")
        else:
            self.physical_dimension = 2

    def define_mps(self,
                   statevector: Data) -> qtn.MatrixProductState:
        """ Define the MPS from the statevector.

        Parameters
        ----------
        `statevector` : qickit.data.Data
            The statevector of the quantum system.

        Returns
        -------
        qtn.MatrixProductState
            The MPS of the quantum system.
        """
        # Reshape the vector to N sites where N is the number of qubits
        # needed to represent the state vector
        dims = [2] * int(np.log2(len(statevector)))

        # TODO: This should be defined using C language
        # Generate MPS from the tensor arrays
        mps = qtn.MatrixProductState.from_dense(statevector, dims)

        # TODO: This should be defined using C language
        # Compress the bond dimension of the MPS to the maximum bond dimension
        for i in range(int(mps.L)-1):
            qtn.tensor_core.tensor_compress_bond(mps[i], mps[i+1], max_bond = self.bond_dimension)

        # Return the MPS
        return mps

    def normalize(self) -> None:
        """ Normalize the MPS.
        """
        # TODO: This should be defined using C language
        # Normalize the MPS
        self.mps.normalize()

    def canonize(self,
                 mode: str) -> None:
        """ Canonize the MPS.

        Parameters
        ----------
        `mode` : str
            The mode of canonization, either "left" or "right".

        Raises
        ------
        ValueError
            If `mode` is not "left" or "right".
        """
        # TODO: This should be defined using C language
        if mode == "left":
            self.mps.left_canonize()
        elif mode == "right":
            self.mps.right_canonize()
        else:
            raise ValueError("`mode` must be either 'left' or 'right'.")

    def compress(self,
                 max_bond_dimension: int) -> None:
        """ Compress the bond dimension of the MPS.

        Parameters
        ----------
        `max_bond_dimension` : int
            The maximum bond dimension of the MPS.
        `mode` : str UNIMPLEMENTED
            The mode of compression, either "left" or "right".
        """
        # TODO: Check the difference between :func:`qtn.tensor_core.tensor_compress_bond()` and :func:`qtn.compress()`. ref: sequential.py, line: 93
        # TODO: This should be defined using C language
        for i in range(int(self.mps.L)-1):
            qtn.tensor_core.tensor_compress_bond(self.mps[i], self.mps[i+1], max_bond = max_bond_dimension)

    def contract(self,
                 indices: Collection[int]) -> None:
        """ Contract the MPS.

        Parameters
        ----------
        `indices` : Collection[int]
            The indices to contract.
        """
        # TODO: This must be defined using C language (main bottleneck)
        self.mps.contract_ind(indices)

    def permute(self,
                shape: str) -> None:
        """ Permute the indices of each tensor in this MPS to match `shape`.

        Parameters
        ----------
        `shape` : str
            The shape to permute, being "lrp" or "lpr".
        """
        # TODO: This should be defined using C language
        self.mps.permute_arrays(shape)

    def change_indexing(self,
                        index_type: str) -> None:
        """ Change the indexing of the MPS to match the statevector.

        Parameters
        ----------
        `index_type` : str
            The indexing type for the statevector.
        """
        # Change the indexing of the statevector
        self.statevector.change_indexing(index_type)

        # Update the MPS definition
        self.mps = self.define_mps(self.statevector)