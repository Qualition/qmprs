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
from scipy import linalg
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
        # qtn.compress() is for 2D tensor networks. For 1D, use qtn.tensor_core.tensor_compress_bond()
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

    def get_submps_indices(self) -> list:
        """ Get the indices of the sub-MPSs.

        Returns
        -------
        `submps_indices` (list): The indices of the sub-MPSs.
        """
        # Initialize sub MPS indices
        submps_indices = []

        # Define the number of sites
        num_sites = self.mps.L

        # If the MPS only has one site
        if num_sites == 1:
            # Add the (0, 0) coordinate to the sub MPS indices
            submps_indices.append([0, 0])

        else:
            # Iterate over the sites of the MPS
            for site in range(num_sites):
                # Initialize the dimension for the left and right sides
                # of the site
                dim_left, dim_right = 1, 1

                # If this is the first site, then only define the right dimension
                if site == 0:
                    _, dim_right = self.mps[site].shape

                # If this is the last site, then only define the left dimension
                elif site == (num_sites - 1):
                    dim_left, _ = self.mps[site].shape

                # Otherwise, define both the left and right dimensions
                else:
                    dim_left, _, dim_right = self.mps[site].shape

                # If the left and right dimensions are both less than 2,
                # then add the (site, site) coordinate to the sub MPS indices
                if dim_left < 2 and dim_right < 2:
                    submps_indices.append([site, site])

                # If the left dimension is less than 2 and the right dimension
                # is greater than or equal to 2, then set the temp variable to the site
                elif dim_left < 2 and dim_right >= 2:
                    temp = site

                # If the left dimension is greater than or equal to 2 and the right dimension
                # is less than 2, then add the (temp, site) coordinate to the sub MPS indices
                elif dim_left >= 2 and dim_right < 2:
                    submps_indices.append([temp, site])

        # Return the sub MPS indices
        return submps_indices
    

    def generate_unitaries(self) -> list:
        """ Generate the unitaries for the bond-d compression of the MPS.

        Parameters
        ----------
        `mps` (qtn.MatrixProductState):
            The MPS to be compressed.

        Returns
        -------
        `generated_unitary_list` (list): A list of unitaries to be applied to the MPS.
        """
        # Define the physical dimension of the MPS
        phy_dim = self.mps.phys_dim()

        # Copy the MPS (as the MPS will be modified in place)
        mps_copy = self.mps.copy(deep=True)

        # Initialize the list of generated unitaries
        generated_unitary_list = []

        # Get the indices of the sub-MPSs
        sub_mps_indices = self.get_submps_indices(mps_copy)

        # Iterate over the sub-MPSs starting and ending indices
        for start_index, end_index in sub_mps_indices:
            # Initialize the generated unitaries, isometries, and kernels lists
            generated_unitaries, isomsetries, kernels = [], [], []

            # Iterate over the indices of the sub-MPS
            for index in range(start_index, end_index + 1):
                # If the index is the end index of the sub-MPS
                if index == (end_index):
                    # If the sub-MPS has only one site
                    if (end_index - start_index) == 0:
                        # Initialize the unitary with 0s
                        unitary = np.zeros((phy_dim, phy_dim), dtype=np.complex128)

                        # Set the first row of the unitary to the data of the MPS at the specified index
                        unitary[0, :] = mps_copy[index].data.reshape((1, -1))

                        # Set the second row of the unitary to the null space of the data of the MPS at the
                        # specified index
                        unitary[1, :] = linalg.null_space(mps_copy[index].data.reshape(1, -1).conj()).reshape(1, -1)

                    # If the sub-MPS has more than one site, the unitary is the data at the specified
                    # site
                    else:
                        unitary = mps_copy[index].data

                    # Convert the unitary to a qtn.Tensor
                    # .T at the end is useful for the application of unitaries as quantum circuit
                    unitary = qtn.Tensor(unitary.reshape((phy_dim, phy_dim)).T, inds=("v", "p"), tags={"G"})

                    # Append the unitary to the list of generated unitaries
                    generated_unitaries.append(unitary)

                    # Append the blank isometries and kernels to the lists (this is to ensure same length
                    # as the generated unitaries)
                    isomsetries.append([])
                    kernels.append([])

                # If the index is not the start index of the sub-MPS
                elif index != start_index:
                    # Initialize the unitary with 0s
                    unitary = np.zeros((phy_dim, phy_dim, phy_dim, phy_dim), dtype=np.complex128)

                    # Set the first row of the unitary to the data of the MPS at the specified index
                    unitary[0, :, :, :] = mps_copy[index].data

                    # Set the second row of the unitary to the null space of the data of the MPS at the
                    # specified index
                    kernel = linalg.null_space(mps_copy[index].data.reshape((phy_dim, -1)).conj())

                    # Multiply the kernel by 1/exp(1j * angle of the first row of the kernel)
                    kernel = kernel * (1 / np.exp(1j * np.angle(kernel[0, :])))
                    unitary[1:phy_dim, :, :, :] = kernel.reshape((phy_dim, phy_dim, phy_dim, phy_dim - 1)).transpose(
                        (3, 2, 0, 1)
                    )

                    # # CHECK IF REQUIRED
                    # Transpose the unitary, such that the indices of the unitary are ordered
                    # as unitary(L,B,T,R)
                    unitary = unitary.transpose((0, 1, 3, 2))

                    # Transpose the unitary, such that the indices of the unitary are ordered
                    # as unitary(B,L,R,T)
                    unitary = unitary.transpose((1, 0, 3, 2))

                    # Convert the unitary to a qtn.Tensor
                    # .T at the end is useful for the application of unitaries as quantum circuit
                    unitary = qtn.Tensor(unitary.reshape((phy_dim**2, phy_dim**2)).T, inds=["L", "R"], tags={"G"})

                    # Append the unitary to the list of generated unitaries
                    generated_unitaries.append(unitary)

                    # Reshape the unitary to (d x d x d x d) where d is the physical dimension
                    unitary = unitary.data.T.reshape(phy_dim, phy_dim, phy_dim, phy_dim)

                    # Get the kernel from the unitary
                    kernel = unitary[:, 1, :, :].reshape(2, 4).T

                    # Multiply the kernel by 1/exp(1j * angle of the first row of the kernel)
                    kernel = kernel * (1 / np.exp(1j * np.angle(kernel[0, :])))

                    # Define the eigenvectors and their corresponding eigenvalues from the |kernel X kernel|
                    [eigenvalues, eigenvectors] = np.linalg.eigh(kernel @ np.conj(kernel.T))

                    # Define the isometry from the eigenvectors and their corresponding eigenvalues
                    isometry = eigenvectors[:, np.where(np.abs(eigenvalues) > 1e-12)].reshape(4, -1)

                    # Append the isometry to the list of isometries
                    isomsetries.append(isometry)

                    # Append the kernel to the list of kernels
                    kernels.append(kernel)

                # If the index is the start index of the sub-MPS
                elif index == start_index:
                    # Initialize the unitary with 0s
                    unitary = np.zeros((phy_dim, phy_dim, phy_dim, phy_dim), dtype=np.complex128)

                    # Set the first row of the unitary to the data of the MPS at the specified index
                    unitary[0, 0, :, :] = mps_copy[index].data.reshape((phy_dim, -1))

                    # Get the kernel from the data of the MPS at the specified index
                    kernel = linalg.null_space(mps_copy[index].data.reshape((1, -1)).conj())

                    # Iterate over the physical dimension
                    for i in range(phy_dim):
                        # Iterate over the physical dimension
                        for j in range(phy_dim):
                            # If the indices are both 0, continue
                            if i == 0 and j == 0:
                                continue

                            # Define the index
                            index = i * phy_dim + j

                            # Set the unitary at the specified index to the kernel at the specified index
                            unitary[i, j, :, :] = kernel[:, index - 1].reshape((phy_dim, phy_dim))

                    
                    # # CHECK if required
                    # Transpose the unitary, such that the indices of the unitary are ordered
                    # as unitary(L,B,T,R)
                    unitary = unitary.transpose((0, 1, 3, 2))

                    # Transpose the unitary, such that the indices of the unitary are ordered
                    # as unitary(B,L,R,T)
                    unitary = unitary.transpose((1, 0, 3, 2))

                    # Convert the unitary to a qtn.Tensor
                    # .T at the end is useful for the application of unitaries as quantum circuit
                    unitary = qtn.Tensor(unitary.reshape((phy_dim**2, phy_dim**2)).T, inds=["L", "R"], tags={"G"})

                    # Append the unitary to the list of generated unitaries
                    generated_unitaries.append(unitary)

                    # Reshape the unitary to (d x d x d x d) where d is the physical dimension
                    unitary = unitary.data.T.reshape((phy_dim, phy_dim, phy_dim, phy_dim))

                    # Get the kernel from the unitary
                    kernel = unitary[:, 1, :, :].reshape(2, 4).T
                    kernel = np.c_[unitary[1, 0, :, :].reshape(1, 4).T, kernel]

                    # Is scipy better or numpy?
                    # Define the eigenvectors and their corresponding eigenvalues from the |kernel X kernel|
                    [eigenvalues, eigenvectors] = np.linalg.eigh(kernel @ np.conj(kernel.T))

                    # Define the isometry from the eigenvectors and their corresponding eigenvalues
                    isometry = eigenvectors[:, np.where(np.abs(eigenvalues) > 1e-12)].reshape(4, -1)

                    # Append the isometry to the list of isometries
                    isomsetries.append(isometry)

                    # Append the kernel to the list of kernels
                    kernels.append(kernel)

            # Append the start index, end index, generated unitaries, isometries, and kernels to the
            generated_unitary_list.append([start_index, end_index, generated_unitaries, isomsetries, kernels])

        # Return the generated unitary list
        return generated_unitary_list

    # CHECKED
    def generate_bond_d_unitary(self,
                                mps: qtn.MatrixProductState) -> list:
        """ Generate the unitary for the bond-d compression of the MPS.

        Parameters
        ----------
        `mps` (qtn.MatrixProductState):
            The MPS to be compressed.

        Returns
        -------
        `generated_unitary_list` (list): A list of unitaries to be applied to the MPS.
        """
        # Define the physical dimension of the MPS
        phy_dim = self.mps.phys_dim()

        # Copy the MPS (as the MPS will be modified in place)
        mps_copy = self.mps.copy(deep=True)

        # Compress the MPS to a bond dimension of phy_dim
        mps_copy.compress(max_bond=phy_dim)

        # Right canonize the compressed MPS
        mps_copy.canonize(mode = 'right')

        # Generate the unitaries
        generated_unitary_list = self.generate_unitaries(mps_copy)

        # Return the generated unitary list
        return generated_unitary_list