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

import copy
import numpy as np
import quimb.tensor as qtn # type: ignore
from scipy import linalg # type: ignore
from typing import Optional, Union

# Import `qickit.data.Data`
from qickit.data import Data # type: ignore

# Import `qickit.types.Collection` and `qickit.types.NestedCollection`
from qickit.types import Collection, NestedCollection # type: ignore

# Define `NumberType` as an alias for `int | float | complex`
NumberType = int | float | complex


# TODO: Confirm all methods and attributes needed for MPS support
class MPS:
    """ `qmprs.mps.MPS` is the class for creating and manipulating matrix product states (MPS).

    Parameters
    ----------
    `statevector` : qickit.data.Data | NestedCollection[NumberType], optional
        The statevector of the quantum system.
    `mps` : qtn.MatrixProductState, optional
        The matrix product state (MPS) of the quantum system.
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
        `bond_dimension` must be an integer greater than 0.
        Cannot initialize with both `statevector` and `mps`.
        Must provide either `statevector` or `mps`.
        Only supports MPS with physical dimension=2.

    Usage
    -----
    >>> statevector = [1, 2, 3, 4]
    >>> bond_dimension = 2
    >>> mps = MPS(statevector, bond_dimension)
    """
    def __init__(self,
                 statevector: Optional[Union[Data, NestedCollection[NumberType]]],
                 mps: Optional[qtn.MatrixProductState],
                 bond_dimension: int) -> None:
        """ Initialize a `qmprs.mps.MPS` instance. Pass only `statevector` to define
        the MPS from the statevector. Pass only `mps` to define the MPS from the MPS.
        """
        # Ensure `bond_dimension` is an integer greater than 0
        if not isinstance(bond_dimension, int) or bond_dimension < 1:
            raise ValueError("`bond_dimension` must be an integer greater than 0.")

        # Determine initialization path
        if statevector is not None and mps is not None:
            raise ValueError("Cannot initialize with both `statevector` and `mps`.")

        elif statevector is not None:
            # Initialize from statevector
            if not isinstance(statevector, Data):
                statevector = Data(statevector)
            statevector.to_quantumstate()
            self.statevector = statevector
            self.mps = self.from_statevector(statevector, bond_dimension)

        elif mps is not None:
            # Initialize from MPS
            if not isinstance(mps, qtn.MatrixProductState):
                raise TypeError("`mps` must be a `qtn.MatrixProductState` instance.")
            self.mps = mps
            self.statevector = self.to_statevector(mps)

        else:
            raise ValueError("Must provide either `statevector` or `mps`.")

        # Define the maximum bond dimension
        self.bond_dimension = bond_dimension

        # Define the number of sites for the MPS
        self.num_sites = self.statevector.num_qubits

        # Define the physical dimension of the MPS
        if self.mps.phys_dim() != 2:
            raise ValueError("Only supports MPS with physical dimension=2.")
        else:
            self.physical_dimension = 2

    @staticmethod
    def from_statevector(statevector: Data,
                         max_bond_dimension: int) -> qtn.MatrixProductState:
        """ Define the MPS from the statevector.

        Parameters
        ----------
        `statevector` : qickit.data.Data
            The statevector of the quantum system.

        Returns
        -------
        `mps` : qtn.MatrixProductState
            The MPS of the quantum system.
        """
        # Define the number of sites
        num_sites = statevector.num_qubits

        # Reshape the vector to N sites where N is the number of qubits
        # needed to represent the state vector
        site_dimensions = [2] * num_sites

        # Generate MPS from the tensor arrays
        mps = qtn.MatrixProductState.from_dense(statevector, site_dimensions)

        # Compress the bond dimension of the MPS to the maximum bond dimension specified
        for i in range(num_sites-1):
            qtn.tensor_core.tensor_compress_bond(mps[i], mps[i+1], max_bond = max_bond_dimension)

        return mps

    @staticmethod
    def to_statevector(mps: qtn.MatrixProductState) -> Data:
        """ Convert the MPS to a statevector.

        Parameters
        ----------
        `mps` : qtn.MatrixProductState
            The matrix product state (MPS) of the quantum system.

        Returns
        -------
        `statevector` : qickit.data.Data
            The statevector of the quantum system.
        """
        # Define the statevector from the MPS using `.to_dense()` method
        statevector = Data(mps.to_dense())

        # Convert the statevector to a quantum state
        statevector.to_quantumstate()

        return statevector

    def normalize(self) -> None:
        """ Normalize the MPS.
        """
        self.mps.normalize()

    def canonicalize(self,
                     mode: str,
                     normalize=False) -> None:
        """ Canonicalize the MPS with the specified mode.

        Parameters
        ----------
        `mode` : str
            The mode of canonicalization, either "left" or "right".
        `normalize` : bool, optional
            Whether to normalize the state. This is different from the `.normalize()` method.

        Raises
        ------
        ValueError
            If `mode` is not "left" or "right".
        """
        if mode == "left":
            self.mps.left_canonize(normalize=normalize)
        elif mode == "right":
            self.mps.right_canonize(normalize=normalize)
        else:
            raise ValueError("`mode` must be either 'left' or 'right'.")

    def compress(self,
                 max_bond_dimension: int,
                 mode="") -> None:
        """ Compress the bond dimension of the MPS.

        Parameters
        ----------
        `max_bond_dimension` : int
            The maximum bond dimension of the MPS.
        `mode` : str, optional
            The mode of compression, either "left", "right", or "flat".

        Raises
        ------
        TypeError
            If `max_bond_dimension` is not an integer.
        ValueError
            If `mode` is not "left", "right", or "flat".
        """
        if not isinstance(max_bond_dimension, int):
            raise TypeError("`max_bond_dimension` must be an integer.")

        # If `mode` is specified, compress the MPS with the specified mode
        if mode == "":
            for i in range(self.num_sites-1):
                qtn.tensor_core.tensor_compress_bond(self.mps[i], self.mps[i+1], max_bond = max_bond_dimension)
        else:
            if mode in ["left", "right", "flat"]:
                self.mps.compress(form=mode, max_bond=max_bond_dimension)
            else:
                raise ValueError(f"`mode` must be either 'left', 'right', or 'flat'. Received {mode}.")

    def contract(self,
                 indices: Collection[int]) -> None:
        """ Contract the MPS.

        Parameters
        ----------
        `indices` : Collection[int]
            The indices to contract.

        Raises
        ------
        TypeError
            If `indices` is not a collection of integers.
        TypeError
            If any element of `indices` is not an integer.
        """
        if not isinstance(indices, Collection):
            raise TypeError("`indices` must be a collection of integers.")
        elif not all(isinstance(index, int) for index in indices):
            raise TypeError("All elements of `indices` must be integers.")

        self.mps.contract_ind(indices)

    def permute(self,
                shape: str) -> None:
        """ Permute the indices of each tensor in this MPS to match `shape`.

        Parameters
        ----------
        `shape` : str
            The shape to permute, being "lrp" or "lpr".

        Raises
        ------
        ValueError
            If `shape` is not "lrp" or "lpr".
        """
        if shape in ["lrp", "lpr"]:
            self.mps.permute_arrays(shape)
        else:
            raise ValueError(f"`shape` must be either 'lrp' or 'lpr'. Received {shape}.")

    def change_indexing(self,
                        index_type: str) -> None:
        """ Change the indexing of the MPS to match the statevector.

        Parameters
        ----------
        `index_type` : str
            The indexing type for the statevector, being "row" or "snake".
        """
        # Change the indexing of the statevector
        self.statevector.change_indexing(index_type)

        # Update the MPS definition
        self.mps = MPS.from_statevector(self.statevector, self.bond_dimension)

    # TODO: Redo the comments for better clarity.
    def get_submps_indices(self) -> list[tuple[int, int]]:
        """ Get the indices of the tensors at each site of the MPS.

        Returns
        -------
        `submps_indices` : (list[tuple[int, int]])
            The indices of the MPS tensors.
        """
        # Initialize the indices of the tensors at each site of the MPS
        submps_indices = []

        # If the MPS only has one site, add the (0, 0) coordinate to the sub MPS indices
        if self.num_sites == 1:
            return [(0, 0)]

        # Otherwise, iterate over the sites of the MPS to define the left (first) and right (last)
        # sites' dimensions
        for site in range(self.num_sites):
            # Initialize the dimension for the first and last sites
            dim_left, dim_right = 1, 1

            # If this is the first site, then only define the right dimension
            if site == 0:
                _, dim_right = self.mps[site].shape

            # If this is the last site, then only define the left dimension
            elif site == (self.num_sites - 1):
                dim_left, _ = self.mps[site].shape

            # Otherwise, define both the left and right dimensions for the intermediate sites
            else:
                dim_left, _, dim_right = self.mps[site].shape

            # If the left and right dimensions are both less than 2,
            # then add the (site, site) coordinate to the sub MPS indices
            if dim_left < 2 and dim_right < 2:
                submps_indices.append((site, site))

            # If the left dimension is less than 2 and the right dimension
            # is greater than or equal to 2, then set the temp variable to the site
            elif dim_left < 2 and dim_right >= 2:
                temp = site

            # If the left dimension is greater than or equal to 2 and the right dimension
            # is less than 2, then add the (temp, site) coordinate to the sub MPS indices
            elif dim_left >= 2 and dim_right < 2:
                submps_indices.append((temp, site))

        return submps_indices

    def generate_unitaries(self):
        """ Generate the unitaries of the MPS.

        Returns
        -------
        `generated_unitary_list` : list
            A list of unitaries to be applied to the MPS.
        """
        # Define the physical dimension of the MPS
        phy_dim = self.physical_dimension

        # Copy the MPS (as the MPS will be modified in place)
        mps_copy = copy.deepcopy(self.mps)

        # Initialize the list of generated unitaries
        generated_unitary_list = []

        # Get the indices of the MPS tensors
        sub_mps_indices = self.get_submps_indices()

        # Iterate over the tensors' starting and ending indices
        for start_index, end_index in sub_mps_indices:
            
            # Initialize lists to store generated unitaries, isometries, and kernels
            generated_unitaries, isomsetries, kernels = [], [], []

            # Iterate over the range from start_index to end_index (inclusive)
            for index in range(start_index, end_index + 1):

                if index == end_index:
                    # Generate a single site unitary for the current tensor
                    self._generate_single_site_unitary(mps_copy[index].data, start_index, end_index, generated_unitaries, isomsetries, kernels)
                    
                elif index != start_index:
                    # Generate a two site unitary for the current tensor
                    self._generate_two_site_unitary(mps_copy[index].data, generated_unitaries, isomsetries, kernels)
                
                else:
                    # Generate a first site unitary for the current tensor
                    self._generate_first_site_unitary(mps_copy[index].data, generated_unitaries, isomsetries, kernels)

            # Append the start_index, end_index, generated unitaries, isometries, and kernels to the list of generated unitaries
            generated_unitary_list.append([start_index, end_index, generated_unitaries, isomsetries, kernels])

        # Return the list of generated unitaries
        return generated_unitary_list


    def _generate_single_site_unitary(self,
                                      mps_copy_data: np.ndarray,
                                      start_index: int,
                                      end_index: int,
                                      generated_unitaries: list[qtn.Tensor],
                                      isomsetries: list[list],
                                      kernels: list[list]) -> None:
        """
        Generates a single site unitary for a given tensor in the MPS.

        Parameters
        ----------
        mps_copy_data : np.ndarray
            The data of the MPS tensor at the specified index.
        start_index : int
            The starting index of the sub-MPS.
        end_index : int
            The ending index of the sub-MPS.
        generated_unitaries : List[qtn.Tensor]
            The list of generated unitaries.
        isomsetries : List[List]
            The list of isometries.
        kernels : List[List]
            The list of kernels.
        """
        # Define the physical dimension of the MPS
        phy_dim = self.physical_dimension

        # Check if the sub-MPS has only one site
        if end_index == start_index:
            # Initialize the unitary with 0s
            unitary = np.zeros((phy_dim, phy_dim), dtype=np.complex128)

            # Set the first row of the unitary to the data of the MPS at the specified index
            unitary[0, :] = mps_copy_data.reshape((1, -1))

            # Set the second row of the unitary to the null space of the data of the MPS at the specified index
            unitary[1, :] = linalg.null_space(mps_copy_data.reshape(1, -1).conj()).reshape(1, -1)
        else:
            # If the sub-MPS has more than one site, the unitary is the MPS tensor at the specified site
            unitary = mps_copy_data

        # Convert the unitary to a qtn.Tensor
        # .T at the end is useful for the application of unitaries as quantum circuit
        unitary = qtn.Tensor(unitary.reshape((phy_dim, phy_dim)).T, inds=("v", "p"), tags={"G"})

        # Append the unitary to the list of generated unitaries
        generated_unitaries.append(unitary)

        # Append the blank isometries and kernels to the lists (this is to ensure same length as the generated unitaries)
        isomsetries.append([])
        kernels.append([])

    def _generate_two_site_unitary(self,
                                mps_copy_data: np.ndarray,
                                generated_unitaries: list[qtn.Tensor],
                                isomsetries: list[np.ndarray],
                                kernels: list[np.ndarray]) -> None:
        """
        Generates a two site unitary for a given tensor in the MPS.

        Parameters
        ----------
        mps_copy_data : np.ndarray
            The data of the MPS tensor at the specified index.
        generated_unitaries : List[qtn.Tensor]
            The list of generated unitaries.
        isomsetries : List[np.ndarray]
            The list of isometries.
        kernels : List[np.ndarray]
            The list of kernels.
        """
        # Define the physical dimension of the MPS
        phy_dim = self.physical_dimension

        # Initialize the unitary with 0s
        unitary = np.zeros((phy_dim, phy_dim, phy_dim, phy_dim), dtype=np.complex128)

        # Set the first row of the unitary to the MPS tensor at the specified site
        unitary[0, :, :, :] = mps_copy_data

        # Set the second row of the unitary to the null space of the MPS tensor at the specified site
        kernel = linalg.null_space(mps_copy_data.reshape((phy_dim, -1)).conj())

        # Multiply the kernel by 1/exp(1j * angle of the first row of the kernel)
        kernel = kernel * (1 / np.exp(1j * np.angle(kernel[0, :])))
        unitary[1:phy_dim, :, :, :] = kernel.reshape((phy_dim, phy_dim, phy_dim, phy_dim - 1)).transpose((3, 2, 0, 1))

        # Transpose the unitary, such that the indices of the unitary are ordered as unitary(L,B,T,R)
        unitary = unitary.transpose((0, 1, 3, 2))

        # Transpose the unitary, such that the indices of the unitary are ordered as unitary(B,L,R,T)
        unitary = unitary.transpose((1, 0, 3, 2))

        # Convert the unitary to a qtn.Tensor
        # .T at the end is useful for the application of unitaries as quantum circuit
        unitary = qtn.Tensor(unitary.reshape((phy_dim**2, phy_dim**2)).T, inds=["L", "R"], tags={"G"})

        # Append the unitary to the list of generated unitaries
        generated_unitaries.append(unitary)

        # Reshape the unitary to (d x d x d x d) where d is the physical dimension
        unitary = unitary.T.reshape(phy_dim, phy_dim, phy_dim, phy_dim)

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

    def _generate_first_site_unitary(self,
                                    mps_copy_data: np.ndarray,
                                    generated_unitaries: list[qtn.Tensor],
                                    isomsetries: list[np.ndarray],
                                    kernels: list[np.ndarray]) -> None:
        """
        Generates a first site unitary for a given tensor in the MPS.

        Parameters
        ----------
        mps_copy_data : np.ndarray
            The data of the MPS tensor at the specified index.
        generated_unitaries : List[qtn.Tensor]
            The list of generated unitaries.
        isomsetries : List[np.ndarray]
            The list of isometries.
        kernels : List[np.ndarray]
            The list of kernels.
        """
        # Define the physical dimension of the MPS
        phy_dim = self.physical_dimension

        # Initialize the unitary with 0s
        unitary = np.zeros((phy_dim, phy_dim, phy_dim, phy_dim), dtype=np.complex128)

        # Set the first row of the unitary to the data of the MPS at the specified index
        unitary[0, 0, :, :] = mps_copy_data.reshape((phy_dim, -1))

        # Get the kernel from the data of the MPS at the specified index
        kernel = linalg.null_space(mps_copy_data.reshape((1, -1)).conj())

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

        # Transpose the unitary, such that the indices of the unitary are ordered as unitary(L,B,T,R)
        unitary = unitary.transpose((0, 1, 3, 2))

        # Transpose the unitary, such that the indices of the unitary are ordered as unitary(B,L,R,T)
        unitary = unitary.transpose((1, 0, 3, 2))

        # Convert the unitary to a qtn.Tensor
        # .T at the end is useful for the application of unitaries as quantum circuit
        unitary = qtn.Tensor(unitary.reshape((phy_dim**2, phy_dim**2)).T, inds=["L", "R"], tags={"G"})

        # Append the unitary to the list of generated unitaries
        generated_unitaries.append(unitary)

        # Reshape the unitary to (d x d x d x d) where d is the physical dimension
        unitary = unitary.T.reshape((phy_dim, phy_dim, phy_dim, phy_dim))

        # Get the kernel from the unitary
        kernel = unitary[:, 1, :, :].reshape(2, 4).T
        kernel = np.c_[unitary[1, 0, :, :].reshape(1, 4).T, kernel]

        # Define the eigenvectors and their corresponding eigenvalues from the |kernel X kernel|
        [eigenvalues, eigenvectors] = np.linalg.eigh(kernel @ np.conj(kernel.T))

        # Define the isometry from the eigenvectors and their corresponding eigenvalues
        isometry = eigenvectors[:, np.where(np.abs(eigenvalues) > 1e-12)].reshape(4, -1)

        # Append the isometry to the list of isometries
        isomsetries.append(isometry)

        # Append the kernel to the list of kernels
        kernels.append(kernel)

    
    
    # TODO: Redo the comments for better clarity.
    def generate_bond_d_unitary(self) -> list:
        """ Generate the unitary for the bond-d compression of the MPS.

        Returns
        -------
        `generated_unitary_list` : list
            A list of unitaries to be applied to the MPS.
        """
        # Copy the MPS (as the MPS will be modified in place with `.compress` and `.canonicalize` methods)
        mps_copy = copy.deepcopy(self)

        # Compress the MPS to a bond dimension of the physical dimension of the MPS
        mps_copy.compress(self.physical_dimension)

        # Right canonicalize the compressed MPS
        mps_copy.canonicalize('right')

        # Generate the unitaries
        generated_unitary_list = mps_copy.generate_unitaries()

        return generated_unitary_list