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
from numpy.typing import NDArray
import quimb.tensor as qtn # type: ignore
from scipy import linalg # type: ignore
from typing import Type, Union

# Import `qickit.data.Data`
from qickit.data import Data # type: ignore

# Import `qickit.circuit.Circuit`
from qickit.circuit import Circuit # type: ignore

# Import `qickit.types.Collection` and `qickit.types.NestedCollection`
from qickit.types import Collection, NestedCollection # type: ignore

# Define `NumberType` as an alias for `int | float | complex`
NumberType = int | float | complex


# TODO: Confirm all methods and attributes needed for MPS support
# TODO: Add string diagrams for visual documentation where possible
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
                 statevector: Union[Data, NestedCollection[NumberType]] | None = None,
                 mps: qtn.MatrixProductState | None = None,
                 bond_dimension: int = 64) -> None:
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

        Usage
        -----
        >>> statevector = Data([1, 2, 3, 4])
        >>> max_bond_dimension = 2
        >>> mps = MPS.from_statevector(statevector, max_bond_dimension)
        """
        # Define the number of sites
        num_sites = statevector.num_qubits

        # Reshape the vector to N sites where N is the number of qubits
        # needed to represent the state vector
        site_dimensions = [2] * num_sites

        # Generate MPS from the tensor arrays
        mps = qtn.MatrixProductState.from_dense(statevector.data, site_dimensions)

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

        Usage
        -----
        >>> mps = qtn.MatrixProductState.random([2, 2, 2, 2], 2)
        >>> statevector = MPS.to_statevector(mps)
        """
        # Define the statevector from the MPS using `.to_dense()` method
        statevector = Data(mps.to_dense())

        # Convert the statevector to a quantum state
        statevector.to_quantumstate()

        return statevector

    def normalize(self) -> None:
        """ Normalize the MPS.

        Usage
        -----
        >>> mps.normalize()
        """
        self.mps.normalize()

    def canonicalize(self,
                     mode: str,
                     normalize=False) -> None:
        """ Canonicalize the MPS with the specified mode. This states how the singular values from
        the SVD are absorbed into the left or right tensors.
        - If `mode` is "left", the singular values are absorbed into the tensors to their
        right. (all tensors contract to unit matrix from left)

                          i              i
            >->->->->->->-o-o-         +-o-o-
            | | | | | | | | | ...  =>  | | | ...
            >->->->->->->-o-o-         +-o-o-

        - If `mode` is "right", the singular values are absorbed into the tensors to their
        left. (all tensors contract to unit matrix from right)

                   i                           i
                -o-o-<-<-<-<-<-<-<          -o-o-+
             ... | | | | | | | | |   ->  ... | | |
                -o-o-<-<-<-<-<-<-<          -o-o-+

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

        Usage
        -----
        >>> mps.canonicalize("left")
        >>> mps.canonicalize("right")
        """
        if mode == "left":
            self.mps.left_canonize(normalize=normalize)
        elif mode == "right":
            self.mps.right_canonize(normalize=normalize)
        else:
            raise ValueError("`mode` must be either 'left' or 'right'.")

    def compress(self,
                 max_bond_dimension: int | None = None,
                 mode: str | None = None) -> None:
        """ SVD Compress the bond dimension of the MPS.

         a)│   │        b)│        │        c)│       │
         ━━●━━━●━━  ->  ━━>━━○━━○━━<━━  ->  ━━>━━━M━━━<━━
           │   │          │  ....  │          │       │
          <*> <*>          contract              <*>
          QR   LQ            -><-                SVD

            d)│            │        e)│   │
        ->  ━━>━━━ML──MR━━━<━━  ->  ━━●───●━━
              │....    ....│          │   │
            contract  contract          ^compressed bond
               -><-      -><-

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
            If `max_bond_dimension` is not specified and `mode` is not specified.
            If `mode` is not "left", "right", or "flat".

        Usage
        -----
        >>> mps.compress(max_bond_dimension=16)
        """
        if not (max_bond_dimension and mode):
            raise ValueError("At least `max_bond_dimension` or `mode` must be specified.")
        if not isinstance(max_bond_dimension, int):
            raise TypeError("`max_bond_dimension` must be an integer.")

        # If `mode` is specified, compress the MPS with the specified mode
        if not mode:
            for i in range(self.num_sites-1):
                qtn.tensor_core.tensor_compress_bond(self.mps[i], self.mps[i+1], max_bond = max_bond_dimension)
        else:
            if mode in ["left", "right", "flat"]:
                if not max_bond_dimension:
                    self.mps.compress(form=mode)
                else:
                    self.mps.compress(form=mode, max_bond=max_bond_dimension)
            else:
                raise ValueError(f"`mode` must be either 'left', 'right', or 'flat'. Received {mode}.")

    def contract(self,
                 indices: Collection[int]) -> None:
        """ Contract tensors connected by the given indices.

        Parameters
        ----------
        `indices` : Collection[int]
            The indices to contract.

        Raises
        ------
        TypeError
            If `indices` is not a collection of integers.
            If any element of `indices` is not an integer.

        Usage
        -----
        >>> mps.contract([0, 1])
        """
        if not isinstance(indices, Collection):
            raise TypeError("`indices` must be a collection of integers.")
        elif not all(isinstance(index, int) for index in indices):
            raise TypeError("All elements of `indices` must be integers.")

        self.mps.contract_ind(indices)

    # TODO
    def polar_decompose(self) -> list:
        """ Perform a polar decomposition on the MPS to retrieve the
        isometries V and positive semidefinite matrix P.

        Returns
        -------
        `isometries` : list
            A list of isometries.
        `positive_semidefinite_matrix` : list
            A list of positive semidefinite matrices.

        Usage
        -----
        >>> mps.polar_decompose()
        """
        return []

    # TODO
    def block_sites(self,
                    block_size: int) -> None:
        pass

    def permute(self,
                shape: str) -> None:
        """ Permute the indices of each tensor in the MPS to match `shape`.

        Parameters
        ----------
        `shape` : str
            The shape to permute, being "lrp" or "lpr".

        Raises
        ------
        ValueError
            If `shape` is not "lrp" or "lpr".

        Usage
        -----
        >>> mps.permute("lrp")
        >>> mps.permute("lpr")
        """
        if shape in ["lrp", "lpr"]:
            self.mps.permute_arrays(shape)
        else:
            raise ValueError(f"`shape` must be either 'lrp' or 'lpr'. Received {shape}.")

    def change_indexing(self,
                        index_type: str) -> None:
        """ Change the indexing of the MPS to match the statevector. This operation basically
        changes the indexing order of the statevector, and then re-defines the MPS for the
        updated statevector.

        Parameters
        ----------
        `index_type` : str
            The indexing type for the statevector, being "row" or "snake".

        Usage
        -----
        >>> mps.change_indexing("row")
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
        `submps_indices` : list[tuple[int, int]]
            The indices of the MPS tensors.

        Usage
        -----
        >>> mps.get_submps_indices()
        """
        # Initialize the indices of the tensors at each site of the MPS
        submps_indices = []

        # If the MPS only has one site, only add the (0, 0) coordinate to the sub MPS indices
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

    def generate_unitaries(self) -> list:
        """ Generate the unitaries of the MPS.

        Returns
        -------
        `generated_unitary_list` : list
            A list of unitaries to be applied to the MPS.

        Usage
        -----
        >>> mps.generate_unitaries()
        """
        # Copy the MPS (as the MPS will be modified in place)
        mps_copy = copy.deepcopy(self.mps)

        # Initialize the list of generated unitaries
        generated_unitary_list = []

        # Get the indices of the MPS tensors
        sub_mps_indices = self.get_submps_indices()

        # Iterate over the tensors' starting and ending indices
        for start_index, end_index in sub_mps_indices:
            generated_unitaries: list = []
            isomsetries: list = []
            kernels: list = []

            # Iterate over the range from start_index to end_index (inclusive)
            for index in range(start_index, end_index + 1):
                if index == end_index:
                    # Generate a single site unitary for the current tensor
                    generated_unitaries, isomsetries, kernels = self._generate_single_site_unitary(mps_copy[index].data,
                                                                                                   start_index,
                                                                                                   end_index,
                                                                                                   generated_unitaries,
                                                                                                   isomsetries,
                                                                                                   kernels)

                elif index != start_index:
                    # Generate a two site unitary for the current tensor
                    generated_unitaries, isomsetries, kernels = self._generate_two_site_unitary(mps_copy[index].data,
                                                                                                generated_unitaries,
                                                                                                isomsetries,
                                                                                                kernels)

                else:
                    # Generate a first site unitary for the current tensor
                    generated_unitaries, isomsetries, kernels = self._generate_first_site_unitary(mps_copy[index].data,
                                                                                                  generated_unitaries,
                                                                                                  isomsetries,
                                                                                                  kernels)

            generated_unitary_list.append([start_index,
                                           end_index,
                                           generated_unitaries,
                                           isomsetries,
                                           kernels])

        return generated_unitary_list

    # TODO: Redo the comments for better clarity.
    def _generate_two_site_unitary(self,
                                   mps_data: NDArray[np.complex128],
                                   generated_unitaries: list[qtn.Tensor],
                                   isometries: list[NDArray],
                                   kernels: list[NDArray]) -> list:
        """ Generate a two site unitary for a given tensor in the MPS.

        Parameters
        ----------
        mps_data : NDArray[np.complex128]
            The data of the MPS tensor at the specified index.
        generated_unitaries : list[qtn.Tensor]
            The list of generated unitaries.
        isometries : list[NDArray]
            The list of isometries.
        kernels : list[NDArray]
            The list of kernels.

        Returns
        -------
        `generated_unitaries` : list[qtn.Tensor]
            The list of generated unitaries.
        `isometries` : list[NDArray]
            The list of isometries.
        `kernels` : list[NDArray]
            The list of kernels.
        """
        # Define the physical dimension of the MPS
        phy_dim = self.physical_dimension

        # Initialize the unitary with 0s
        unitary = np.zeros((phy_dim, phy_dim, phy_dim, phy_dim), dtype=np.complex128)

        # Set the first row of the unitary to the MPS tensor at the specified site
        unitary[0, :, :, :] = mps_data

        # Set the second row of the unitary to the null space of the MPS tensor at the specified site
        kernel = linalg.null_space(mps_data.reshape((phy_dim, -1)).conj())

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
        isometries.append(isometry)

        # Append the kernel to the list of kernels
        kernels.append(kernel)

        return [generated_unitaries, isometries, kernels]

    # TODO: Redo the comments for better clarity.
    def _generate_first_site_unitary(self,
                                     mps_data: np.ndarray,
                                     generated_unitaries: list[qtn.Tensor],
                                     isometries: list[NDArray[np.number]],
                                     kernels: list[NDArray[np.number]]) -> list:
        """ Generate the first site unitary for a given tensor in the MPS.

        Parameters
        ----------
        mps_data : NDArray[np.number]
            The data of the MPS tensor at the specified index.
        generated_unitaries : List[qtn.Tensor]
            The list of generated unitaries.
        isometries : list[NDArray[np.number]]
            The list of isometries.
        kernels : list[NDArray[np.number]]
            The list of kernels.

        Returns
        -------
        `generated_unitaries` : list[qtn.Tensor]
            The list of generated unitaries.
        `isometries` : list[NDArray]
            The list of isometries.
        `kernels` : list[NDArray]
            The list of kernels.
        """
        # Define the physical dimension of the MPS
        phy_dim = self.physical_dimension

        # Initialize the unitary with 0s
        unitary = np.zeros((phy_dim, phy_dim, phy_dim, phy_dim), dtype=np.complex128)

        # Set the first row of the unitary to the data of the MPS at the specified index
        unitary[0, 0, :, :] = mps_data.reshape((phy_dim, -1))

        # Get the kernel from the data of the MPS at the specified index
        kernel = linalg.null_space(mps_data.reshape((1, -1)).conj())

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
        isometries.append(isometry)

        # Append the kernel to the list of kernels
        kernels.append(kernel)

        return [generated_unitaries, isometries, kernels]

    # TODO: Redo the comments for better clarity.
    def _generate_single_site_unitary(self,
                                      mps_data: NDArray[np.complex128],
                                      start_index: int,
                                      end_index: int,
                                      generated_unitaries: list[qtn.Tensor],
                                      isometries: list[NDArray],
                                      kernels: list[NDArray]) -> list:
        """ Generate a single site unitary for a given tensor in the MPS.

        Parameters
        ----------
        mps_data : NDArray[np.complex128]
            The data of the MPS tensor at the specified index.
        start_index : int
            The starting index of the sub-MPS.
        end_index : int
            The ending index of the sub-MPS.
        generated_unitaries : List[qtn.Tensor]
            The list of generated unitaries.
        isometries : list[NDArray]
            The list of isometries.
        kernels : list[NDArray]
            The list of kernels.

        Returns
        -------
        `generated_unitaries` : list[qtn.Tensor]
            The list of generated unitaries.
        `isometries` : list[NDArray]
            The list of isometries.
        `kernels` : list[NDArray]
            The list of kernels.
        """
        # Define the physical dimension of the MPS
        phy_dim = self.physical_dimension

        # Check if the sub-MPS has only one site
        if end_index == start_index:
            # Initialize the unitary with 0s
            unitary = np.zeros((phy_dim, phy_dim), dtype=np.complex128)

            # Set the first row of the unitary to the data of the MPS at the specified index
            unitary[0, :] = mps_data.reshape((1, -1))

            # Set the second row of the unitary to the null space of the data of the MPS at the specified index
            unitary[1, :] = linalg.null_space(mps_data.reshape(1, -1).conj()).reshape(1, -1)
        else:
            # If the sub-MPS has more than one site, the unitary is the MPS tensor at the specified site
            unitary = mps_data

        # Convert the unitary to a qtn.Tensor
        # .T at the end is useful for the application of unitaries as quantum circuit
        unitary = qtn.Tensor(unitary.reshape((phy_dim, phy_dim)).T, inds=("v", "p"), tags={"G"})

        # Append the unitary to the list of generated unitaries
        generated_unitaries.append(unitary)

        # Append the blank isometries and kernels to the lists (this is to ensure same length as the generated unitaries)
        isometries.append(np.array([]))
        kernels.append(np.array([]))

        return [generated_unitaries, isometries, kernels]

    # TODO: Redo the comments for better clarity.
    def generate_bond_d_unitary(self) -> list:
        """ Generate the unitary for the bond-d (physical dimension) compression of the MPS.

        Returns
        -------
        `generated_unitary_list` : list
            A list of unitaries to be applied to the MPS.

        Usage
        -----
        >>> mps.generate_bond_d_unitary()
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

    def _apply_unitary_layer(self,
                             generated_unitary_list: list) -> None:
        """ Apply the unitary layer on the MPS.

        Parameters
        ----------
        `generated_unitary_list` : list
            A list of unitaries to be applied to the MPS.
        """
        # Iterate over the generated unitary list and the start and end indices
        for start_index, end_index, generated_unitaries, _, _ in generated_unitary_list:
            # Iterate over the indices of the MPS
            for index in range(start_index, end_index + 1):
                # If the index is the end index
                if index == end_index:
                    # Apply the generated unitary gates to the MPS (use `.gate_` as the operation is inplace)
                    # o-o-o-o-o-o-o
                    # | | | | | | |
                    #     GGG
                    #     | |
                    self.mps.gate_(generated_unitaries[index - start_index].data, where=[index])

                    # Define the location to contract the tensors after applying the unitary gates
                    loc = np.where([isinstance(self.mps[site], tuple) for site in range(self.num_sites)])[0][0]

                    # Contract the tensors at the specified location
                    # o-o-o-GGG-o-o-o
                    # | | | / \ | | |
                    self.contract(self.mps[loc][-1].inds[-1])

                else:
                    # Apply a two-site gate and then split resulting tensor to retrieve the MPS form:
                    #     -o-o-A-B-o-o-
                    #      | | | | | |            -o-o-GGG-o-o-           -o-o-X~Y-o-o-
                    #      | | GGG | |     ==>     | | | | | |     ==>     | | | | | |
                    #      | | | | | |                 i j                     i j
                    #          i j
                    # As might be found in Time-evolving block decimation (TEBD) algorithm
                    self.mps.gate_split_(generated_unitaries[index - start_index].data,
                                         where=[index, index + 1])

        # Permute the arrays of the MPS
        self.permute(shape='lpr')

        # Compress the MPS
        self.compress(mode='right')

    def _apply_inverse_unitary_layer(self,
                                     generated_unitary_list: list):
        """ Apply the inverse unitary layer on the MPS.

        Parameters
        ----------
        `generated_unitary_list` : list
            A list of unitaries to be applied to the MPS.
        """
        # Iterate over the generated unitary list and the start and end indices
        for start_index, end_index, generate_unitaries, _, _ in generated_unitary_list:
            # Iterate over the indices of the MPS in reverse order
            for index in list(reversed(range(start_index, end_index + 1))):
                # If the index is the end index
                if index == end_index:
                    # Add the generated unitary gates to the MPS (use `.gate_` as the operation is inplace)
                    # o-o-o-o-o-o-o
                    # | | | | | | |
                    #     GGG
                    #     | |
                    self.mps.gate_(generate_unitaries[index - start_index].data.conj().T, where=[index])

                    # Define the location to contract the tensors after applying the unitary gates
                    loc = np.where([isinstance(self.mps[jt], tuple) for jt in range(self.num_sites)])[0][0]

                    # Contract the tensors at the specified location
                    # o-o-o-GGG-o-o-o
                    # | | | / \ | | |
                    self.contract(self.mps[loc][-1].inds[-1])

                else:
                    # Apply a two-site gate and then split resulting tensor to retrieve the MPS form:
                    #     -o-o-A-B-o-o-
                    #      | | | | | |            -o-o-GGG-o-o-           -o-o-X~Y-o-o-
                    #      | | GGG | |     ==>     | | | | | |     ==>     | | | | | |
                    #      | | | | | |                 i j                     i j
                    #          i j
                    # As might be found in Time-evolving block decimation (TEBD) algorithm
                    self.mps.gate_split_(generate_unitaries[index - start_index].data.conj().T,
                                         where=[index, index + 1])

        # Permute the arrays of the MPS
        self.permute(shape='lpr')

        # Compress the MPS
        self.compress(mode='right')

    def apply_unitary_layer(self,
                            unitary_layer: list,
                            inverse: bool = False) -> None:
        """ Apply the unitary layer on the MPS.

        Parameters
        ----------
        `unitary_layer` : list
            A list of unitaries to be applied to the MPS.
        `inverse` : bool, optional
            Whether to apply the inverse unitary layer.

        Usage
        -----
        >>> mps.apply_unitary_layer(unitary_layer, inverse=True)
        """
        if inverse:
            self._apply_inverse_unitary_layer(unitary_layer)
        else:
            self._apply_unitary_layer(unitary_layer)

    def apply_unitary_layers(self,
                             unitary_layers: list[list],
                             inverse: bool = False) -> None:
        """ Apply the unitary layers on the MPS.

        Parameters
        ----------
        `unitary_layers` : list[list]
            A list of unitary layers to be applied to the MPS.
        `inverse` : bool, optional
            Whether to apply the inverse unitary layers.

        Usage
        -----
        >>> mps.apply_unitary_layers(unitary_layers, inverse=True)
        """
        # Iterate over the unitary layers in reverse order, and apply the unitary layers to the MPS
        for layer in reversed(unitary_layers):
            self.apply_unitary_layer(layer, inverse=inverse)

    @staticmethod
    def _circuit_from_unitary_layer(circuit: Circuit,
                                    unitary_layer: list) -> None:
        """ Apply a unitary layer to the quantum circuit.

        Parameters
        ----------
        `circuit` : qickit.circuit.Circuit
            The quantum circuit.
        `unitary_layer` : list
            The unitary layer to be applied to the circuit.
        """
        # Iterate over the generated unitary list
        for start_index, end_index, generated_unitaries, _, _ in unitary_layer:
            # Iterate over the start and end indices
            for index in range(start_index, end_index + 1):
                # Define the unitary matrix
                unitary = generated_unitaries[index - start_index].data

                # If this is the last index, then apply the unitary to the last qubit
                if index == end_index:
                    circuit.unitary(unitary, [index])

                # Otherwise, apply the unitary to the current and next qubits
                else:
                    circuit.unitary(unitary, [index + 1, index])

    # NOTE: I think this should be in `MPS` (here). What do you think?
    def circuit_from_unitary_layers(self,
                                    qc_framework: Type[Circuit]) -> Circuit:
        """ Generate a quantum circuit from the MPS unitary layers.

        Parameters
        ----------
        `qc_framework` : type[qickit.circuit.Circuit]
            The quantum circuit framework.

        Returns
        -------
        `circuit` : qickit.circuit.Circuit
            The quantum circuit.
        """
        # Define the quantum circuit
        circuit = qc_framework(self.num_sites, self.num_sites)

        # Generate the unitary layers
        unitary_layers = self.generate_unitaries()

        # Iterate over the unitary layers in reverse order and apply the unitary layer
        for layer in reversed(range(len(unitary_layers))):
            MPS._circuit_from_unitary_layer(circuit, unitary_layers[layer])

        return circuit

    # TODO: Specify the parameters for best readability
    def draw(self) -> None:
        """ Draw the MPS.

        Returns
        -------
        `fig`
            The figure of the MPS.

        Usage
        -----
        >>> mps.draw()
        """
        self.mps.draw()