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

__all__ = ["MPS"]

from collections.abc import Sequence
import copy
import numpy as np
from numpy.typing import NDArray
import quimb.tensor as qtn # type: ignore
from scipy import linalg # type: ignore
from typing import Literal, SupportsComplex, SupportsIndex
from qickit.circuit import Circuit # type: ignore
from qickit.primitives import Ket # type: ignore

NumberType = float | SupportsComplex


# TODO: Confirm all methods and attributes needed for MPS support
# TODO: Add string diagrams for visual documentation where possible (ensure consistency)
class MPS:
    """ `qmprs.mps.MPS` is the class for creating and manipulating matrix product states (MPS).
    This class wraps the `quimb.tensor.MatrixProductState` class to provide a more user-friendly
    interface for creating and manipulating MPS.

    Refer to the link below for more information on the `quimb.tensor.MatrixProductState` class.
    https://quimb.readthedocs.io/en/latest/autoapi/quimb/tensor/tensor_1d/index.html#quimb.tensor.tensor_1d.MatrixProductState

    Notes
    -----
    Matrix product states (MPS) are a class of 1D tensor networks that are widely used
    in quantum computing to approximate the state of a quantum system. The MPS representation
    allows for a polynomial or even exponential reduction in the number of parameters required
    to represent a quantum state, making it a powerful tool for quantum state synthesis and simulation.

    The MPS representation is defined by first performing successive SVDs on the statevector
    of the quantum system. We use SVD to find low-rank structure in the tensor network and reduce
    the dimension of the tensors. We then choose a canonical form for the MPS, which is either "left"
    or "right". The canonical form of the MPS states how the singular values from the SVD are
    absorbed (contracted) with the left or right tensors.

    The MPS can be further compressed by truncating the bond dimension of the MPS. The bond
    dimension of the MPS determines the dimension of the unitary layers, and affects the
    fidelity of the approximation.

    Given each site will be represented as a $\chi\times\chi$ unitary matrix, the overall MPS will have a
    scaling of $O(N\chi^2)$, where N is the number of sites and $\chi$ is the bond dimension. If we can
    keep the bond dimension constant, the MPS will have a linear scaling with the number of sites.

    MPS Diagram:
    ```
     D D D D...D D D
    O-O-O-O-...-O-O-O
    | | | |     | | |
    d d d d     d d d
    ```
    where O represents the tensor at each site, d is the physical dimension, and D is the bond dimension
    (also known as rank). For qubit systems, the physical dimension is 2.

    Parameters
    ----------
    `statevector` : NDArray[np.complex128], optional
        The statevector of the quantum system.
    `mps` : qtn.MatrixProductState, optional
        The matrix product state (MPS) of the quantum system.
    `bond_dimension` : int
        The maximum bond dimension of the MPS.

    Attributes
    ----------
    `statevector` : qickit.primitives.Ket
        The statevector of the quantum system.
    `mps` : qtn.MatrixProductState
        The matrix product state (MPS) of the quantum system.
    `bond_dimension` : int
        The maximum bond dimension of the MPS.
    `num_sites` : int
        The number of sites for the MPS.
    `physical_dimension` : int
        The physical dimension of the MPS.
    `canonical_form` : Literal["left", "right"]
        The canonical form of the MPS.
    `normalized` : bool
        Whether the MPS is normalized.

    Raises
    ------
    TypeError
        If `bond_dimension` is not an integer.
    ValueError
        If `bond_dimension` is less than 1.
        `bond_dimension` must be an integer greater than 0.
        Cannot initialize with both `statevector` and `mps`.
        Must provide either `statevector` or `mps`.
        Only supports MPS with physical dimension of 2.

    Usage
    -----
    >>> statevector = [1, 2, 3, 4]
    >>> bond_dimension = 2
    >>> mps = MPS(statevector=statevector,
    ...           bond_dimension=bond_dimension)
    """
    def __init__(self,
                 statevector: NDArray[np.complex128] | None = None,
                 mps: qtn.MatrixProductState | None = None,
                 bond_dimension: int=64) -> None:
        """ Initialize a `qmprs.mps.MPS` instance. Pass only `statevector` to define
        the MPS from the statevector. Pass only `mps` to define the MPS from the MPS.
        """
        if not isinstance(bond_dimension, SupportsIndex) or bond_dimension < 1:
            raise ValueError("`bond_dimension` must be an integer greater than 0.")

        if statevector is not None and mps is not None:
            raise ValueError("Cannot initialize with both `statevector` and `mps`.")

        # Initialize the MPS from the statevector
        elif statevector is not None:
            if not isinstance(statevector, Ket):
                statevector = Ket(statevector)
            self.statevector = statevector
            self.mps = self.from_statevector(statevector, bond_dimension)

        # Initialize the MPS from the MPS
        elif mps is not None:
            if not isinstance(mps, qtn.MatrixProductState):
                raise TypeError("`mps` must be a `qtn.MatrixProductState` instance.")
            self.mps = mps
            self.statevector = self.to_statevector(mps)

        else:
            raise ValueError("Must provide either `statevector` or `mps`.")

        # Check if the MPS is normalized to 2-norm
        self.normalized = self.mps.norm() == 1

        # Define the maximum bond dimension
        self.bond_dimension = bond_dimension

        # Define the number of sites for the MPS
        self.num_sites = self.statevector.num_qubits # type: ignore

        # The physical dimension must be equal to two, given we define the synthesis
        # for qubit-based paradigms
        if self.mps.phys_dim() != 2:
            raise ValueError("Only supports MPS with physical dimension of 2.")
        self.physical_dimension = 2

    @staticmethod
    def from_statevector(statevector: Ket,
                         max_bond_dimension: int) -> qtn.MatrixProductState:
        """ Define the MPS from the statevector.

        Parameters
        ----------
        `statevector` : qickit.primitives.Ket
            The statevector of the quantum system.

        Returns
        -------
        `mps` : qtn.MatrixProductState
            The MPS of the quantum system.

        Usage
        -----
        >>> statevector = Ket([1, 2, 3, 4])
        >>> max_bond_dimension = 2
        >>> mps = MPS.from_statevector(statevector, max_bond_dimension)
        """
        num_sites = statevector.num_qubits

        # Generate MPS from the statevector
        mps = qtn.MatrixProductState.from_dense(statevector.data)

        # Compress the bond dimension of the MPS to the maximum bond dimension specified
        # This is to ensure the resulting MPS bond dimension is equal to the maximum
        # bond dimension specified during initialization
        for i in range(num_sites-1):
            qtn.tensor_core.tensor_compress_bond(mps[i], mps[i+1], max_bond=max_bond_dimension)

        return mps

    @staticmethod
    def to_statevector(mps: qtn.MatrixProductState) -> Ket:
        """ Convert the MPS to a `qickit.primitives.Ket` statevector instance.

        Parameters
        ----------
        `mps` : qtn.MatrixProductState
            The matrix product state (MPS) of the quantum system.

        Returns
        -------
        qickit.primitives.Ket
            The statevector of the quantum system.

        Usage
        -----
        >>> mps = qtn.MatrixProductState.random([2, 2, 2, 2], 2)
        >>> statevector = MPS.to_statevector(mps)
        """
        return Ket(mps.to_dense())

    def normalize(self) -> None:
        """ Normalize the MPS.

        Usage
        -----
        >>> mps.normalize()
        """
        self.mps.normalize()

    def canonicalize(self,
                     mode: Literal["left", "right"],
                     normalize=False) -> None:
        """ Convert the MPS to the canonical form. This states how the singular values from
        the SVD are absorbed (contracted) with the left or right tensors.
        - If `mode` is "left", the singular values are absorbed into the tensors to their
        right. (all tensors contract to unit matrix from left)

        ```
                          i              i
            >->->->->->->-o-o-         +-o-o-
            | | | | | | | | | ...  ->  | | | ...
            >->->->->->->-o-o-         +-o-o-
        ```

        - If `mode` is "right", the singular values are absorbed into the tensors to their
        left. (all tensors contract to unit matrix from right)

        ```
                   i                           i
                -o-o-<-<-<-<-<-<-<          -o-o-+
             ... | | | | | | | | |   ->  ... | | |
                -o-o-<-<-<-<-<-<-<          -o-o-+
        ```

        Parameters
        ----------
        `mode` : Literal["left", "right"]
            The mode of canonicalization.
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
            self.canonical_form = "left"
        elif mode == "right":
            self.mps.right_canonize(normalize=normalize)
            self.canonical_form = "right"
        else:
            raise ValueError("`mode` must be either 'left' or 'right'.")

    def compress(self,
                 max_bond_dimension: int | None = None,
                 mode: Literal["left", "right", "flat"] | None = None) -> None:
        """ SVD Compress the bond dimension of the MPS.

        ```
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
        ```

        Parameters
        ----------
        `max_bond_dimension` : int
            The maximum bond dimension of the MPS.
        `mode` : Literal["left", "right", "flat"], optional
            The mode of compression.

        Raises
        ------
        ValueError
            If `mode` is not "left", "right", or "flat".

        Usage
        -----
        >>> mps.compress(max_bond_dimension=16)
        >>> mps.compress(mode="left")
        >>> mps.compress(max_bond_dimension=32, mode="right")
        """
        if not (max_bond_dimension or mode):
            self.mps.compress()

        elif not mode:
            for i in range(self.num_sites-1):
                qtn.tensor_core.tensor_compress_bond(self.mps[i], self.mps[i+1], max_bond=max_bond_dimension)

        else:
            if mode in ["left", "right", "flat"]:
                if not max_bond_dimension:
                    self.mps.compress(form=mode)
                else:
                    self.mps.compress(form=mode, max_bond=max_bond_dimension)
            else:
                raise ValueError(f"`mode` must be either 'left', 'right', or 'flat'. Received {mode}.")

    def contract_site(self,
                      sites: Sequence[int]) -> None:
        """ Contract/block tensors sites together.

        # TODO: Add Figure

        Parameters
        ----------
        `sites` : Sequence[int]
            The sites to contract.

        Raises
        ------
        TypeError
            If `sites` is not a sequence of integers.
            If any element of `sites` is not an integer.

        Usage
        -----
        >>> mps.contract([0, 1])
        """
        if not isinstance(sites, Sequence):
            raise TypeError("`sites` must be a collection of integers.")
        elif not all(isinstance(index, int) for index in sites):
            raise TypeError("All elements of `sites` must be integers.")

        self.mps ^= (self.mps.site_tag(i) for i in sites)

    def contract_index(self,
                       index: str) -> None:
        """ Contract tensors connected by the given index.

        Notes
        -----
        Tensor contraction is performed when two tensors share a common index or
        when a single tensor has two identical indices.

        For instance, a 4th order tensor $T_{ijkl}$ can be contracted into $T_{jl}$ given
        $i$ and $k$ being identical by summing over the values of index $i$. The resulting
        tensor will be of order 2.

        We perform contraction for a single tensor by the following steps:
        1) Choose two indices to contract. $T_{ijkl}$ -> choose $i$ and $k$.
        2) Set these indices as equal. $T_{ijkl}$ -> $T_{ijil}$.
        3) Sum over this index. Assume $i = {1, 2, 3}$, so $T_{ijil} = T_{1j1l} + T_{2j2l} + T_{3j3l}$.

        Using the Einstein summation convention, we can represent this as simply $T_{jl}$.

        Furthermore, we perform contraction for two tensors by the following steps:
        1) Choose a common index between two tensors. $A_{ij} \otimes v_{l}$ -> choose $l = j$.
        2) Set these indices as equal. $A_{ij} \otimes v_{l}$ -> $A_{ij} \otimes v_{j}$.
        3) Sum over this index. Assume $j = {1, 2, 3}$, so $A_{ij} \otimes v_{j} = A_{i1} \otimes v_{1} + A_{i2} \otimes v_{2} + A_{i3} \otimes v_{3}$.

        The result is a 1st order tensor $u_i$. This is the same as performing matrix-vector multiplication.

        Parameters
        ----------
        `index` : str
            The index to contract.

        Usage
        -----
        >>> mps.contract_ind("k0")
        """
        self.mps.contract_ind(index)

    # TODO: Should this return sth, or should we access the isometries and positive semidefinite matrix from the MPS?
    def polar_decompose(self,
                        indices: Sequence[int]) -> None:
        """ Perform a polar decomposition on the MPS to retrieve the
        isometries V and positive semidefinite matrix P.

        # TODO: Add diagram (9) from Wei et al.

        Parameters
        ----------
        `indices` : Sequence[int]
            The indices of each tensor to polar decompose.

        Returns
        -------
        `isometries` : qtn.Tensor
            The isometries.
        `positive_semidefinite_matrix` : qtn.Tensor
            The positive semidefinite matrix.

        Usage
        -----
        >>> mps.polar_decompose()
        """
        self.mps.split_tensor(
            tags=self.mps.site_tag(indices[0]),
            left_inds=[self.mps.site_ind(i) for i in indices],
            method="polar_right",
            ltags="V",
            rtags="P",
        )

    def permute(self,
                shape: Literal["lrp", "lpr"]) -> None:
        """ Permute the indices of each tensor in the MPS to match `shape`.

        Parameters
        ----------
        `shape` : Literal["lrp", "lpr"]
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

    def get_submps_indices(self) -> list[tuple[int, int]]:
        """ Return the indices of the tensors at each site of the MPS.

        Returns
        -------
        `submps_indices` : list[tuple[int, int]]
            The indices of the MPS tensors.

        Usage
        -----
        >>> mps.get_submps_indices()
        """
        submps_indices = []

        if self.num_sites == 1:
            return [(0, 0)]

        for site in range(self.num_sites):
            # Reset the dimension variables for each iteration
            dim_left, dim_right = 1, 1

            # Define the dimensions for each site
            # The first and last sites are connected to only one site
            # as opposed to the other sites in the middle which are connected
            # to two sites to their left and right
            #
            #  |
            #  ●━━ `dim_right`
            if site == 0:
                _, dim_right = self.mps[site].shape # type: ignore
            #
            #              |
            # `dim_left` ━━●
            elif site == (self.num_sites - 1):
                dim_left, _ = self.mps[site].shape # type: ignore
            #
            #              |
            # `dim_left` ━━●━━ `dim_right`
            else:
                dim_left, _, dim_right = self.mps[site].shape # type: ignore

            if dim_left < 2 and dim_right < 2:
                submps_indices.append((site, site))
            elif dim_left < 2 and dim_right >= 2:
                temp = site
            elif dim_left >= 2 and dim_right < 2:
                submps_indices.append((temp, site))

        return submps_indices

    def _generate_two_site_unitary(self,
                                   mps_data: NDArray[np.complex128],
                                   generated_unitaries: list[qtn.Tensor]) -> list[qtn.Tensor]:
        """ Generate a two site unitary for a given tensor in the MPS.

        Parameters
        ----------
        mps_data : NDArray[np.complex128]
            The data of the MPS tensor at the specified index.
        generated_unitaries : list[qtn.Tensor]
            The list of generated unitaries.

        Returns
        -------
        `generated_unitaries` : list[qtn.Tensor]
            The list of generated unitaries.
        """
        phy_dim = self.physical_dimension
        unitary = np.zeros((phy_dim, phy_dim, phy_dim, phy_dim), dtype=np.complex128)

        # Construct the orthonormal basis of the MPS using SVD
        orthonormal_basis = linalg.null_space(mps_data.reshape((phy_dim, -1)).conj())

        # Define the angle rotation for the orthonormal basis
        angle_rotation = np.exp(1j * np.angle(orthonormal_basis[0]))

        # Normalize the orthonormal basis
        orthonormal_basis = orthonormal_basis / angle_rotation

        # Set the first row of the unitary to the MPS tensor at the specified site
        # Given the physical dimension will always be equal to 2, we omit unitary[1:phy_dim]
        # to unitary[1] and explicitly set the value
        unitary[0] = mps_data
        unitary[1] = orthonormal_basis.reshape((phy_dim, phy_dim, phy_dim, phy_dim - 1)).transpose((3, 2, 0, 1))

        # Transpose the unitary, such that the indices of the unitary are ordered as unitary(B,L,R,T)
        unitary = unitary.transpose((1, 0, 2, 3))

        # Convert the unitary to a qtn.Tensor
        # .T at the end is useful for the application of unitaries as quantum circuit gates
        unitary = qtn.Tensor(
            unitary.reshape((phy_dim**2, phy_dim**2)).T, # type: ignore
            inds=["L", "R"],
            tags={"G"}
        )

        generated_unitaries.append(unitary)

        return generated_unitaries

    def _generate_first_site_unitary(self,
                                     mps_data: NDArray[np.complex128],
                                     generated_unitaries: list[qtn.Tensor]) -> list[qtn.Tensor]:
        """ Generate the first site unitary for a given tensor in the MPS.

        Parameters
        ----------
        mps_data : NDArray[np.complex128]
            The data of the MPS tensor at the specified index.
        generated_unitaries : List[qtn.Tensor]
            The list of generated unitaries.

        Returns
        -------
        `generated_unitaries` : list[qtn.Tensor]
            The list of generated unitaries.
        """
        phy_dim = self.physical_dimension
        unitary = np.zeros((phy_dim, phy_dim, phy_dim, phy_dim), dtype=np.complex128)

        # Construct the orthonormal basis of the MPS using SVD
        orthonormal_basis = linalg.null_space(mps_data.reshape((1, -1)).conj())

        # Given the physical dimension will always be equal to 2, we omit the loop and explicitly set the values
        # for faster computation
        unitary[0, 0] = mps_data.reshape((phy_dim, -1))
        unitary[0, 1] = orthonormal_basis[:, 0].reshape((phy_dim, phy_dim))
        unitary[1, 0] = orthonormal_basis[:, 1].reshape((phy_dim, phy_dim))
        unitary[1, 1] = orthonormal_basis[:, 2].reshape((phy_dim, phy_dim))

        # Transpose the unitary, such that the indices of the unitary are ordered as unitary(B,L,R,T)
        unitary = unitary.transpose((1, 0, 2, 3))

        # Convert the unitary to a qtn.Tensor
        # .T at the end is useful for the application of unitaries as quantum circuit gates
        unitary = qtn.Tensor(
            unitary.reshape((phy_dim**2, phy_dim**2)).T, # type: ignore
            inds=["L", "R"],
            tags={"G"}
        )

        generated_unitaries.append(unitary)

        return generated_unitaries

    def _generate_single_site_unitary(self,
                                      mps_data: NDArray[np.complex128],
                                      start_index: int,
                                      end_index: int,
                                      generated_unitaries: list[qtn.Tensor]) -> list[qtn.Tensor]:
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

        Returns
        -------
        `generated_unitaries` : list[qtn.Tensor]
            The list of generated unitaries.
        """
        phy_dim = self.physical_dimension

        # Check if the sub-MPS has only one site, and define the unitary accordingly
        if end_index == start_index:
            unitary = np.zeros((phy_dim, phy_dim), dtype=np.complex128)
            unitary[0] = mps_data.reshape((1, -1))
            unitary[1] = linalg.null_space(mps_data.reshape(1, -1).conj()).reshape(1, -1)
        else:
            unitary = mps_data

        # Convert the unitary to a qtn.Tensor
        # .T at the end is useful for the application of unitaries as quantum circuit
        unitary = qtn.Tensor(
            unitary.reshape((phy_dim, phy_dim)).T, # type: ignore
            inds=("v", "p"),
            tags={"G"}
        )

        generated_unitaries.append(unitary)

        return generated_unitaries

    def generate_unitaries(self) -> list:
        """ Generate the unitaries that prepare the MPS.

        Returns
        -------
        `generated_unitary_list` : list
            A list of unitaries to be applied to the MPS.

        Raises
        ------
        ValueError
            If all the generated unitaries are not unitary.

        Usage
        -----
        >>> mps.generate_unitaries()
        """
        # Copy the MPS (as the MPS will be modified in place)
        mps_copy = self.mps.copy(deep=True)

        generated_unitary_list = []

        sub_mps_indices = self.get_submps_indices()

        for start_index, end_index in sub_mps_indices:
            generated_unitaries: list[qtn.Tensor] = []

            for index in range(start_index, end_index + 1):
                if index == end_index:
                    generated_unitaries = self._generate_single_site_unitary(mps_copy[index].data, # type: ignore
                                                                             start_index,
                                                                             end_index,
                                                                             generated_unitaries)

                elif index != start_index:
                    generated_unitaries = self._generate_two_site_unitary(mps_copy[index].data, # type: ignore
                                                                          generated_unitaries)

                else:
                    generated_unitaries = self._generate_first_site_unitary(mps_copy[index].data, # type: ignore
                                                                            generated_unitaries)

            # Check if all the generated unitaries are actually unitary
            for generated_unitary in generated_unitaries:
                if not np.allclose(np.eye(generated_unitary.shape[0]) - generated_unitary.data @ generated_unitary.data.T.conj(), 0): # type: ignore
                    raise ValueError("ValueError : every generated unitary in the list must be a unitary.")
                if not np.allclose(np.eye(generated_unitary.shape[0]) - generated_unitary.data.T.conj() @ generated_unitary.data, 0): # type: ignore
                    raise ValueError("ValueError : every generated unitary in the list must be a unitary.")

            generated_unitary_list.append([start_index,
                                           end_index,
                                           generated_unitaries])

        return generated_unitary_list

    def generate_bond_d_unitary(self) -> list:
        """ Generate the unitaries that prepare the bond-d (physical dimension) compression of the MPS.

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

        mps_copy.compress(mode="right", max_bond_dimension=self.physical_dimension)
        mps_copy.canonicalize(mode="right", normalize=True)

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
        for start_index, end_index, generated_unitaries in generated_unitary_list:
            for index in range(start_index, end_index + 1):
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
                    self.contract_index(self.mps[loc][-1].inds[-1]) # type: ignore

                else:
                    # Apply a two-site gate using TEBD and then split resulting tensor to retrieve the MPS form:
                    #     -o-o-A-B-o-o-
                    #      | | | | | |            -o-o-GGG-o-o-           -o-o-X~Y-o-o-
                    #      | | GGG | |     ==>     | | | | | |     ==>     | | | | | |
                    #      | | | | | |                 i j                     i j
                    #          i j
                    self.mps.gate_split_(generated_unitaries[index - start_index].data,
                                         where=[index, index + 1])

        self.permute(shape="lpr")
        self.compress(mode="right")

    def _apply_inverse_unitary_layer(self,
                                     generated_unitary_list: list) -> None:
        """ Apply the inverse unitary layer on the MPS.

        Parameters
        ----------
        `generated_unitary_list` : list
            A list of unitaries to be applied to the MPS.
        """
        for start_index, end_index, generate_unitaries in generated_unitary_list:
            for index in list(reversed(range(start_index, end_index + 1))):
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
                    self.contract_index(self.mps[loc][-1].inds[-1]) # type: ignore

                else:
                    # Apply a two-site gate using TEBD and then split resulting tensor to retrieve the MPS form:
                    #     -o-o-A-B-o-o-
                    #      | | | | | |            -o-o-GGG-o-o-           -o-o-X~Y-o-o-
                    #      | | GGG | |     ==>     | | | | | |     ==>     | | | | | |
                    #      | | | | | |                 i j                     i j
                    #          i j
                    self.mps.gate_split_(generate_unitaries[index - start_index].data.conj().T,
                                         where=[index, index + 1])

        self.permute(shape='lpr')

    def apply_unitary_layer(self,
                            unitary_layer: list,
                            inverse: bool=False) -> None:
        """ Apply the unitary layer on the MPS. If inverse is True,
        we apply the inverse of the unitary layer to the MPS.

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
                             inverse: bool=False) -> None:
        """ Apply the unitary layers on the MPS. If inverse is True,
        we apply the inverse of the unitary layers in reverse order
        to the MPS.

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
        for layer in reversed(unitary_layers):
            self.apply_unitary_layer(layer, inverse=inverse)

    @staticmethod
    def _apply_unitary_layer_to_circuit(circuit: Circuit,
                                        unitary_layer: list) -> None:
        """ Apply a unitary layer to the quantum circuit.

        Parameters
        ----------
        `circuit` : qickit.circuit.Circuit
            The quantum circuit.
        `unitary_layer` : list
            The unitary layer to be applied to the circuit.
        """
        for start_index, end_index, generated_unitaries in unitary_layer:
            for index in range(start_index, end_index + 1):
                unitary = generated_unitaries[index - start_index].data

                if index == end_index:
                    circuit.unitary(unitary, [index])
                else:
                    circuit.unitary(unitary, [index + 1, index])

    def circuit_from_unitary_layers(self,
                                    circuit_framework: type[Circuit],
                                    unitary_layers: list[list]) -> Circuit:
        """ Generate a quantum circuit from the MPS unitary layers.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The quantum circuit framework.
        `unitary_layers` : list[list]
            A list of unitary layers to be applied to the circuit.

        Returns
        -------
        `circuit` : qickit.circuit.Circuit
            The quantum circuit with the unitary layers applied.
        """
        circuit = circuit_framework(self.num_sites)

        for layer in reversed(unitary_layers):
            MPS._apply_unitary_layer_to_circuit(circuit, layer)

        return circuit

    def draw(self) -> None:
        """ Draw the MPS.

        Usage
        -----
        >>> mps.draw()
        """
        self.mps.draw()