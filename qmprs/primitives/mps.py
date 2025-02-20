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

""" MPS module.

This class wraps the quimb.tensor.MatrixProductState class to provide a more user-friendly
interface for creating and manipulating MPS.

Refer to the link below for more information on the quimb.tensor.MatrixProductState class.
https://quimb.readthedocs.io/en/latest/autoapi/quimb/tensor/tensor_1d/index.html#quimb.tensor.tensor_1d.MatrixProductState
"""

from __future__ import annotations

__all__ = ["MPS"]

from collections.abc import Sequence
import copy
from autoray import do # type: ignore
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import quimb.tensor as qtn # type: ignore
from quimb.tensor.tensor_1d_compress import tensor_network_1d_compress # type: ignore
from scipy import linalg # type: ignore
from typing import Literal, TypeAlias
from quick.circuit.metrics.metrics import _get_submps_indices as get_submps_indices
from quick.predicates import is_unitary_matrix
from quick.primitives import Ket

""" Type aliases for unitary blocks and unitary layers.
`UnitaryBlock`: A unitary block is a single unitary operator that acts on the MPS.
`UnitaryLayer`: A unitary layer is a list of unitary blocks to be applied to the MPS.
"""
UnitaryBlock: TypeAlias = tuple[int, int, list[qtn.Tensor]]
UnitaryLayer: TypeAlias = list[UnitaryBlock]


class MPS:
    """ `qmprs.primitives.MPS` is the class for creating and manipulating matrix product
    states (MPS). This class wraps the `quimb.tensor.MatrixProductState` class to provide
    a more user-friendly interface for creating and manipulating MPS.

    Refer to the link below for more information on the `quimb.tensor.MatrixProductState` class.
    https://quimb.readthedocs.io/en/latest/autoapi/quimb/tensor/tensor_1d/index.html#quimb.tensor.tensor_1d.MatrixProductState

    Notes
    -----
    Matrix product states (MPS) are a class of 1D tensor networks that are widely used
    in quantum computing to approximate the state of a quantum system. The MPS representation
    allows for a polynomial or even exponential reduction in the number of parameters required
    to represent a quantum state, making it a powerful tool for quantum state synthesis and
    simulation.

    The MPS representation is defined by first performing successive SVDs on the statevector
    of the quantum system. We use SVD to find low-rank structure in the tensor network and
    reduce the dimension of the tensors. We then choose a canonical form for the MPS, which
    is either "left" or "right". The canonical form of the MPS states how the singular values
    from the SVD are absorbed (contracted) with the left or right tensors.

    The MPS can be further compressed by truncating the bond dimension of the MPS. The bond
    dimension of the MPS determines the dimension of the unitary layers, and affects the
    fidelity of the approximation. This is because the bond dimension captures the entanglement
    structure of the quantum many-body system, where a higher bond dimension implies a higher
    degree of entanglement.

    Given each site will be represented as a $\chi\times\chi$ unitary matrix, the overall MPS
    will have a scaling of $O(N\chi^2)$, where N is the number of sites and $\chi$ is the bond
    dimension. Given a bond dimension of $2^{N/2}$ we can exactly represent any quantum state
    of N qubits. However, for practical purposes, if we can keep the bond dimension constant,
    the MPS will have a linear scaling with the number of sites.

    The MPS can be written in the following

    |ψ⟩ = ∑_{i1,i2,...,iN} Tr(A^{i1}A^{i2}...A^{iN}) |i1,i2,...,iN⟩

    Where for arbitrary states, the MPS would be open-boundary condition, and non-translational
    invariant, where A^{i} are not necessarily equal, and the first and last tensors are vectors.

    MPS Diagram:
    ```
     D D D D...D D D
    O-O-O-O-...-O-O-O
    | | | |     | | |
    d d d d     d d d
    ```

    where O represents the tensor at each site, d is the physical dimension, and D is the bond
    dimension (also known as rank). For qubit systems, the physical dimension is 2.

    An important note is that the MPS representation is aimed for at least 2 qubits, as the MPS
    approximates the entanglement structure of the quantum many-body systems.

    Parameters
    ----------
    `statevector` : quick.primitives.Ket | NDArray[np.complex128], optional, default=None
        The statevector of the quantum system.
    `mps` : qtn.MatrixProductState, optional, default=None
        The matrix product state (MPS) of the quantum system.
    `bond_dimension` : int
        The maximum bond dimension of the MPS.

    Attributes
    ----------
    `statevector` : quick.primitives.Ket
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
        - If `bond_dimension` is not an integer.
    ValueError
        - If `bond_dimension` is less than 1.
        - `bond_dimension` must be an integer greater than 0.
        - Cannot initialize with both `statevector` and `mps`.
        - Must provide either `statevector` or `mps`.
        - Only supports MPS with physical dimension of 2.
        - The statevector must have at least 2 qubits.
        - The MPS must have at least 2 tensors.

    Usage
    -----
    >>> statevector = [1, 2, 3, 4]
    >>> bond_dimension = 2
    >>> mps = MPS(statevector=statevector,
    ...           bond_dimension=bond_dimension)
    """
    def __init__(
            self,
            statevector: Ket | NDArray[np.complex128] | None = None,
            mps: qtn.MatrixProductState | None = None,
            bond_dimension: int=64
        ) -> None:
        """ Initialize a `qmprs.primitives.MPS` instance.

        Pass only `statevector` to define the MPS from the statevector.
        Pass only `mps` to define the MPS from the MPS.
        """
        if not isinstance(bond_dimension, int) or bond_dimension < 1:
            raise ValueError(
                "`bond_dimension` must be an integer greater than 0. "
                f"Received {bond_dimension}."
            )

        # Initialize the MPS from the statevector
        if (statevector is not None) and (mps is None):
            if not isinstance(statevector, Ket):
                statevector = Ket(statevector)

            if statevector.num_qubits == 1: # type: ignore
                raise ValueError(
                    "The statevector must have at least 2 qubits. "
                    f"Received {statevector.num_qubits}."
                )

            self.statevector = statevector
            self.mps = self.from_statevector(statevector, bond_dimension)

        # Initialize the MPS from the MPS
        elif (mps is not None) and (statevector is None):
            if not isinstance(mps, qtn.MatrixProductState):
                raise TypeError(
                    "`mps` must be a `qtn.MatrixProductState` instance. "
                    f"Received {type(mps)}."
                )

            if mps.num_tensors == 1:
                raise ValueError(
                    "The MPS must have at least 2 tensors. "
                    f"Received {mps.num_tensors}."
                )

            self.mps = mps
            self.statevector = self.to_statevector(mps)

        else:
            raise ValueError("Must provide either `statevector` or `mps` not both.")

        # Define the maximum bond dimension
        self.bond_dimension = bond_dimension

        # Define the number of sites for the MPS
        self.num_sites = self.statevector.num_qubits # type: ignore

        # The physical dimension must be equal to two, given we define the synthesis
        # for qubit-based paradigms
        if self.mps.phys_dim() != 2:
            raise ValueError(
                "Only supports MPS with physical dimension of 2. "
                f"Received {self.mps.phys_dim()}."
            )

        self.physical_dimension = 2

    @staticmethod
    def from_statevector(
            statevector: Ket,
            max_bond_dimension: int
        ) -> qtn.MatrixProductState:
        """ Define the MPS from the statevector.

        Parameters
        ----------
        `statevector` : quick.primitives.Ket
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
        # Generate MPS from the statevector
        mps = qtn.MatrixProductState.from_dense(statevector.data)

        # Compress the bond dimension of the MPS to the maximum bond dimension specified
        # This is to ensure the resulting MPS bond dimension is equal to the maximum
        # bond dimension specified during initialization
        mps = tensor_network_1d_compress(tn=mps, max_bond=max_bond_dimension)

        return mps

    @staticmethod
    def to_statevector(mps: qtn.MatrixProductState) -> Ket:
        """ Convert the MPS to a `quick.primitives.Ket` statevector instance.

        Parameters
        ----------
        `mps` : qtn.MatrixProductState
            The matrix product state (MPS) of the quantum system.

        Returns
        -------
        quick.primitives.Ket
            The statevector of the quantum system.

        Usage
        -----
        >>> mps = qtn.MatrixProductState.random([2, 2, 2, 2], 2)
        >>> statevector = MPS.to_statevector(mps)
        """
        return Ket(mps.to_dense())

    @property
    def norm(self) -> float:
        """ The 2-norm of the MPS.

        Returns
        -------
        float
            The 2-norm of the MPS.

        Usage
        -----
        >>> mps.norm
        """
        return self.mps.norm() # type: ignore

    @property
    def is_normalized(self) -> bool:
        """ Check if the MPS is normalized.

        Returns
        -------
        bool
            Whether the MPS is normalized.

        Usage
        -----
        >>> mps.is_normalized
        """
        return True if np.isclose(self.norm, 1) else False

    def normalize(self) -> None:
        """ Normalize the MPS.

        Usage
        -----
        >>> mps.normalize()
        """
        if not self.is_normalized:
            self.mps.normalize()

    @property
    def orthogonal_center_range(self) -> tuple[int, int]:
        """ The range of the orthogonality center.

        Returns
        -------
        tuple[int, int]
            The range of the orthogonality center.

        Usage
        -----
        >>> mps.orthogal_center_range
        """
        return self.mps.calc_current_orthog_center() # type: ignore

    @property
    def canonical_form(self) -> Literal["left", "right", "unknown"]:
        """ The canonical form of the MPS.

        Returns
        -------
        Literal["left", "right", "unknown"]
            The canonical form of the MPS. Can be "left", "right",
            or "unknown".

        Usage
        -----
        >>> mps.canonical_form
        """
        if self.orthogonal_center_range == (0, 0):
            return "right"
        elif self.orthogonal_center_range == (self.num_sites - 1, self.num_sites - 1):
            return "left"
        else:
            return "unknown"

    def canonicalize(
            self,
            mode: Literal["left", "right"],
            normalize=False
        ) -> None:
        """ Convert the MPS to the canonical form. This states how the singular values from
        the SVD are absorbed (contracted) with the left or right tensors. Enables the loss-less
        conversion of all tensors in a TN into isometries
        (i.e. inner product preserving transformations between Hilbert space).

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
            - If `mode` is not "left" or "right".

        Usage
        -----
        >>> mps.canonicalize("left")
        >>> mps.canonicalize("right", normalize=True)
        """
        if mode == "left":
            self.mps.left_canonize(normalize=normalize)
        elif mode == "right":
            self.mps.right_canonize(normalize=normalize)
        else:
            raise ValueError("`mode` must be either 'left' or 'right'.")

    def compress(
            self,
            max_bond_dimension: int | None = None,
            mode: Literal["left", "right"] | None = None
        ) -> None:
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
        `mode` : Literal["left", "right"], optional
            The mode of compression.

        Raises
        ------
        ValueError
            - If `mode` is not "left", or "right".

        Usage
        -----
        >>> mps.compress(max_bond_dimension=16)
        >>> mps.compress(mode="left")
        >>> mps.compress(max_bond_dimension=32, mode="right")
        """
        if not (max_bond_dimension or mode):
            self.mps.compress()

        elif not mode:
            self.mps = tensor_network_1d_compress(tn=self.mps, max_bond=max_bond_dimension)
            self.bond_dimension = max_bond_dimension

        else:
            if mode in ["left", "right"]:
                if not max_bond_dimension:
                    self.mps.compress(form=mode)
                else:
                    self.mps.compress(form=mode, max_bond=max_bond_dimension)
                    self.bond_dimension = max_bond_dimension
            else:
                raise ValueError(
                    "`mode` must be either 'left', or 'right'. "
                    f"Received {mode}."
                )

    def contract_site(
            self,
            sites: Sequence[int]
        ) -> None:
        """ Contract/block tensors sites together.

        # TODO: Add Figure

        Parameters
        ----------
        `sites` : Sequence[int]
            The sites to contract.

        Raises
        ------
        TypeError
            - If `sites` is not a sequence of integers.

        Usage
        -----
        >>> mps.contract([0, 1])
        """
        if not isinstance(sites, Sequence) or not all(isinstance(index, int) for index in sites):
            raise TypeError("`sites` must be a collection of integers.")

        self.mps ^= (self.mps.site_tag(i) for i in sites)

    def contract_index(
            self,
            index: str
        ) -> None:
        """ Contract tensors connected by the given index.

        # TODO: Add Figure

        Parameters
        ----------
        `index` : str
            The index to contract.

        Usage
        -----
        >>> mps.contract_ind("k0")
        """
        self.mps.contract_ind(index)

    def polar_decompose(
            self,
            indices: Sequence[int]
        ) -> None:
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

    def permute(
            self,
            shape: Literal["lrp", "lpr"]
        ) -> None:
        """ Permute the indices of each tensor in the MPS to match `shape`.
        This is used to define the orthogonality of the MPS.

        Parameters
        ----------
        `shape` : Literal["lrp", "lpr"]
            The shape to permute, being "lrp" or "lpr".

        Raises
        ------
        ValueError
            - If `shape` is not "lrp" or "lpr".

        Usage
        -----
        >>> mps.permute("lrp")
        >>> mps.permute("lpr")
        """
        if shape not in ["lrp", "lpr"]:
            raise ValueError(f"`shape` must be either 'lrp' or 'lpr'. Received {shape}.")

        self.mps.permute_arrays(shape)

    def _generate_first_site_unitary(
            self,
            mps_data: NDArray[np.complex128],
            generated_unitaries: list[qtn.Tensor]
        ) -> list[qtn.Tensor]:
        """ Generate the first site unitary of the MPS.

        Notes
        -----
        This method implements equation (9) from [1].

        For more information, refer to the publication below:
        [1] Ran, Shi-Ju.
        Encoding of Matrix Product States into Quantum Circuits of One- and Two-Qubit Gates (2020).
        https://arxiv.org/abs/1908.07958

        Parameters
        ----------
        `mps_data` : NDArray[np.complex128]
            The data of the MPS tensor at the specified index.
        `generated_unitaries` : list[qtn.Tensor]
            The list of generated unitaries to add the first site unitary to.

        Returns
        -------
        `generated_unitaries` : list[qtn.Tensor]
            The list of generated unitaries after adding the first site unitary.
        """
        phy_dim = self.physical_dimension
        unitary = np.zeros((phy_dim, phy_dim, phy_dim, phy_dim), dtype=np.complex128)

        # Construct the orthonormal basis (kernel) of the MPS using SVD
        orthonormal_basis = linalg.null_space(mps_data.reshape((1, -1)).conj())

        # Given the physical dimension will always be equal to 2, we omit the loop and explicitly set the values
        # for faster computation
        # Unitary with i,j = 0 being set as the mps data of the first site
        unitary[0, 0] = mps_data.reshape((phy_dim, -1))

        # The other (d^2-1) elements are derived from the orthonormal basis
        # Given d=2, we can directly set the three values
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

    def _generate_two_site_unitary(
            self,
            mps_data: NDArray[np.complex128],
            generated_unitaries: list[qtn.Tensor]
        ) -> list[qtn.Tensor]:
        """ Generate a two site unitary.

        Notes
        -----
        This method implements equation (7) from [1].

        For more information, refer to the publication below:
        [1] Ran, Shi-Ju.
        Encoding of Matrix Product States into Quantum Circuits of One- and Two-Qubit Gates (2020).
        https://arxiv.org/abs/1908.07958

        Parameters
        ----------
        `mps_data` : NDArray[np.complex128]
            The data of the MPS tensor at the specified index.
        `generated_unitaries` : list[qtn.Tensor]
            The list of generated unitaries to add the two site unitary to.

        Returns
        -------
        `generated_unitaries` : list[qtn.Tensor]
            The list of generated unitaries after adding the two site unitary.
        """
        phy_dim = self.physical_dimension
        unitary = np.zeros((phy_dim, phy_dim, phy_dim, phy_dim), dtype=np.complex128)

        # Construct the orthonormal basis (kernel) of the MPS using SVD
        orthonormal_basis = linalg.null_space(mps_data.reshape((phy_dim, -1)).conj())

        # Perform phase correction to stabilize the computation
        # This improves the fidelity of the MPS to circuit encoding
        angle_rotation = np.exp(1j * np.angle(orthonormal_basis[0]))
        orthonormal_basis = orthonormal_basis / angle_rotation

        # When i=0, the unitary is equal to the mps data, and when i=1 it's equal to the orthogonal basis
        # of the mps data (kernel)
        # Given the physical dimension will always be equal to 2, we omit unitary[1:phy_dim]
        # to unitary[1] and explicitly set the value
        unitary[0] = mps_data
        unitary[1] = orthonormal_basis.reshape(
            (phy_dim, phy_dim, phy_dim, phy_dim - 1)
        ).transpose((3, 2, 0, 1))

        # Transpose the unitary, such that the indices of the unitary are ordered as
        # unitary(B,L,R,T)
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

    def _generate_last_site_unitary(
            self,
            mps_data: NDArray[np.complex128],
            start_index: int,
            end_index: int,
            generated_unitaries: list[qtn.Tensor]
        ) -> list[qtn.Tensor]:
        """ Generate the last site unitary of the MPS.

        Notes
        -----
        This method implements equation (6) from [1].

        For more information, refer to the publication below:
        [1] Ran, Shi-Ju.
        Encoding of Matrix Product States into Quantum Circuits of One- and Two-Qubit Gates (2020).
        https://arxiv.org/abs/1908.07958

        Parameters
        ----------
        `mps_data` : NDArray[np.complex128]
            The data of the MPS tensor at the specified index.
        `start_index` : int
            The starting index of the sub-MPS.
        `end_index` : int
            The ending index of the sub-MPS.
        `generated_unitaries` : list[qtn.Tensor]
            The list of generated unitaries.

        Returns
        -------
        `generated_unitaries` : list[qtn.Tensor]
            The list of generated unitaries.
        """
        phy_dim = self.physical_dimension

        # If the sub MPS has only one site (i.e., start index is equal to end index)
        if end_index == start_index:
            unitary = np.zeros((phy_dim, phy_dim), dtype=np.complex128)
            unitary[0] = mps_data.reshape((1, -1))
            # Set the second row of the unitary to the orthonormal basis (kernel) of the MPS tensor
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

    def generate_unitary_layer(self) -> UnitaryLayer:
        """ Generate a unitary layer that evolves the product state
        |00..0> to the MPS. Each unitary layer is a list of unitary
        matrices (blocks) to be applied to the MPS.

        Notes
        -----
        The unitary layer is based on the analytical decomposition
        of equations (6) to (9) from [1].

        For more information, refer to the publication below:
        [1] Ran, Shi-Ju.
        Encoding of Matrix Product States into Quantum Circuits of One- and Two-Qubit Gates (2020).
        https://arxiv.org/abs/1908.07958

        Returns
        -------
        `generated_unitary_layer` : UnitaryLayer
            The unitary layer to be applied to the MPS.

        Raises
        ------
        ValueError
            - If all the generated unitaries are not unitary.

        Usage
        -----
        >>> mps.generate_unitary_layer()
        """
        # Copy the MPS (as the MPS will be modified in place)
        mps_copy = self.mps.copy(deep=True)

        generated_unitary_layer: list[UnitaryBlock] = []

        sub_mps_indices = get_submps_indices(self.mps)

        for start_index, end_index in sub_mps_indices:
            generated_unitaries: list[qtn.Tensor] = []

            for index in range(start_index, end_index + 1):
                if index == end_index:
                    generated_unitaries = self._generate_last_site_unitary(
                        mps_copy[index].data, # type: ignore
                        start_index,
                        end_index,
                        generated_unitaries
                    )

                elif index != start_index:
                    generated_unitaries = self._generate_two_site_unitary(
                        mps_copy[index].data, # type: ignore
                        generated_unitaries
                    )

                else:
                    generated_unitaries = self._generate_first_site_unitary(
                        mps_copy[index].data, # type: ignore
                        generated_unitaries
                    )

            # Check if all the generated unitaries are actually unitary
            for generated_unitary in generated_unitaries:
                if not is_unitary_matrix(generated_unitary.data):
                    raise ValueError("All the generated unitaries must be unitary.")

            # A unitary layer is a list of unitaries to be applied to the MPS
            # at the specified indices from start index to end index
            generated_unitary_layer.append(
                (start_index, end_index, generated_unitaries)
            )

        return generated_unitary_layer

    def generate_bond_D_unitary_layer(self) -> UnitaryLayer:
        """ Truncate the unitary layer's bond dimension to 2.

        Notes
        -----
        Given a bond dimension of a power of 2, we would require $log2(\chi)$ qubits
        to represent the unitaries. For sequential encoding by Shi-ju Ran, we will use
        bond dimension of physical dimension, which will require one qubit per index.

        This is needed to ensure we only use one and two qubit gates.

        Publication:
        https://arxiv.org/pdf/2209.00595, Figure 1

        Returns
        -------
        `generated_unitary_layer` : UnitaryLayer
            The unitary layer to be applied to the MPS.

        Raises
        ------
        ValueError
            - If all the generated unitaries are not unitary.

        Usage
        -----
        >>> mps.generate_bond_D_unitary_layer()
        """
        # Copy the MPS (as the MPS will be modified in place with `.compress` and `.canonicalize` methods)
        mps_copy = copy.deepcopy(self)

        mps_copy.compress(mode="right", max_bond_dimension=self.physical_dimension)

        # To facilitate the loss-less conversion of all core
        # tensors in a TN into isometries (i.e. inner product
        # preserving transformations between Hilbert space)
        # we will canocalize the MPS
        mps_copy.canonicalize(mode="right", normalize=True)

        generated_unitary_layer = mps_copy.generate_unitary_layer()

        return generated_unitary_layer

    def _apply_unitary_layer(
            self,
            unitary_layer: UnitaryLayer
        ) -> None:
        """ Apply the unitary layer on the MPS.

        Parameters
        ----------
        `unitary_layer` : UnitaryLayer
            The unitary layer to be applied to the MPS.
        """
        for start_index, end_index, unitary_blocks in unitary_layer:
            for index in range(start_index, end_index + 1):
                if index == end_index:
                    # Apply the generated unitary gates to the MPS
                    # (use `.gate_` as the operation is inplace)
                    # o-o-o-o-o-o-o
                    # | | | | | | |
                    #     GGG
                    #     | |
                    self.mps.gate_(unitary_blocks[index - start_index].data, where=[index])

                    # Define the location to contract the tensors after
                    # applying the unitary gates
                    loc = np.where(
                        [isinstance(self.mps[site], tuple) for site in range(self.num_sites)]
                    )[0][0]

                    # Contract the tensors at the specified location
                    # o-o-o-GGG-o-o-o
                    # | | | / \ | | |
                    self.contract_index(self.mps[loc][-1].inds[-1]) # type: ignore

                else:
                    # Apply a two-site gate using TEBD and then split resulting
                    # tensor to retrieve the MPS form:
                    #     -o-o-A-B-o-o-
                    #      | | | | | |            -o-o-GGG-o-o-           -o-o-X~Y-o-o-
                    #      | | GGG | |     ==>     | | | | | |     ==>     | | | | | |
                    #      | | | | | |                 i j                     i j
                    #          i j
                    self.mps.gate_split_(
                        unitary_blocks[index - start_index].data,
                        where=[index, index + 1]
                    )

        self.permute(shape="lpr")
        self.compress(mode="right")

    def _apply_inverse_unitary_layer(
            self,
            unitary_layer: UnitaryLayer
        ) -> None:
        """ Apply the inverse unitary layer on the MPS.

        Parameters
        ----------
        `unitary_layer` : UnitaryLayer
            The unitary layer to be applied to the MPS.
        """
        for start_index, end_index, unitary_blocks in unitary_layer:
            for index in list(reversed(range(start_index, end_index + 1))):
                if index == end_index:
                    # Add the generated unitary gates to the MPS
                    # (use `.gate_` as the operation is inplace)
                    # o-o-o-o-o-o-o
                    # | | | | | | |
                    #     GGG
                    #     | |
                    self.mps.gate_(
                        unitary_blocks[index - start_index].data.conj().T,
                        where=[index]
                    )

                    # Define the location to contract the tensors after
                    # applying the unitary gates
                    loc = np.where(
                        [isinstance(self.mps[jt], tuple) for jt in range(self.num_sites)]
                    )[0][0]

                    # Contract the tensors at the specified location
                    # o-o-o-GGG-o-o-o
                    # | | | / \ | | |
                    self.contract_index(self.mps[loc][-1].inds[-1]) # type: ignore

                else:
                    # Apply a two-site gate using TEBD and then split resulting
                    # tensor to retrieve the MPS form:
                    #     -o-o-A-B-o-o-
                    #      | | | | | |            -o-o-GGG-o-o-           -o-o-X~Y-o-o-
                    #      | | GGG | |     ==>     | | | | | |     ==>     | | | | | |
                    #      | | | | | |                 i j                     i j
                    #          i j
                    self.mps.gate_split_(
                        unitary_blocks[index - start_index].data.conj().T,
                        where=[index, index + 1]
                    )

        self.permute(shape='lpr')

    def apply_unitary_layer(
            self,
            unitary_layer: UnitaryLayer,
            inverse: bool=False
        ) -> None:
        """ Apply the unitary layer on the MPS. If inverse is True,
        we apply the inverse of the unitary layer to the MPS.

        Parameters
        ----------
        `unitary_layer` : UnitaryLayer
            The unitary layer to be applied to the MPS.
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

    def apply_unitary_layers(
            self,
            unitary_layers: list[UnitaryLayer],
            inverse: bool=False
        ) -> None:
        """ Apply the unitary layers on the MPS. If inverse is True,
        we apply the inverse of the unitary layers in reverse order
        to the MPS.

        Parameters
        ----------
        `unitary_layers` : list[UnitaryLayer]
            A list of unitary layers (list of unitaries) to be applied to the MPS.
        `inverse` : bool, optional, default=False
            Whether to apply the inverse unitary layers.

        Usage
        -----
        >>> mps.apply_unitary_layers(unitary_layers, inverse=True)
        """
        for layer in reversed(unitary_layers):
            self.apply_unitary_layer(layer, inverse=inverse)

    def fidelity_with_zero_state(self) -> float:
        """ Compute the cosine similarity between MPS state and 0 product state
        <psi|00...0>.

        Returns
        -------
        `fidelity` : float
            The fidelity of the MPS with the zero state.

        Usage
        -----
        >>> mps.fidelity_with_zero_state()
        """
        zero_state = np.zeros(2**self.num_sites, dtype=np.complex128)
        zero_state[0] = 1

        # Compute the current statevector of the MPS
        current_statevector = self.to_statevector(self.mps).data.flatten()

        return abs(np.dot(current_statevector.conj().T, zero_state))

    def draw(self) -> plt.Figure:
        """ Draw the MPS.

        Usage
        -----
        >>> mps.draw()
        """
        return self.mps.draw(return_fig=True)

    def __str__(self) -> str:
        """ Get the string representation of the MPS.

        Returns
        -------
        str
            The string representation of the MPS.

        Usage
        -----
        >>> str(mps)
        """
        return (
            f"{self.__class__.__name__}"
            f"(num_sites={self.num_sites}, "
            f"bond_dimension={self.bond_dimension}, "
            f"normalized={self.is_normalized}, "
            f"canonical_form={self.canonical_form})"
        )

    def __repr__(self) -> str:
        """ Get the string representation of the MPS.

        Returns
        -------
        str
            The string representation of the MPS.

        Usage
        -----
        >>> repr(mps)
        """
        return (
            f"{self.__class__.__name__}"
            f"(mps={self.mps}, "
            f"bond_dimension={self.bond_dimension})"
        )

    def __len__(self) -> int:
        """ Get the number of sites for the MPS.

        Returns
        -------
        int
            The number of sites for the MPS.

        Usage
        -----
        >>> len(mps)
        """
        return self.num_sites

    def __eq__(
            self,
            value: object
        ) -> bool:
        """ Check if two MPS instances are equal.

        Parameters
        ----------
        `value` : object
            The object to compare.

        Returns
        -------
        `bool`
            Whether the two MPS instances are equal.

        Raises
        ------
        TypeError
            If `value` is not an instance of `qmprs.primitives.MPS`.

        Usage
        -----
        >>> mps1 == mps2
        """
        if not isinstance(value, MPS):
            raise TypeError("`value` must be an instance of `qmprs.primitives.MPS`.")

        # Check if the geometry hash of the MPSs are equal
        geometry_hash_eq = \
            self.mps.geometry_hash(strict_index_order=True) == \
            value.mps.geometry_hash(strict_index_order=True)

        # Check if all the tensors in the MPSs are equal
        all_close_eq = all(do("allclose", x, y) for x, y in zip(self.mps.arrays, self.mps.arrays))

        return geometry_hash_eq and all_close_eq