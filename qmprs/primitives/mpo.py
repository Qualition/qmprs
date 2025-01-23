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

__all__ = ["MPO"]

from collections.abc import Sequence
import numpy as np
from numpy.typing import NDArray
import quimb.tensor as qtn
from quimb.tensor.tensor_1d_compress import tensor_network_1d_compress # type: ignore
from typing import Literal, SupportsIndex
from quick.primitives import Operator # type: ignore


# TODO: Add analytical decomposition for MPO
class MPO:
    """ `qmprs.primitives.MPO` is the class for creating and manipulating matrix product operators (MPO).
    This class wraps the `quimb.tensor.MatrixProductOperator` class to provide a more user-friendly
    interface for creating and manipulating MPO.

    Refer to the link below for more information on the `quimb.tensor.MatrixProductOperator` class.
    https://quimb.readthedocs.io/en/latest/autoapi/quimb/tensor/tensor_1d/index.html#quimb.tensor.tensor_1d.MatrixProductOperator

    Notes
    -----
    Matrix product Operators (MPO) are a class of 1D tensor networks that are widely used
    in quantum computing to approximate a quantum operator. The MPO representation allows for a
    polynomial or even exponential reduction in the number of parameters required to represent a
    quantum operator, making it a powerful tool for quantum unitary synthesis and simulation.

    The MPO representation is defined by first performing successive SVDs on the matrix representation
    of the quantum operator. We use SVD to find low-rank structure in the tensor network and reduce
    the dimension of the tensors. We then choose a canonical form for the MPO, which is either "left"
    or "right". The canonical form of the MPO states how the singular values from the SVD are
    absorbed (contracted) with the left or right tensors.

    The MPO can be further compressed by truncating the bond dimension of the MPO. The bond
    dimension of the MPO determines the dimension of the unitary layers, and affects the
    fidelity of the approximation. This is because the bond dimension captures the entanglement
    structure of the quantum operator, where a higher bond dimension implies a higher
    degree of entanglement.

    Given each site will be represented as a $\chi\times\chi$ unitary matrix, the overall MPO will have a
    scaling of $O(N\chi^2)$, where N is the number of sites and $\chi$ is the bond dimension. Given a bond
    dimension of $2^{N/2}$ we can exactly represent any quantum operator of N qubits. However, for practical
    purposes, if we can keep the bond dimension constant, the MPO will have a linear scaling with the number
    of sites.

    MPO Diagram:
    ```
    d d d d     d d d
    |D|D|D|D...D|D|D|
    O-O-O-O-...-O-O-O
    |D|D|D|D   D|D|D|
    d d d d     d d d
    ```

    where O represents the tensor at each site, d is the physical dimension, and D is the bond dimension
    (also known as rank). For qubit systems, the physical dimension is 2.

    An important note is that the MPO representation is aimed for at least 2 qubits, as the MPO approximates
    the entanglement structure of the quantum many-body systems.

    Parameters
    ----------
    `operator` : NDArray[np.complex128], optional
        The matrix of the quantum operator.
    `mpo` : qtn.MatrixProductOperator, optional
        The matrix product operator (MPO) of the quantum operator.
    `bond_dimension` : int
        The maximum bond dimension of the MPO.

    Attributes
    ----------
    `operator` : quick.primitives.Operator
        The matrix of the quantum operator.
    `mpo` : qtn.MatrixProductOperator
        The matrix product operator (MPO) of the quantum operator.
    `bond_dimension` : int
        The maximum bond dimension of the MPO.
    `num_sites` : int
        The number of sites for the MPO.
    `physical_dimension` : int
        The physical dimension of the MPO.
    `canonical_form` : Literal["left", "right"]
        The canonical form of the MPO.
    `normalized` : bool
        Whether the MPO is normalized.

    Raises
    ------
    TypeError
        - If `bond_dimension` is not an integer.
    ValueError
        - If `bond_dimension` is less than 1.
        - `bond_dimension` must be an integer greater than 0.
        - Cannot initialize with both `operator` and `mpo`.
        - Must provide either `operator` or `mpo`.
        - Only supports MPO with physical dimension of 2.
        - The operator must have at least 2 qubits.
        - The MPO must have at least 2 tensors.

    Usage
    -----
    >>> operator = np.array([[1, 0, 0, 0],
    ...                      [0, 1, 0, 0],
    ...                      [0, 0, 1, 0],
    ...                      [0, 0, 0, 1]])
    >>> bond_dimension = 2
    >>> mpo = MPO(operator=operator,
    ...           bond_dimension=bond_dimension)
    """
    def __init__(
            self,
            operator: NDArray[np.complex128] | None = None,
            mpo: qtn.MatrixProductOperator | None = None,
            bond_dimension: int=64
        ) -> None:
        """ Initialize a `qmprs.primitives.MPO` instance. Pass only `operator` to define
        the MPO from the operator. Pass only `mpo` to define the MPO from the MPO.
        """
        if not isinstance(bond_dimension, SupportsIndex) or bond_dimension < 1:
            raise ValueError("`bond_dimension` must be an integer greater than 0.")

        if operator is not None and mpo is not None:
            raise ValueError("Cannot initialize with both `operator` and `mpo`.")

        # Initialize the MPO from the operator
        elif operator is not None:
            if not isinstance(operator, Operator):
                operator = Operator(operator)

            if operator.num_qubits == 1: # type: ignore
                raise ValueError("The operator must have at least 2 qubits.")

            self.operator = operator
            self.mpo = self.from_operator(operator, bond_dimension)

        # Initialize the MPO from the MPO
        elif mpo is not None:
            if not isinstance(mpo, qtn.MatrixProductOperator):
                raise TypeError("`mpo` must be a `qtn.MatrixProductOperator` instance.")

            if mpo.num_tensors == 1:
                raise ValueError("The MPO must have at least 2 tensors.")

            self.mpo = mpo
            self.operator = self.to_operator(mpo)

        else:
            raise ValueError("Must provide either `operator` or `mpo`.")

        # Check if the MPO is normalized to 2-norm
        self.normalized = self.mpo.norm() == 1

        # Define the maximum bond dimension
        self.bond_dimension = bond_dimension

        # Define the number of sites for the MPO
        self.num_sites = self.operator.num_qubits # type: ignore

        # The physical dimension must be equal to two, given we define the synthesis
        # for qubit-based paradigms
        if self.mpo.phys_dim() != 2:
            raise ValueError("Only supports MPO with physical dimension of 2.")
        self.physical_dimension = 2

    @staticmethod
    def from_operator(
            operator: Operator,
            max_bond_dimension: int
        ) -> qtn.MatrixProductOperator:
        """ Define the MPO from the operator.

        Parameters
        ----------
        `operator` : quick.primitives.Operator
            The matrix of the quantum operator.

        Returns
        -------
        `mpo` : qtn.MatrixProductOperator
            The MPO of the quantum operator.

        Usage
        -----
        >>> operator = Operator(np.array([[1, 0, 0, 0],
        ...                               [0, 1, 0, 0],
        ...                               [0, 0, 1, 0],
        ...                               [0, 0, 0, 1]]))
        >>> max_bond_dimension = 2
        >>> mpo = MPO.from_operator(operator, max_bond_dimension)
        """
        # Generate MPO from the operator
        mpo = qtn.MatrixProductOperator.from_dense(operator.data)

        # Compress the bond dimension of the MPO to the maximum bond dimension specified
        # This is to ensure the resulting MPO bond dimension is equal to the maximum
        # bond dimension specified during initialization
        mpo = tensor_network_1d_compress(tn=mpo, max_bond=max_bond_dimension)

        return mpo

    @staticmethod
    def to_operator(mpo: qtn.MatrixProductOperator) -> Operator:
        """ Convert the MPO to a `quick.primitives.Operator` matrix instance.

        Parameters
        ----------
        `mpo` : qtn.MatrixProductOperator
            The matrix product operator (MPO) of the quantum operator.

        Returns
        -------
        quick.primitives.Operator
            The matrix of the quantum operator.

        Usage
        -----
        >>> mpo = qtn.MatrixProductOperator.random([2, 2, 2, 2], 2)
        >>> operator = MPO.to_operator(mpo)
        """
        return Operator(mpo.to_dense())

    def normalize(self) -> None:
        """ Normalize the MPO.

        Usage
        -----
        >>> mpo.normalize()
        """
        pass

    def canonicalize(
            self,
            mode: Literal["left", "right"],
            normalize=False
        ) -> None:
        """ Convert the MPO to the canonical form. This states how the singular values from
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
        >>> mpo.canonicalize("left")
        >>> mpo.canonicalize("right", normalize=True)
        """
        if mode == "left":
            self.mpo.left_canonize(normalize=normalize)
            self.canonical_form = "left"
        elif mode == "right":
            self.mpo.right_canonize(normalize=normalize)
            self.canonical_form = "right"
        else:
            raise ValueError("`mode` must be either 'left' or 'right'.")

    def compress(
            self,
            max_bond_dimension: int | None = None,
            mode: Literal["left", "right"] | None = None
        ) -> None:
        """ SVD Compress the bond dimension of the MPO.

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
            The maximum bond dimension of the MPO.
        `mode` : Literal["left", "right"], optional
            The mode of compression.

        Raises
        ------
        ValueError
            - If `mode` is not "left", or "right".

        Usage
        -----
        >>> mpo.compress(max_bond_dimension=16)
        >>> mpo.compress(mode="left")
        >>> mpo.compress(max_bond_dimension=32, mode="right")
        """
        if not (max_bond_dimension or mode):
            self.mpo.compress()

        elif not mode:
            self.mpo = tensor_network_1d_compress(tn=self.mpo, max_bond=max_bond_dimension)

        else:
            if mode in ["left", "right", "flat"]:
                if not max_bond_dimension:
                    self.mpo.compress(form=mode)
                else:
                    self.mpo.compress(form=mode, max_bond=max_bond_dimension)
            else:
                raise ValueError(f"`mode` must be either 'left', or 'right'. Received {mode}.")

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
        >>> mpo.contract([0, 1])
        """
        if not isinstance(sites, Sequence) or not all(isinstance(index, int) for index in sites):
            raise TypeError("`sites` must be a collection of integers.")

        self.mpo ^= (self.mpo.site_tag(i) for i in sites)

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
        >>> mpo.contract_ind("k0")
        """
        self.mpo.contract_ind(index)

    def permute(
            self,
            shape: Literal["lrp", "lpr"]
        ) -> None:
        """ Permute the indices of each tensor in the MPO to match `shape`.
        This is used to define the orthogonality of the MPO.

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
        >>> mpo.permute("lrp")
        >>> mpo.permute("lpr")
        """
        if shape not in ["lrp", "lpr"]:
            raise ValueError(f"`shape` must be either 'lrp' or 'lpr'. Received {shape}.")

        self.mpo.permute_arrays(shape)