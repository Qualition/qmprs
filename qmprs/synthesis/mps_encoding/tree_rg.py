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

__all__ = ["TreeRG"]

from quick.circuit import Circuit # type: ignore

from qmprs.primitives import MPS
from qmprs.synthesis.mps_encoding import MPSEncoder


class TreeRG(MPSEncoder):
    """ `qmprs.synthesis.mps_encoding.TreeRG` is the class for preparing MPS
    using Tree-RG encoding. The circuit is constructed using the tree structure
    and utilization of fixed-points described in the 2023 paper by Wei et al.

    ref: https://arxiv.org/pdf/2307.01696

    Notes
    -----
    - The circuit depth scales $O(log(N))$ where N is the number of qubits.
    # TODO: Add a better explanation on the origin of tree structure from the Prof. Chan lecture part II.
    - Tree-RG utilizes the tree structure described in DMRG literature to reduce the circuit depth.
    # TODO: Add a better explanation on the fixed-points and how they are utilized in Tree-RG.
    - Tree-RG utilizes fixed-points described in https://arxiv.org/abs/quant-ph/0410227 to approximate
    positive semidefinite matrix P as unitary matrices which are then contracted with the isometries V.
    The consequence of this is that there will be considerable long-range correlation maintained within
    the MPS.
    # TODO: Confirm this.
    - The Tree-RG encoding approach mostly allows for encoding of short-range correlated states, however,
    it can also be used for long-range correlated states. This is further improved by using mid-circuit
    measurement (adaptive) circuits implementation in `AdaptiveTreeRG`.

    Parameters
    ----------
    `circuit_framework` : Type[Circuit]
        The quantum circuit framework.

    Attributes
    ----------
    `circuit_framework` : Type[Circuit]
        The quantum circuit framework.

    Raises
    ------
    ValueError
        If the block size is not a positive integer.
        If the block size does not divide the number of sites.

    Usage
    -----
    >>> tree_rg = TreeRG(Circuit)
    """
    # TODO
    def prepare_mps(
            self,
            mps: MPS,
            **kwargs
        ) -> Circuit:

        block_size = kwargs.get("block_size")

        if not isinstance(block_size, int) or block_size < 1:
            raise ValueError("The block size must be a positive integer.")

        if mps.num_sites % block_size != 0:
            raise ValueError("The block size must divide the number of sites.")

        mps.normalize()
        mps.compress(mode="right")

        # Block/contract the MPS sites, and then polar decompose
        for site in range(0, mps.num_sites, block_size):
            mps.contract_site(range(site, site + block_size))
            mps.polar_decompose(range(site, site + block_size))

        # Define the circuit to prepare the MPS
        circuit = self.circuit_framework(mps.num_sites, mps.num_sites)

        # TODO: Implement the Tree-RG encoding circuit

        return circuit