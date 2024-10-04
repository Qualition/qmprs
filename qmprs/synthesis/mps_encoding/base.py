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

__all__ = ["MPSEncoder"]

from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
from typing import Type, Literal
from qickit.circuit import Circuit # type: ignore
from qickit.primitives import Ket # type: ignore

from qmprs.mps import MPS


class MPSEncoder(ABC):
    """ `qmprs.synthesis.mps_encoding.MPSEncoder` is the class for preparing quantum states using Matrix Product
    States (MPS).

    Parameters
    ----------
    `circuit_framework` : type[qickit.circuit.Circuit]
        The quantum circuit framework.

    Attributes
    ----------
    `circuit_framework` : type[qickit.circuit.Circuit]
        The quantum circuit framework.

    Usage
    -----
    >>> encoder = MPSEncoder(Circuit)
    """
    def __init__(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Initialize a `qickit.MPSEncoder` instance.
        """
        self.circuit_framework = circuit_framework

    def prepare_state(
            self,
            statevector: Ket | NDArray[np.complex128],
            bond_dimension: int,
            compression_percentage: float=0.0,
            index_type: Literal["row", "snake"]="row",
            **kwargs
        ) -> Circuit:
        """ Prepare the quantum state using statevector.

        Parameters
        ----------
        `statevector` : qickit.primitives.Ket | NDArray[np.complex128]
            The statevector of the quantum system.
        `bond_dimension` : int
            The maximum bond dimension.
        `compression_percentage` : float
            The compression percentage.
        `index_type` : Literal["row", "snake"], optional, default="row"
            The indexing type.
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        `circuit` : qickit.circuit.Circuit
            The quantum circuit.

        Usage
        -----
        >>> circuit = encoder.prepare_state(statevector,
        ...                                 bond_dimension,
        ...                                 compression_percentage,
        ...                                 index_type,
        ...                                 **kwargs)
        """
        if not isinstance(statevector, Ket):
            statevector = Ket(statevector)

        statevector.change_indexing(index_type) # type: ignore

        if compression_percentage > 0.0:
            statevector.compress(compression_percentage)

        mps = MPS(statevector, bond_dimension=bond_dimension)

        return self.prepare_mps(mps, **kwargs)

    @abstractmethod
    def prepare_mps(
            self,
            mps: MPS,
            **kwargs
        ) -> Circuit:
        """ Prepare the quantum state using MPS.

        Parameters
        ----------
        `mps` : qmprs.mps.MPS
            The MPS to be prepared.
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        `circuit` : qickit.circuit.Circuit
            The quantum circuit.

        Usage
        -----
        >>> circuit = encoder.prepare_mps(mps, **kwargs)
        """