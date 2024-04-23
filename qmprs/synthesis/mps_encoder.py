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

__all__ = ['MPSEncoder']

from abc import ABC, abstractmethod
from typing import Type

# Import `qickit.data.Data`
from qickit.data import Data

# Import `qickit.circuit.Circuit`
from qickit.circuit import Circuit

# Import `qickit.types.collection.Collection` and `qickit.types.collection.NestedCollection`
from qickit.types import Collection, NestedCollection

# Import `qmprs.mps.MPS`
from qmprs.mps import MPS


class MPSEncoder(ABC):
    """ `qickit.MPSEncoder` is the class for preparing quantum states using Matrix Product
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

    # TODO: Consider doing this as an override instead
    @abstractmethod
    def prepare_state(self,
                      statevector: Data | NestedCollection,
                      bond_dimension: int,
                      compression_percentage: float=0.0,
                      index_type: str='row',
                      *args,
                      **kwargs) -> Circuit:
        """ Prepare the quantum state using statevector.

        Parameters
        ----------
        `statevector` : Data | NestedCollection
            The statevector of the quantum system.
        `bond_dimension` : int
            The maximum bond dimension.
        `compression_percentage` : float
            The compression percentage.
        `index_type` : str
            The indexing type.
        `*args`
            Additional arguments.
        `**kwargs`
            Additional keyword arguments.
        """
        pass

    @abstractmethod
    def prepare_mps(self,
                    mps: MPS,
                    bond_dimension: int,
                    compression_percentage: float=0.0,
                    index_type: str='row',
                    *args,
                    **kwargs) -> Circuit:
        """ Prepare the quantum state using MPS.

        Parameters
        ----------
        `mps` : qmprs.mps.MPS
            The matrix product state (MPS).
        `bond_dimension` : int
            The maximum bond dimension.
        `compression_percentage` : float
            The compression percentage.
        `index_type` : str
            The indexing type.
        `*args`
            Additional arguments.
        `**kwargs`
            Additional keyword arguments.
        """
        pass