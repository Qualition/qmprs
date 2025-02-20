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

import abc
import numpy as np
from abc import ABC, abstractmethod
from numpy.typing import NDArray
from qmprs.primitives import MPS
from quick.circuit import Circuit
from quick.primitives import Ket
from typing import Literal, Type

__all__ = ["MPSEncoder"]

class MPSEncoder(ABC, metaclass=abc.ABCMeta):
    circuit_framework: Type[Circuit]
    def __init__(self, circuit_framework: type[Circuit]) -> None: ...
    def prepare_state(self, statevector: Ket | NDArray[np.complex128], bond_dimension: int, compression_percentage: float = 0.0, index_type: Literal["row", "snake"] = "row", **kwargs) -> Circuit: ...
    @abstractmethod
    def prepare_mps(self, mps: MPS, **kwargs) -> Circuit: ...
