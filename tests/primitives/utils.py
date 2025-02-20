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

from __future__ import annotations

__all__ = ["allclose_structure"]

from autoray import do
from quimb.tensor import TensorNetwork # type: ignore


def allclose_structure(
        tna: TensorNetwork,
        tnb: TensorNetwork
    ) -> bool:
    """ Check if two `quimb.tensor.TensorNetwork` instances are equal in structure.

    Parameters
    ----------
    `tna` : TensorNetwork
        The first tensor network.
    `tnb` : TensorNetwork
        The second tensor network.

    Returns
    -------
    bool
        True if the two tensor networks are equal in structure, False otherwise.
    """
    geometry_hash_eq = \
        tna.geometry_hash(strict_index_order=True) == tnb.geometry_hash(strict_index_order=True)
    all_close_eq = all(do("allclose", x, y) for x, y in zip(tna.arrays, tnb.arrays))
    return geometry_hash_eq and all_close_eq