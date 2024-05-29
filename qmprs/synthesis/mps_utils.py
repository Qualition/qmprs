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

__all__ = ['check_unitaries']

import numpy as np


def check_unitaries(generated_unitary_list: list) -> None:
    """ Check if the generated unitaries are unitary.

    Parameters
    ----------
    `generated_unitary_list` : list
        The list of generated unitaries.

    Raises
    ------
    `ValueError`:
        If all the generated unitaries are not unitary.
    """
    # Iterate over the list of generated unitaries
    for _, _, generated_unitaries, _, _ in generated_unitary_list:
        # Iterate over the generated unitaries
        for generated_unitary in generated_unitaries:
            if not np.allclose(np.eye(generated_unitary.shape[0]) - generated_unitary.data @ generated_unitary.data.T.conj(), 0):
                raise ValueError("ValueError : every generated unitary in the list must be a unitary.")
            if not np.allclose(np.eye(generated_unitary.shape[0]) - generated_unitary.data.T.conj() @ generated_unitary.data, 0):
                raise ValueError("ValueError : every generated unitary in the list must be a unitary.")