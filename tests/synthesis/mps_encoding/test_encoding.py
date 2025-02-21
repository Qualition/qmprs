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

__all__ = ["Template"]

from abc import ABC, abstractmethod


class Template(ABC):
    """ `tests.synthesis.mps_encoding.Template` is the template for creating MPS encoding testers.
    """
    @abstractmethod
    def test_prepare_state(self) -> None:
        """ Test the preparation of the MPS from a statevector.
        """

    @abstractmethod
    def test_prepare_mps(self) -> None:
        """ Test the preparation of the MPS from a MPS.
        """