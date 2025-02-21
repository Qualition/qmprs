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

""" We omit the tests for the `qmprs.primitives.MPS` class methods that are
purely wrappers around the `quimb.tensor.MatrixProductState` class methods as
they are already tested in quimb.

Additionally, we temporarily omit the layer generation tests for the `qmprs.primitives.MPS`
class as they are tested through `tests.synthesis.mps_encoding.TestSequential` class. We
will add them in the future for better coverage.
"""

from __future__ import annotations

__all__ = ["TestMPS"]

import numpy as np
from numpy.typing import NDArray
import pytest
import quimb.tensor as qtn # type: ignore
from quimb.tensor import MatrixProductState, tensor_core # type: ignore
from quimb.tensor.tensor_1d_compress import tensor_network_1d_compress
from quick.primitives import Ket
import random
from typing import Literal

from qmprs.primitives import MPS

from tests.primitives.utils import allclose_structure

# Random statevector for testing
generate_state = lambda x: np.array([random.random() + 1j * random.random() for _ in range(2**x)])


class TestMPS:
    """ `tests.mps.TestMPS` is the class for testing the `qmprs.primitives.MPS` class.
    """
    def test_init_bond_dimension_failure(self) -> None:
        """ Test the failure of initialization of the `qmprs.primitives.MPS`
        with a negative bond dimension or float bond dimension.
        """
        with pytest.raises(ValueError):
            MPS([1, 2, 3], bond_dimension=-1) # type: ignore

        with pytest.raises(ValueError):
            MPS([1, 2, 3], bond_dimension=1.2) # type: ignore

    def test_init_from_statevector_fail(self) -> None:
        """ Test the failure of initialization of the `qmprs.primitives.MPS`
        from a statevector with only one qubit.
        """
        with pytest.raises(ValueError):
            state = np.array([1, 0])
            MPS(statevector=state, bond_dimension=64)

    def test_init_from_mps_type_fail(self) -> None:
        """ Test the failure of initialization of the `qmprs.primitives.MPS`
        from an incompatible MPS type.
        """
        with pytest.raises(TypeError):
            MPS(mps="not a `qtn.MatrixProductState` instance", bond_dimension=64) # type: ignore

    def test_init_from_mps_num_sites_fail(self) -> None:
        """ Test the failure of initialization of the `qmprs.primitives.MPS`
        from an MPS with only one site.
        """
        with pytest.raises(ValueError):
            MPS(mps=qtn.MPS_rand_state(L=1, bond_dim=2, phys_dim=2), bond_dimension=64)

    def test_init_from_both_fail(self) -> None:
        """ Test the failure of initialization of the `qmprs.primitives.MPS`
        when both `mps` and `statevector` parameters are passed.
        """
        with pytest.raises(ValueError):
            state = np.array([1, 0])
            mps = qtn.MPS_rand_state(L=1, bond_dim=2, phys_dim=2)
            MPS(statevector=state, mps=mps, bond_dimension=64)

    @pytest.mark.parametrize("state", [
        generate_state(2),
        generate_state(4),
        generate_state(8)
    ])
    def test_mps_from_statevector(
            self,
            state: NDArray[np.complex128]
        ) -> None:
        """ Test the `.from_statevector()` method.

        Parameters
        ----------
        `state` : NDArray[np.complex128]
            The statevector to test.
        """
        statevector = Ket(state)

        mps = MPS.from_statevector(statevector, 64)

        checker = MatrixProductState.from_dense(statevector.data) # type: ignore
        checker = tensor_network_1d_compress(checker, max_bond=64)

        assert allclose_structure(mps, checker) # type: ignore

    @pytest.mark.parametrize("state", [
        generate_state(2),
        generate_state(4),
        generate_state(8)
    ])
    def test_mps_to_statevector(
            self,
            state: NDArray[np.complex128]
        ) -> None:
        """ Test the `.to_statevector()` method.

        Parameters
        ----------
        `state` : NDArray[np.complex128]
            The statevector to test.
        """
        statevector = Ket(state)

        mps = MatrixProductState.from_dense(statevector.data) # type: ignore
        tensor_core.tensor_compress_bond(mps[0], mps[1], max_bond=64)

        statevector_from_mps = MPS.to_statevector(mps)

        assert statevector == statevector_from_mps

    @pytest.mark.parametrize("mps_data, is_normalized", [
        [qtn.MPS_rand_state(L=8, bond_dim=64, phys_dim=2, normalize=False), False],
        [qtn.MPS_rand_state(L=10, bond_dim=64, phys_dim=2, normalize=False), False],
        [qtn.MPS_rand_state(L=8, bond_dim=64, phys_dim=2, normalize=True), True],
        [qtn.MPS_rand_state(L=10, bond_dim=64, phys_dim=2, normalize=True), True]
    ])
    def test_normalize(
            self,
            mps_data: qtn.MatrixProductState,
            is_normalized: bool
        ) -> None:
        """ Test the `.normalize()` method.

        Parameters
        ----------
        `mps_data` : qtn.MatrixProductState
            The MPS data to test.
        `is_normalized` : bool
            The expected normalization status of the MPS.
        """
        mps = MPS(mps=mps_data, bond_dimension=64)
        assert mps.is_normalized == is_normalized

        mps.normalize()
        assert mps.is_normalized
        assert mps.norm == pytest.approx(1.0)

    @pytest.mark.parametrize("mps_data, mode, normalize", [
        [qtn.MPS_rand_state(L=8, bond_dim=64, phys_dim=2, normalize=False), "left", False],
        [qtn.MPS_rand_state(L=10, bond_dim=64, phys_dim=2, normalize=False), "right", False],
        [qtn.MPS_rand_state(L=8, bond_dim=64, phys_dim=2, normalize=True), "left", True],
        [qtn.MPS_rand_state(L=10, bond_dim=64, phys_dim=2, normalize=True), "right", True],
    ])
    def test_canonicalize(
            self,
            mps_data: qtn.MatrixProductState,
            mode: Literal["left", "right"],
            normalize: bool
        ) -> None:
        """ Test the `.canonicalize()` method.

        Parameters
        ----------
        `mps_data` : qtn.MatrixProductState
            The MPS data to test.
        `mode` : Literal["left", "right"]
            The canonical form to test.
        `normalize` : bool
            The expected normalization status of the MPS.
        """
        mps = MPS(mps=mps_data, bond_dimension=64)

        mps.canonicalize(mode=mode, normalize=normalize)

        assert mps.is_normalized == normalize
        assert mps.canonical_form == mode

    @pytest.mark.parametrize("state, max_bond, mode", [
        [generate_state(2),  32, "left"],
        [generate_state(4), 16, "right"],
        [generate_state(8), 8, "left"]
    ])
    def test_compress_mode(
            self,
            state: NDArray[np.complex128],
            max_bond: int,
            mode: Literal["left", "right"]
        ) -> None:
        """ Test the `.compress()` method with specifying a mode.
        """
        mps = MPS(statevector=state, bond_dimension=64)
        assert mps.bond_dimension == 64

        mps.compress(max_bond_dimension=max_bond, mode=mode)
        assert mps.bond_dimension == max_bond
        assert mps.canonical_form == mode