# Copyright (c) 2022-2026 Jared Magnusson
# SPDX-License-Identifier: Apache-2.0

import cantera as ct
import numpy as np
import pytest

from minuteman.utils import types


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ((6, 5), (np.array([6]), np.array([5]))),
        ((6,), (np.array([6]),)),
        (
            ([1, 2], [[3], [4]], 5),
            (
                np.array([[1, 2], [1, 2]]),
                np.array([[3, 3], [4, 4]]),
                np.array([[5, 5], [5, 5]]),
            ),
        ),
    ],
)
def test_broadcast_inputs(args, expected):
    actual = types.broadcast_inputs(*args)
    np.testing.assert_equal(actual, expected)
    # modify args to check that copy is made
    actual2 = np.asarray(args[0]) * 10
    assert not np.array_equal(actual2, np.asarray(args[0]))


def test_eq_soln_toarray():
    gas = ct.Solution("airNASA9.yaml")
    tin = 1e4
    pin = 100.0
    rtol = 1e-12
    gas.TPX = tin, pin, "N2: 0.79, O2: 0.21"
    states = types.eq_soln_toarray(gas)
    tout = gas.T
    pout = gas.P
    assert tout == pytest.approx(states.T.item(), rel=rtol)
    assert pout == pytest.approx(states.P.item(), rel=rtol)
    # make sure original gas object unchanged
    states.equilibrate("SV")
    assert tout == pytest.approx(tin, rel=rtol)
    assert pout == pytest.approx(pin, rel=rtol)
