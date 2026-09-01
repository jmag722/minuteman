# Copyright (c) 2022-2026 Jared Magnusson
# SPDX-License-Identifier: Apache-2.0

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
