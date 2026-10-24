# Copyright (c) 2022-2026 Jared Magnusson
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from minuteman import eq


def test_initialize_states_tpx():
    pair = "TP"
    arr1 = [300.5, 320.5]
    arr2 = [100.5, 1000.3]
    actual = eq.initialize_states(
        property_pair=pair,
        arr1=arr1,
        arr2=arr2,
    )
    np.testing.assert_allclose(actual.T, arr1, rtol=1e-9)
    np.testing.assert_allclose(actual.P, arr2, rtol=1e-9)
    np.testing.assert_allclose(actual("N2").X, [[0.79], [0.79]], rtol=1e-9)
    np.testing.assert_allclose(actual("O2").X, [[0.21], [0.21]], rtol=1e-9)


def test_initialize_states_tdy():
    pair = "TD"
    arr1 = [300.5, 320.5]
    arr2 = [0.05, 0.1]
    y = "N2: 0.767, O2: 0.233"
    actual = eq.initialize_states(
        property_pair=pair,
        arr1=arr1,
        arr2=arr2,
        mass_fraction=y,
    )
    np.testing.assert_allclose(actual.T, arr1, rtol=1e-9)
    np.testing.assert_allclose(actual.density_mass, arr2, rtol=1e-9)
    np.testing.assert_allclose(actual("N2").Y, [[0.767], [0.767]], rtol=1e-9)
    np.testing.assert_allclose(actual("O2").Y, [[0.233], [0.233]], rtol=1e-9)
