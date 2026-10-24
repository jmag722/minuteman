# Copyright (c) 2022-2026 Jared Magnusson
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from minuteman import eq
from minuteman.utils.bounds_check import OutOfBoundsError


def test_initialize_states_tpx():
    pair = "TP"
    property1 = [300.5, 320.5]
    property2 = [100.5, 1000.3]
    actual = eq.initialize_states(
        state_pair=pair,
        property1=property1,
        property2=property2,
    )
    np.testing.assert_allclose(actual.T, property1, rtol=1e-9)
    np.testing.assert_allclose(actual.P, property2, rtol=1e-9)
    n2_idx = actual.species_index("N2")
    xn2 = actual.X[..., n2_idx]
    o2_idx = actual.species_index("O2")
    xo2 = actual.X[..., o2_idx]
    np.testing.assert_allclose(xn2, [0.79, 0.79], rtol=1e-9)
    np.testing.assert_allclose(xo2, [0.21, 0.21], rtol=1e-9)


def test_initialize_states_tdy():
    pair = "TD"
    property1 = [300.5, 320.5]
    property2 = [0.05, 0.1]
    y = "N2: 0.767, O2: 0.233"
    actual = eq.initialize_states(
        state_pair=pair,
        property1=property1,
        property2=property2,
        mass_fraction=y,
    )
    np.testing.assert_allclose(actual.T, property1, rtol=1e-9)
    np.testing.assert_allclose(actual.density_mass, property2, rtol=1e-9)
    n2_idx = actual.species_index("N2")
    yn2 = actual.Y[..., n2_idx]
    o2_idx = actual.species_index("O2")
    yo2 = actual.Y[..., o2_idx]
    np.testing.assert_allclose(yn2, [0.767, 0.767], rtol=1e-9)
    np.testing.assert_allclose(yo2, [0.233, 0.233], rtol=1e-9)


@pytest.mark.parametrize(
    ("pair", "prop1", "prop2"),
    [
        ("TP", -100, 1e3),
        ("HP", -100, -1),
        ("SP", -100, -1),
        ("SV", 300, -1),
        ("TD", 100, -1),
        ("UV", -1, -5),
    ],
)
def test_out_of_bounds(pair, prop1, prop2):
    with pytest.raises(OutOfBoundsError):
        eq.initialize_states(pair, prop1, prop2)
