# Copyright (c) 2022-2026 Jared Magnusson
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from minuteman.eq import normal_shock


def test_lookup_table_by_pt_tablecheck():
    """This test comes from Table 14.1 in
    Anderson, J. D. (1989). Hypersonic and high temperature gas
    dynamics. AIAA.
    """
    u1 = 10972.8  # m/s
    p1 = 63.660  # Pa
    T1 = 269.5383

    soln = normal_shock.lookup_table_by_pressure_temperature(p1, T1, u1)
    assert soln.pressure_ratio == pytest.approx(1387, rel=5e-2)
    assert soln.density_ratio == pytest.approx(15.19, rel=1e-2)
    assert soln.temperature_ratio == pytest.approx(41.64, rel=5e-2)


def test_lookup_table_by_pt_shock_cal():
    # Expected answer taken from ShockCal 2.10.16.2001
    # https://www.engapplets.vt.edu/fluids/shockcal/shockCalHelp.html
    soln = normal_shock.lookup_table_by_pressure_temperature(
        pressure=200, temperature=300, velocity=4000
    )
    np.testing.assert_equal(soln.velocity_upstream, 4000.0)
    np.testing.assert_allclose(soln.velocity_downstream, 381.734635, rtol=2e-2)
    np.testing.assert_allclose(soln.mach_upstream, 11.6129507, rtol=1e-2)
    np.testing.assert_allclose(
        soln.states_upstream.density_mass, 0.002322844, rtol=5e-3
    )

    np.testing.assert_allclose(soln.mach_downstream, 0.297241949, rtol=7e-2)
    np.testing.assert_allclose(soln.pressure_ratio, 169.093349, rtol=2.2e-3)
    np.testing.assert_allclose(soln.temperature_ratio, 13.6312828, rtol=2e-2)
    np.testing.assert_allclose(soln.density_ratio, 10.4784833, rtol=2e-2)


@pytest.mark.parametrize(
    ("p1", "T1", "u1"),
    [
        # 9.631e-6 atm, 166 K, 16.17 kft/s, T2=4600 K
        (0.975861, 166.0, 4928.616),
    ],
)
def test_lookup_table_by_pt_huber(p1, T1, u1):
    soln = normal_shock.lookup_table_by_pressure_temperature(
        pressure=p1,
        temperature=T1,
        velocity=u1,
    )
    np.testing.assert_equal(soln.velocity_upstream, u1)
    np.testing.assert_allclose(soln.states_upstream.P, p1, rtol=1e-12)
    np.testing.assert_allclose(soln.states_upstream.T, T1, rtol=1e-12)

    np.testing.assert_allclose(soln.states_downstream.T, 4600, rtol=8e-4)
    np.testing.assert_allclose(soln.mach_upstream, 19.07, rtol=1.5e-3)
    np.testing.assert_allclose(soln.pressure_ratio, 471.5, rtol=3e-3)
    np.testing.assert_allclose(soln.temperature_ratio, 27.71, rtol=8e-4)
    # ruff: disable[ERA001]
    # np.testing.assert_allclose(soln.speed_of_sound_ratio, 5.340, rtol=1e-6)
    # np.testing.assert_allclose(soln.mach_downstream, 0.2727, rtol=1e-6)
    # ruff: enable[ERA001]
