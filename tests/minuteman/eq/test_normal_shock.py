# Copyright (c) 2022-2026 Jared Magnusson
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pytest
import scipy.constants as sc

from minuteman.eq import normal_shock
from minuteman.utils.bounds_check import OutOfBoundsError

current_dir = Path(__file__).parent


def test_lookup_table_by_pt_anderson():
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
    np.testing.assert_equal(soln.upstream.velocity, 4000.0)
    np.testing.assert_allclose(soln.downstream.velocity, 381.734635, rtol=2e-2)
    np.testing.assert_allclose(soln.upstream.mach, 11.6129507, rtol=1e-2)
    np.testing.assert_allclose(soln.upstream.density, 0.002322844, rtol=5e-3)

    np.testing.assert_allclose(soln.downstream.mach, 0.297241949, rtol=7e-2)
    np.testing.assert_allclose(soln.downstream.velocity, 381.734635, rtol=2e-2)
    np.testing.assert_allclose(
        soln.downstream.density, 0.024339886, rtol=1.6e-2
    )
    np.testing.assert_allclose(
        soln.downstream.pressure, 33818.6698, rtol=2.2e-3
    )
    np.testing.assert_allclose(
        soln.downstream.temperature, 4089.38484, rtol=1.9e-2
    )
    np.testing.assert_allclose(
        soln.downstream.speed_of_sound_eq, 1284.25559, rtol=9e-3
    )


def test_lookup_table_by_pt_shape():
    # this test is simply to ensure the root-finder can handle multidimensional
    # inputs
    p1 = np.array([1e2, 5e3])
    t1 = np.array([3e2, 3e3])
    p1, t1 = np.meshgrid(p1, t1, indexing="ij")
    u1 = np.zeros_like(p1) + 5e3
    soln = normal_shock.lookup_table_by_pressure_temperature(p1, t1, u1)
    assert soln.pressure_ratio.shape == p1.shape


def test_lookup_table_by_pt_huber_p_high():
    """Eq. Normal Shock Parameters
    Huber, P. W. (1963). Hypersonic shock-heated flow parameters for velocities
    to 46,000 feet per second and altitudes to 323,000 feet.
    National Aeronautics and Space Administration.

    Table IV (b), pg 26; 59,800 ft; T1=217 K; a1=968 ft/s; p1=0.7135e-1 atm
    """
    expected = np.genfromtxt(
        current_dir / "huber1963_table4b.csv", delimiter=",", names=True
    )
    p1 = 0.7135e-1 * sc.atm
    t1 = 217.0
    u1 = sc.kilo * sc.foot * expected["u1"]
    soln = normal_shock.lookup_table_by_pressure_temperature(
        pressure=p1,
        temperature=t1,
        velocity=u1,
    )
    m1 = expected["M1"]
    t21 = expected["T21"]
    t2 = expected["T2"]
    p21 = expected["p21"]
    r21 = expected["r21"]
    a21 = expected["a21"]
    m2 = expected["M2"]

    ne = expected["ne"] * sc.mega
    # low number densities agree poorly
    idx_ne = ne > 0
    idx_ne = ne > 1e14

    xe = expected["Xe"]
    # low number densities agree poorly
    idx_xe = xe > 3e-6

    gam_s = expected["gamma_s"]
    idx_gam = gam_s > 0

    np.testing.assert_allclose(soln.upstream.mach, m1, rtol=3.2e-3)
    np.testing.assert_allclose(soln.downstream.temperature, t2, rtol=9.4e-3)
    np.testing.assert_allclose(soln.temperature_ratio, t21, rtol=9.4e-3)
    np.testing.assert_allclose(soln.pressure_ratio, p21, rtol=7.6e-3)
    np.testing.assert_allclose(soln.density_ratio, r21, rtol=1.1e-2)
    np.testing.assert_allclose(soln.speed_of_sound_ratio, a21, rtol=7.6e-3)
    np.testing.assert_allclose(soln.downstream.mach, m2, rtol=1.1e-2)
    np.testing.assert_allclose(
        soln.downstream.mole_fraction("e-")[idx_xe], xe[idx_xe], rtol=0.15
    )
    assert (
        soln.downstream.mass_fraction("N2").shape
        == soln.downstream.mole_fraction("N2").shape
    )
    np.testing.assert_array_equal(
        soln.downstream.speed_of_sound_eq <= soln.downstream.speed_of_sound_fr,
        np.full(p21.shape, fill_value=True, dtype=bool),
    )
    np.testing.assert_allclose(
        soln.downstream.number_density("e-")[idx_ne], ne[idx_ne], rtol=0.17
    )
    np.testing.assert_allclose(
        soln.downstream.isentropic_exponent[idx_gam],
        gam_s[idx_gam],
        rtol=5.4e-3,
    )


def test_lookup_table_by_pt_huber_p_low():
    """Eq. Normal Shock Parameters
    Huber, P. W. (1963). Hypersonic shock-heated flow parameters for velocities
    to 46,000 feet per second and altitudes to 323,000 feet.
    National Aeronautics and Space Administration.

    Table IV (k), pg 35; 294,800 ft; T1=166 K; a1=847 ft/s; p1=1.064e-6 atm
    """
    expected = np.genfromtxt(
        current_dir / "huber1963_table4k.csv", delimiter=",", names=True
    )
    p1 = 1.064e-6 * sc.atm
    t1 = 166.0
    u1 = sc.kilo * sc.foot * expected["u1"]
    soln = normal_shock.lookup_table_by_pressure_temperature(
        pressure=p1,
        temperature=t1,
        velocity=u1,
    )
    m1 = expected["M1"]
    t21 = expected["T21"]
    t2 = expected["T2"]
    p21 = expected["p21"]
    r21 = expected["r21"]
    a21 = expected["a21"]
    m2 = expected["M2"]

    ne = expected["ne"] * sc.mega
    # low number densities agree poorly
    idx_ne = ne > 1e14

    gam_s = expected["gamma_s"]
    idx_gam = gam_s > 0

    np.testing.assert_allclose(soln.upstream.mach, m1, rtol=2e-3)
    np.testing.assert_allclose(soln.downstream.temperature, t2, rtol=1.1e-2)
    np.testing.assert_allclose(soln.temperature_ratio, t21, rtol=1.1e-2)
    np.testing.assert_allclose(soln.pressure_ratio, p21, rtol=4.2e-3)
    np.testing.assert_allclose(soln.density_ratio, r21, rtol=1.4e-2)
    np.testing.assert_allclose(soln.speed_of_sound_ratio, a21, rtol=9.7e-3)
    np.testing.assert_allclose(soln.downstream.mach, m2, rtol=9.0e-3)
    np.testing.assert_allclose(
        soln.downstream.number_density("e-")[idx_ne], ne[idx_ne], rtol=0.15
    )
    np.testing.assert_allclose(
        soln.downstream.isentropic_exponent[idx_gam],
        gam_s[idx_gam],
        rtol=2.9e-3,
    )
    assert (
        soln.downstream.mass_fraction("N2").shape
        == soln.downstream.mole_fraction("N2").shape
    )
    np.testing.assert_array_equal(
        soln.downstream.speed_of_sound_eq <= soln.downstream.speed_of_sound_fr,
        np.full(p21.shape, fill_value=True, dtype=bool),
    )


def test_lookup_table_by_rt_shock_cal():
    # Expected answer taken from ShockCal 2.10.16.2001
    # https://www.engapplets.vt.edu/fluids/shockcal/shockCalHelp.html
    soln = normal_shock.lookup_table_by_density_temperature(
        1e-3, 300.0, velocity=6e3
    )
    np.testing.assert_allclose(soln.upstream.pressure, 86.45696544151603)
    np.testing.assert_allclose(
        soln.downstream.density, 0.013645753, rtol=1.5e-3
    )
    np.testing.assert_allclose(soln.downstream.mach, 0.263370116, rtol=2.3e-3)
    np.testing.assert_allclose(
        soln.downstream.temperature, 5958.38335, rtol=1.5e-2
    )


@pytest.mark.parametrize(
    ("func", "p", "t", "v"),
    [
        (normal_shock.lookup_table_by_pressure_temperature, -1.0, 1e3, 1e4),
        (normal_shock.lookup_table_by_pressure_temperature, 1e2, 0.0, 1e4),
        (normal_shock.lookup_table_by_pressure_temperature, 1e2, 1e4, 0.1),
        (normal_shock.lookup_table_by_density_temperature, -1.0, 1e3, 1e4),
        (normal_shock.lookup_table_by_density_temperature, 1e-3, 0.0, 1e4),
    ],
)
def test_out_of_bounds(func, p, t, v):
    with pytest.raises(OutOfBoundsError):
        func(p, t, v)
