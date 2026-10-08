# Copyright (c) 2022-2026 Jared Magnusson
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import scipy.constants as sc

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
    np.testing.assert_equal(soln.upstream.velocity, 4000.0)
    np.testing.assert_allclose(soln.downstream.velocity, 381.734635, rtol=2e-2)
    np.testing.assert_allclose(soln.upstream.mach, 11.6129507, rtol=1e-2)
    np.testing.assert_allclose(soln.upstream.density, 0.002322844, rtol=5e-3)

    np.testing.assert_allclose(soln.downstream.mach, 0.297241949, rtol=7e-2)
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
    np.testing.assert_equal(soln.upstream.velocity, u1)
    np.testing.assert_allclose(soln.upstream.pressure, p1, rtol=1e-12)
    np.testing.assert_allclose(soln.upstream.temperature, T1, rtol=1e-12)

    np.testing.assert_allclose(soln.downstream.temperature, 4600, rtol=8e-4)
    np.testing.assert_allclose(soln.upstream.mach, 19.07, rtol=1.5e-3)
    np.testing.assert_allclose(soln.pressure_ratio, 471.5, rtol=3e-3)
    np.testing.assert_allclose(soln.temperature_ratio, 27.71, rtol=8e-4)
    np.testing.assert_allclose(soln.speed_of_sound_ratio, 5.340, rtol=7e-5)
    np.testing.assert_allclose(soln.downstream.mach, 0.2727, rtol=4e-3)


def test_look_table_by_pt_huber_p_low():
    # Table IV. pg 35.
    # altitude=294800 ft; T1=166 K; a1=847 ft/s; p=1.064e-6 atm
    p1 = 1.064e-6 * sc.atm
    t1 = 166.0
    u1 = (
        sc.kilo
        * sc.foot
        * np.array(
            [
                3.837,
                4.972,
                5.943,
                7.238,
                8.391,
                9.879,
                11.19,
                11.95,
                12.35,
                12.65,
                12.95,
                13.33,
                13.86,
                14.60,
                15.65,
                17.03,
                18.70,
                20.60,
                22.57,
                26.51,
                28.29,
                29.02,
                29.71,
                30.60,
                31.91,
                33.80,
                36.32,
                39.43,
                42.95,
                50.02,
                55.20,
            ]
        )
    )
    soln = normal_shock.lookup_table_by_pressure_temperature(
        pressure=p1,
        temperature=t1,
        velocity=u1,
    )
    t2 = np.concatenate(
        (
            np.linspace(8e2, 2.0e3, 4),
            np.linspace(2.2e3, 4.8e3, 14),
            np.linspace(5e3, 10e3, 11),
            [11e3, 12e3],
        )
    )
    m1 = [
        4.528,
        5.868,
        7.014,
        8.537,
        9.897,
        11.65,
        13.20,
        14.10,
        14.57,
        14.93,
        15.28,
        15.72,
        16.35,
        17.22,
        18.46,
        20.09,
        22.06,
        24.30,
        26.62,
        31.26,
        33.37,
        34.22,
        35.04,
        36.09,
        37.64,
        39.87,
        42.84,
        46.51,
        50.66,
        59.01,
        65.11,
    ]
    p21 = [
        23.91,
        40.62,
        58.56,
        89.02,
        122.6,
        173.6,
        225.4,
        257.4,
        274.6,
        287.7,
        300.8,
        318.5,
        344.8,
        383.6,
        443.2,
        528.0,
        640.8,
        781.6,
        941.7,
        1305,
        1486,
        1559,
        1631,
        1727,
        1878,
        2108,
        2439,
        2879,
        3422,
        4651,
        5663,
    ]
    r21 = [
        4.961,
        5.618,
        6.079,
        7.256,
        8.803,
        10.83,
        12.33,
        12.73,
        12.56,
        12.30,
        12.06,
        12.00,
        12.19,
        12.66,
        13.55,
        14.78,
        16.25,
        17.82,
        19.24,
        21.08,
        20.81,
        19.84,
        19.09,
        18.60,
        18.48,
        18.73,
        19.28,
        20.00,
        20.72,
        21.54,
        21.42,
    ]
    a21 = [
        2.161,
        2.610,
        2.986,
        3.162,
        3.324,
        3.558,
        3.841,
        4.149,
        4.444,
        4.624,
        4.696,
        4.740,
        4.804,
        4.920,
        5.076,
        5.287,
        5.549,
        5.857,
        6.194,
        7.039,
        7.801,
        8.388,
        8.609,
        8.795,
        9.088,
        9.499,
        10.04,
        10.73,
        11.50,
        13.24,
        14.87,
    ]
    m2 = [
        0.4224,
        0.4003,
        0.3864,
        0.3721,
        0.3382,
        0.3024,
        0.2789,
        0.2668,
        0.2609,
        0.2625,
        0.2697,
        0.2764,
        0.2792,
        0.2764,
        0.2685,
        0.2570,
        0.2446,
        0.2330,
        0.2233,
        0.2107,
        0.2056,
        0.2057,
        0.2132,
        0.2206,
        0.2242,
        0.2241,
        0.2213,
        0.2167,
        0.2126,
        0.2069,
        0.2044,
    ]
    ne = [
        -1,
        -1,
        1.8e1,
        3.5e4,
        5.7e5,
        5.6e6,
        3.5e7,
        1.4e8,
        3.7e8,
        9.8e8,
        2.3e9,
        4.9e9,
        9.9e9,
        1.9e10,
        3.6e10,
        6e10,
        1e11,
        1.6e11,
        2.6e11,
        8.1e11,
        2.5e12,
        7.7e12,
        2.0e13,
        4.3e13,
        9.2e13,
        1.7e14,
        3e14,
        4.7e14,
        7.2e14,
        1.3e15,
        1.7e15,
    ]
    ne = np.asarray(ne) * sc.mega
    # low number densities agree poorly
    idx_ne = ne > 1e14
    gam_s = [
        -1,
        -1,
        -1,
        1.143,
        1.113,
        1.111,
        1.134,
        1.195,
        1.270,
        1.282,
        1.242,
        1.192,
        1.146,
        1.120,
        1.106,
        1.100,
        1.100,
        1.100,
        1.103,
        1.128,
        1.196,
        1.260,
        1.223,
        1.172,
        1.142,
        1.129,
        1.124,
        1.125,
        1.127,
        1.141,
        1.171,
    ]
    gam_s = np.asarray(gam_s)
    idx_gam = gam_s > 0

    np.testing.assert_allclose(soln.upstream.mach, m1, rtol=2e-3)
    np.testing.assert_allclose(soln.downstream.temperature, t2, rtol=1.1e-2)
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
