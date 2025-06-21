import pytest
import numpy as np
import minuteman.compressible.isentropic as isen
import minuteman.compressible.sod as sod


def test_solve_p21():
    gam4 = 1.4
    gam1 = 1.4
    p4 = 1e5
    p1 = 1e4
    r4 = 1
    r1 = .125
    a4 = isen.speed_sound(gam=gam4, p=p4, rho=r4)
    a1 = isen.speed_sound(gam=gam1, p=p1, rho=r1)
    actual = sod.solve_p21(p41=p4/p1, a41=a4/a1, gam4=gam4, gam1=gam1)
    assert actual == pytest.approx(3.0313017805065474)


def test_contact_surface_speed():
    # see Anderson Example 7.1
    T = 300
    pratio = 10
    gam = 1.4
    a1 = (gam*287*T)**0.5
    assert sod.contact_surface_speed(
        pratio, a1, gam) == pytest.approx(756.0737668928916)


def test_moving_shock_speed():
    # see Anderson Example 7.1
    T = 300
    pratio = 10
    gam = 1.4
    a1 = (gam*287*T)**0.5
    actual = sod.moving_shock_speed(pratio, a1, gam)
    assert actual == pytest.approx(1024.9)


def test_moving_shock_temperature_ratio():
    # see Anderson Example 7.1
    pratio = 10
    gam = 1.4
    assert sod.moving_shock_temperature_ratio(
        pratio, gam) == pytest.approx(2.622950819672131)


def test_moving_shock_density_ratio():
    # see Anderson Example 7.1
    pratio = 10
    gam = 1.4
    assert sod.moving_shock_density_ratio(pratio, gam) == pytest.approx(3.8125)


def test_velocity_expansion_fan():
    actual = sod.velocity_expansion_fan(
        a4=300, x=np.array([1, 2, 3]), t=0.1, gam=1.4)
    np.testing.assert_allclose(actual, np.array(
        [258.33333333, 266.66666667, 275.]))


def test_sound_speed_expansion_fan():
    actual = sod.sound_speed_expansion_fan(
        a4=205, u=np.array([1000, 2000, 30]), gam=1.43)
    np.testing.assert_allclose(actual, np.array([-10., -225.,  198.55]))


def test_shock_tube():
    actual = sod.shock_tube(t=0.01, p_driver=1e5, p_driven=1e4,
                            rho_driver=1, rho_driven=0.125, R_driver=287, R_driven=287,
                            gam_driver=1.4, gam_driven=1.4)
    # tested with https://onlineflowcalculator.com/pages/CFLOW/calculator.html
    actual["pressure"][actual["region1"]].mean() == pytest.approx(1e4)
    actual["pressure"][actual["region2"]].mean() == pytest.approx(3.03130e+4)
    actual["pressure"][actual["region3"]].mean() == pytest.approx(3.03130e+4)
    actual["pressure"][actual["region4"]].mean() == pytest.approx(1e5)

    actual["temperature"][actual["region1"]].mean() == pytest.approx(278.75)
    actual["temperature"][actual["region2"]].mean() == pytest.approx(397.71)
    actual["temperature"][actual["region3"]].mean() == pytest.approx(247.75)
    actual["temperature"][actual["region4"]].mean() == pytest.approx(348.43)

    actual["density"][actual["region1"]].mean() == pytest.approx(.125)
    actual["density"][actual["region2"]].mean() == pytest.approx(.27)
    actual["density"][actual["region3"]].mean() == pytest.approx(.43)
    actual["density"][actual["region4"]].mean() == pytest.approx(1)

    actual["sound_speed"][actual["region1"]].mean() == pytest.approx(334.66)
    actual["sound_speed"][actual["region2"]].mean() == pytest.approx(399.75)
    actual["sound_speed"][actual["region3"]].mean() == pytest.approx(315.51)
    actual["sound_speed"][actual["region4"]].mean() == pytest.approx(374.17)

    actual["speed"][actual["region1"]].mean() == 0.
    actual["speed"][actual["region2"]].mean() == pytest.approx(293.29)
    actual["speed"][actual["region3"]].mean() == pytest.approx(293.29)
    actual["speed"][actual["region4"]].mean() == 0.
