import numpy as np
import pytest

import minuteman.cpg.shock_tube as shock_tube
from minuteman import cpg


def test_speed_of_sound_from_pressure():
    gam4 = 1.4
    gam1 = 1.4
    p4 = 1e5
    p1 = 1e4
    r4 = 1
    r1 = 0.125
    a4 = cpg.speed_of_sound_from_pressure(
        specific_heat_ratio=gam4, pressure=p4, density=r4
    )[0]
    a1 = cpg.speed_of_sound_from_pressure(
        specific_heat_ratio=gam1, pressure=p1, density=r1
    )[0]
    actual = shock_tube.moving_shock_pressure_ratio(
        pressure_ratio=p4 / p1,
        speed_of_sound_ratio=a4 / a1,
        specific_heat_ratio_driver=gam4,
        specific_heat_ratio_driven=gam1,
    )
    assert actual == pytest.approx(3.0313017805065474)


def test_contact_surface_speed():
    # see Anderson Example 7.1
    T = 300
    pratio = 10
    gam = 1.4
    a1 = (gam * 287 * T) ** 0.5
    assert shock_tube.contact_surface_speed(pratio, a1, gam) == pytest.approx(
        756.0737668928916
    )


def test_moving_shock_speed():
    # see Anderson Example 7.1
    T = 300
    pratio = 10
    gam = 1.4
    a1 = (gam * 287 * T) ** 0.5
    actual = shock_tube.moving_shock_speed(pratio, a1, gam)
    assert actual == pytest.approx(1024.9)


def test_moving_shock_temperature_ratio():
    # see Anderson Example 7.1
    pratio = 10
    gam = 1.4
    assert shock_tube.moving_shock_temperature_ratio(
        pratio, gam
    ) == pytest.approx(2.622950819672131)


def test_moving_shock_density_ratio():
    # see Anderson Example 7.1
    pratio = 10
    gam = 1.4
    assert shock_tube.moving_shock_density_ratio(pratio, gam) == pytest.approx(
        3.8125
    )


def test_expansion_fan_velocity():
    actual = shock_tube.expansion_fan_velocity(
        speed_of_sound_driver=300,
        position=np.array([1, 2, 3]),
        time=0.1,
        specfic_heat_ratio_driver=1.4,
    )
    np.testing.assert_allclose(
        actual, np.array([258.33333333, 266.66666667, 275.0])
    )


def test_expansion_fan_speed_of_sound():
    actual = shock_tube.expansion_fan_speed_of_sound(
        speed_of_sound_driver=205,
        velocity=np.array([1000, 2000, 30]),
        specific_heat_ratio_driver=1.43,
    )
    np.testing.assert_allclose(actual, np.array([-10.0, -225.0, 198.55]))


def test_shock_tube():
    actual = shock_tube.solve_sod(
        time=0.01,
        pressure_l=1e5,
        pressure_r=1e4,
        density_l=1,
        density_r=0.125,
        gas_constant_l=287,
        gas_constant_r=287,
        specific_heat_ratio_l=1.4,
        specific_heat_ratio_r=1.4,
    )
    # tested with https://onlineflowcalculator.com/pages/CFLOW/calculator.html
    actual.pressure[actual.region_1].mean() == pytest.approx(1e4)
    actual.pressure[actual.region_2].mean() == pytest.approx(3.03130e4)
    actual.pressure[actual.region_3].mean() == pytest.approx(3.03130e4)
    actual.pressure[actual.region_4].mean() == pytest.approx(1e5)

    actual.temperature[actual.region_1].mean() == pytest.approx(278.75)
    actual.temperature[actual.region_2].mean() == pytest.approx(397.71)
    actual.temperature[actual.region_3].mean() == pytest.approx(247.75)
    actual.temperature[actual.region_4].mean() == pytest.approx(348.43)

    actual.density[actual.region_1].mean() == pytest.approx(0.125)
    actual.density[actual.region_2].mean() == pytest.approx(0.27)
    actual.density[actual.region_3].mean() == pytest.approx(0.43)
    actual.density[actual.region_4].mean() == pytest.approx(1)

    actual.speed_of_sound[actual.region_1].mean() == pytest.approx(334.66)
    actual.speed_of_sound[actual.region_2].mean() == pytest.approx(399.75)
    actual.speed_of_sound[actual.region_3].mean() == pytest.approx(315.51)
    actual.speed_of_sound[actual.region_4].mean() == pytest.approx(374.17)

    actual.velocity[actual.region_1].mean() == 0.0
    actual.velocity[actual.region_2].mean() == pytest.approx(293.29)
    actual.velocity[actual.region_3].mean() == pytest.approx(293.29)
    actual.velocity[actual.region_4].mean() == 0.0
