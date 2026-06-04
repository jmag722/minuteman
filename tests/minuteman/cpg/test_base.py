import numpy as np

import minuteman.cpg.base as base


def test_speed_of_sound_from_temperature():
    actual = base.speed_of_sound_from_temperature(
        temperature=300, gas_constant=287, specific_heat_ratio=1.4
    )
    expected = np.array([347.188709494])
    np.testing.assert_allclose(actual, expected)


def test_speed_of_sound_from_pressure():
    actual = base.speed_of_sound_from_pressure(
        specific_heat_ratio=1.1, pressure=1e5, density=1.2
    )
    expected = 302.76503541
    np.testing.assert_allclose(actual, expected)


def test_mach_number():
    np.testing.assert_equal(base.mach_number(1, 4), np.array([0.25]))
