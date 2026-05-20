import numpy as np
import pytest

import minuteman.thermodynamics.caloric_perfect as calp


def test_boltzmann():
    assert calp.boltzmann_imperial == pytest.approx(5.6573e-24)


def test_universal_gas_constant():
    assert calp.universal_gas_constant_si == pytest.approx(8314.462)
    assert calp.universal_gas_constant_imperial_lbm == pytest.approx(
        1545.35, rel=1e-5)
    assert calp.universal_gas_constant_imperial_slug == pytest.approx(
        49720.0, rel=1e-5)


def test_gas_constant_air():
    assert calp.gas_constant_air_si == pytest.approx(287.055)
    assert calp.gas_constant_air_imperial_lbm == pytest.approx(
        53.353, rel=1e-5)
    assert calp.gas_constant_air_imperial_slug == pytest.approx(
        1716.57, rel=1e-5)


def test_entropy_change_tp():
    actual = calp.entropy_change_tp(
        temperature_ratio=2,
        pressure_ratio=1.5,
        specific_heat_constant_pressure=1.3,
        gas_constant=287
    )
    expected = np.array([-115.4673947])
    np.testing.assert_allclose(actual, expected)


def test_entropy_change_tv():
    actual = calp.entropy_change_tv(
        temperature_ratio=11,
        specific_volume_ratio=0.1,
        specific_heat_constant_volume=0.9,
        gas_constant=300
    )
    expected = np.array([-688.6174222])
    np.testing.assert_allclose(actual, expected)


def test_entropy_pv():
    actual = calp.entropy_change_pv(
        pressure_ratio=3,
        specific_volume_ratio=0.5,
        specific_heat_constant_pressure=25,
        specific_heat_constant_volume=1.4
    )
    expected = np.array([-15.79062231])
    np.testing.assert_allclose(actual, expected)


def test_isentropic_process_from_temperature():
    actual = calp.isentropic_process_from_temperature(
        temperature_ratio=0.7, specific_heat_ratio=1.35)
    expected = calp.IsentropicProcessResult(
        temperature_ratio=np.array([0.7]),
        pressure_ratio=np.array([0.2526509942]),
        density_ratio=np.array([0.3609299917]),
        speed_of_sound_ratio=np.array([0.8366600265]),
        specific_heat_ratio=np.array([1.35])
    )
    np.testing.assert_allclose(actual.temperature_ratio,
                               expected.temperature_ratio, rtol=0.0)
    np.testing.assert_allclose(actual.pressure_ratio,
                               expected.pressure_ratio)
    np.testing.assert_allclose(actual.density_ratio,
                               expected.density_ratio)
    np.testing.assert_allclose(actual.speed_of_sound_ratio,
                               expected.speed_of_sound_ratio)
    np.testing.assert_allclose(actual.specific_heat_ratio,
                               expected.specific_heat_ratio, rtol=0.0)


def test_isentropic_process_from_pressure():
    actual = calp.isentropic_process_from_pressure(
        pressure_ratio=1.2, specific_heat_ratio=1.4)
    expected = calp.IsentropicProcessResult(
        temperature_ratio=np.array([1.053472524]),
        pressure_ratio=np.array([1.2]),
        density_ratio=np.array([1.139089983]),
        speed_of_sound_ratio=np.array([1.026388096]),
        specific_heat_ratio=np.array([1.4])
    )
    np.testing.assert_allclose(actual.temperature_ratio,
                               expected.temperature_ratio)
    np.testing.assert_allclose(actual.pressure_ratio,
                               expected.pressure_ratio, rtol=0.0)
    np.testing.assert_allclose(actual.density_ratio,
                               expected.density_ratio)
    np.testing.assert_allclose(actual.speed_of_sound_ratio,
                               expected.speed_of_sound_ratio)
    np.testing.assert_allclose(actual.specific_heat_ratio,
                               expected.specific_heat_ratio, rtol=0.0)


def test_isentropic_process_from_density():
    actual = calp.isentropic_process_from_density(
        density_ratio=2, specific_heat_ratio=1.3)
    expected = calp.IsentropicProcessResult(
        temperature_ratio=np.array([1.231144413]),
        pressure_ratio=np.array([2.462288827]),
        density_ratio=np.array([2]),
        speed_of_sound_ratio=np.array([1.109569472]),
        specific_heat_ratio=np.array([1.3])
    )
    np.testing.assert_allclose(actual.temperature_ratio,
                               expected.temperature_ratio)
    np.testing.assert_allclose(actual.pressure_ratio,
                               expected.pressure_ratio)
    np.testing.assert_allclose(actual.density_ratio,
                               expected.density_ratio, rtol=0.0)
    np.testing.assert_allclose(actual.speed_of_sound_ratio,
                               expected.speed_of_sound_ratio)
    np.testing.assert_allclose(actual.specific_heat_ratio,
                               expected.specific_heat_ratio, rtol=0.0)


def test_isentropic_process_from_speed_of_sound():
    actual = calp.isentropic_process_from_speed_of_sound(
        speed_of_sound_ratio=1.125, specific_heat_ratio=1.4)
    expected = calp.IsentropicProcessResult(
        temperature_ratio=np.array([1.265625]),
        pressure_ratio=np.array([2.280697346]),
        density_ratio=np.array([1.802032471]),
        speed_of_sound_ratio=np.array([1.125]),
        specific_heat_ratio=np.array([1.4])
    )
    np.testing.assert_allclose(actual.temperature_ratio,
                               expected.temperature_ratio)
    np.testing.assert_allclose(actual.pressure_ratio,
                               expected.pressure_ratio)
    np.testing.assert_allclose(actual.density_ratio,
                               expected.density_ratio)
    np.testing.assert_allclose(actual.speed_of_sound_ratio,
                               expected.speed_of_sound_ratio, rtol=0.0)
    np.testing.assert_allclose(actual.specific_heat_ratio,
                               expected.specific_heat_ratio, rtol=0.0)


def test_entropy_state():
    actual = calp.entropy_state(
        pressure=100, density=2.5, specific_heat_ratio=1.3, gas_constant=200
    )
    expected = np.array([2275.9948230344603])
    np.testing.assert_allclose(actual, expected)


def test_total_energy():
    np.testing.assert_allclose(
        calp.total_energy(100, 1.2, 5000, 1.3),
        np.array([15000333.333333334])
    )


def test_specific_enthalpy():
    np.testing.assert_equal(
        calp.specific_enthalpy(100, 1e4, 0.8),
        np.array([12600])
    )
