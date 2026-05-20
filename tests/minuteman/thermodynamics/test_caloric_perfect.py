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


class TestIsentropicRelations:
    def test_p(self, check_dicts):
        actual = calp.isentropic_process(p21=1.2)
        expected = {"p21": 1.2, "t21": 1.053472524, "r21": 1.139089983,
                    "a21": 1.026388096, "gam": 1.4}
        check_dicts(actual, expected)

    def test_a(self, check_dicts):
        actual = calp.isentropic_process(a21=1.125)
        expected = {"p21": 2.280697346, "r21": 1.802032471, "t21": 1.265625,
                    "a21": 1.125, "gam": 1.4}
        check_dicts(actual, expected)

    def test_r(self, check_dicts):
        actual = calp.isentropic_process(r21=2, gam=1.3)
        expected = {"p21": 2.462288827, "r21": 2, "t21": 1.231144413,
                    "a21": 1.109569472, "gam": 1.3}
        check_dicts(actual, expected)

    def test_t(self, check_dicts):
        actual = calp.isentropic_process(t21=0.7, gam=1.35)
        expected = {"p21": 0.2526509942, "r21": 0.3609299917, "t21": 0.7,
                    "a21": 0.8366600265, "gam": 1.35}
        check_dicts(actual, expected)


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
