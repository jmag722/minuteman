import numpy as np

import minuteman.cpg.normal_shock as normal_shock
from minuteman.cpg.normal_shock import NormalShockTable


def compare_tables(actual: NormalShockTable, expected: NormalShockTable,
                   **kwargs):
    np.testing.assert_allclose(
        actual.mach_upstream, expected.mach_upstream, **kwargs)
    np.testing.assert_allclose(
        actual.mach_downstream, expected.mach_downstream, **kwargs)
    np.testing.assert_allclose(
        actual.temperature_ratio, expected.temperature_ratio, **kwargs)
    np.testing.assert_allclose(
        actual.pressure_ratio, expected.pressure_ratio, **kwargs)
    np.testing.assert_allclose(
        actual.density_ratio, expected.density_ratio, **kwargs)
    np.testing.assert_allclose(
        actual.total_pressure_ratio, expected.total_pressure_ratio, **kwargs)
    np.testing.assert_allclose(
        actual.pitot_pressure_ratio, expected.pitot_pressure_ratio, **kwargs)
    np.testing.assert_allclose(
        actual.specific_heat_ratio, expected.specific_heat_ratio, **kwargs)


def test_lookup_table_by_upstream_mach():
    actual = normal_shock.lookup_table_by_upstream_mach(3.0)
    expected = NormalShockTable(
        mach_upstream=np.array([3.0]),
        mach_downstream=np.array([0.4752]),
        temperature_ratio=np.array([2.679]),
        pressure_ratio=np.array([10.33]),
        density_ratio=np.array([3.857]),
        total_pressure_ratio=np.array([0.3283]),
        pitot_pressure_ratio=np.array([12.06]),
        specific_heat_ratio=np.array([1.4])
    )
    compare_tables(actual, expected, rtol=1e-3)


def test_lookup_table_by_downstream_mach():
    actual = normal_shock.lookup_table_by_downstream_mach(0.9805)
    expected = NormalShockTable(
        mach_upstream=np.array([1.020]),
        mach_downstream=np.array([0.9805]),
        temperature_ratio=np.array([1.013]),
        pressure_ratio=np.array([1.047]),
        density_ratio=np.array([1.033]),
        total_pressure_ratio=np.array([1.0]),
        pitot_pressure_ratio=np.array([1.938]),
        specific_heat_ratio=np.array([1.4])
    )
    compare_tables(actual, expected, rtol=1e-3)


def test_lookup_table_by_pressure():
    actual = normal_shock.lookup_table_by_pressure(2916)
    expected = NormalShockTable(
        mach_upstream=np.array([50]),
        mach_downstream=np.array([0.3784]),
        temperature_ratio=np.array([487.1]),
        pressure_ratio=np.array([2916]),
        density_ratio=np.array([5.988]),
        total_pressure_ratio=np.array([0.1144e-5]),
        pitot_pressure_ratio=np.array([3219]),
        specific_heat_ratio=np.array([1.4])
    )
    compare_tables(actual, expected, rtol=1e-3)


def test_lookup_table_by_density():
    actual = normal_shock.lookup_table_by_density(2.637)
    expected = NormalShockTable(
        mach_upstream=np.array([1.98]),
        mach_downstream=np.array([0.5808]),
        temperature_ratio=np.array([1.671]),
        pressure_ratio=np.array([4.407]),
        density_ratio=np.array([2.637]),
        total_pressure_ratio=np.array([0.7302]),
        pitot_pressure_ratio=np.array([5.539]),
        specific_heat_ratio=np.array([1.4])
    )
    compare_tables(actual, expected, rtol=1e-3)


def test_lookup_table_by_temperature():
    actual = normal_shock.lookup_table_by_temperature(2.0, 1.3)
    expected = NormalShockTable(
        mach_upstream=np.array([2.66849127]),
        mach_downstream=np.array([0.47653910]),
        temperature_ratio=np.array([2.0]),
        pressure_ratio=np.array([7.91921689]),
        density_ratio=np.array([3.95960844]),
        total_pressure_ratio=np.array([0.39284291]),
        pitot_pressure_ratio=np.array([9.15630005635]),
        specific_heat_ratio=np.array([1.3])
    )
    compare_tables(actual, expected, rtol=1e-6)


def test_lookup_table_by_total_pressure():
    actual = normal_shock.lookup_table_by_total_pressure(0.6, 1.35)
    expected = NormalShockTable(
        mach_upstream=np.array([2.23100735]),
        mach_downstream=np.array([0.53469283]),
        temperature_ratio=np.array([1.78189249]),
        pressure_ratio=np.array([5.56977161]),
        density_ratio=np.array([3.12576186]),
        total_pressure_ratio=np.array([0.6]),
        pitot_pressure_ratio=np.array([6.72385620146]),
        specific_heat_ratio=np.array([1.35])
    )
    compare_tables(actual, expected, rtol=1e-6)


def test_lookup_table_by_pitot_pressure():
    actual = normal_shock.lookup_table_by_pitot_pressure(3.887, 1.35)
    expected = NormalShockTable(
        mach_upstream=np.array([1.64168094]),
        mach_downstream=np.array([0.65185322]),
        temperature_ratio=np.array([1.36978829]),
        pressure_ratio=np.array([2.94758044]),
        density_ratio=np.array([2.15185109]),
        total_pressure_ratio=np.array([0.87573717]),
        pitot_pressure_ratio=np.array([3.887]),
        specific_heat_ratio=np.array([1.35])
    )
    compare_tables(actual, expected)


def test_entropy_change():
    ds = normal_shock.entropy_change(
        total_pressure_ratio=.5615, gas_constant=1716)
    np.testing.assert_allclose(ds, np.array([990.4]), rtol=1e-4)


def test_internal_energy_change():
    de = normal_shock.internal_energy_change(
        pressure_upstream=1e5,
        pressure_downstream=3e4,
        density_upstream=1.0/0.8,
        density_downstream=1.0/3.0
    )
    np.testing.assert_allclose(de, np.array([-143000.0]))
