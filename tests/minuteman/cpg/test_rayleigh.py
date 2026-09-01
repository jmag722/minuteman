# Copyright (c) 2022-2026 Jared Magnusson
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from minuteman.cpg import FlowSpeedRegime, rayleigh
from minuteman.cpg.rayleigh import RayleighFlowTable, RayleighTemperatureRegime
from minuteman.utils.bounds_check import OutOfBoundsError


def compare_tables(
    actual: RayleighFlowTable,
    expected: RayleighFlowTable,
    **kwargs,
):
    np.testing.assert_allclose(actual.mach, expected.mach, **kwargs)
    np.testing.assert_allclose(
        actual.temperature_ratio,
        expected.temperature_ratio,
        **kwargs,
    )
    np.testing.assert_allclose(
        actual.pressure_ratio,
        expected.pressure_ratio,
        **kwargs,
    )
    np.testing.assert_allclose(
        actual.density_ratio,
        expected.density_ratio,
        **kwargs,
    )
    np.testing.assert_allclose(
        actual.velocity_ratio,
        expected.velocity_ratio,
        **kwargs,
    )
    np.testing.assert_allclose(
        actual.total_pressure_ratio,
        expected.total_pressure_ratio,
        **kwargs,
    )
    np.testing.assert_allclose(
        actual.total_temperature_ratio,
        expected.total_temperature_ratio,
        **kwargs,
    )
    np.testing.assert_allclose(
        actual.entropy_ratio,
        expected.entropy_ratio,
        **kwargs,
    )
    np.testing.assert_allclose(
        actual.specific_heat_ratio,
        expected.specific_heat_ratio,
        **kwargs,
    )


def test_lookup_table_by_mach():
    actual = rayleigh.lookup_table_by_mach(mach=np.array([2.0, 0.98]))
    expected = RayleighFlowTable(
        mach=np.array([2.0, 0.98]),
        temperature_ratio=np.array([0.52892561, 1.00635667]),
        pressure_ratio=np.array([0.36363636, 1.02364622]),
        density_ratio=np.array([0.68750000214, 1.01718034902]),
        total_pressure_ratio=np.array([1.50309597, 1.00019443]),
        total_temperature_ratio=np.array([0.79338842, 0.99971472]),
        entropy_ratio=np.array([1.21757520, 0.001193033]),
        specific_heat_ratio=np.array([1.4, 1.4]),
    )
    compare_tables(actual, expected, rtol=1e-5)


def test_lookup_table_by_pressure():
    actual = rayleigh.lookup_table_by_pressure(
        pressure_ratio=np.array([1.8, 0.1]),
        specific_heat_ratio=np.array([1.4, 1.3]),
    )
    expected = RayleighFlowTable(
        mach=np.array([0.48795003, 4.11376675]),
        temperature_ratio=np.array([0.77142857, 0.16923076]),
        pressure_ratio=np.array([1.8, 0.1]),
        density_ratio=np.array([2.33333338, 0.59090909171]),
        total_pressure_ratio=np.array([1.11905129, 13.0368330]),
        total_temperature_ratio=np.array([0.67346938, 0.52071005]),
        entropy_ratio=np.array([1.49607584, 5.39554689]),
        specific_heat_ratio=np.array([1.4, 1.3]),
    )
    compare_tables(actual, expected)


def test_lookup_table_by_temperature():
    actual = rayleigh.lookup_table_by_temperature(
        temperature_ratio=np.array([0.5, 0.1]),
        specific_heat_ratio=np.array([1.3, 1.35]),
        flow_regime=np.array(
            [
                RayleighTemperatureRegime.lowspeed,
                RayleighTemperatureRegime.highspeed,
            ],
        ),
    )
    expected = RayleighFlowTable(
        mach=np.array([0.35892634, 5.36667966]),
        temperature_ratio=np.array([0.5, 0.1]),
        pressure_ratio=np.array([1.97006097, 0.05892428]),
        density_ratio=np.array([3.94012196648, 0.58924285781]),
        total_pressure_ratio=np.array([1.16808875, 32.5671696]),
        total_temperature_ratio=np.array([0.44318444, 0.51406117]),
        entropy_ratio=np.array([3.68170227, 6.04989769]),
        specific_heat_ratio=np.array([1.3, 1.35]),
    )
    compare_tables(actual, expected, rtol=1e-6)


def test_lookup_table_by_density():
    actual = rayleigh.lookup_table_by_density(
        density_ratio=np.array([3.33333333333, 0.584])
    )
    expected = RayleighFlowTable(
        mach=np.array([0.38924947, 25.0]),
        temperature_ratio=np.array([0.59399999, 0.004691312]),
        pressure_ratio=np.array([1.98, 0.00273972]),
        density_ratio=np.array([3.333333333, 0.584]),
        total_pressure_ratio=np.array([1.16120326, 32498.9762]),
        total_temperature_ratio=np.array([0.50999999, 0.49258772]),
        entropy_ratio=np.array([2.50616270, 12.8672533]),
        specific_heat_ratio=np.array([1.4, 1.4]),
    )
    compare_tables(actual, expected, rtol=1e-5)


def test_lookup_table_by_total_pressure():
    actual = rayleigh.lookup_table_by_total_pressure(
        total_pressure_ratio=np.array([90.0, 1.2]),
        specific_heat_ratio=np.array([1.2, 1.33]),
        flow_regime=np.array(
            [FlowSpeedRegime.supersonic, FlowSpeedRegime.subsonic],
        ),
    )
    expected = RayleighFlowTable(
        mach=np.array([5.15182760, 0.28211298]),
        temperature_ratio=np.array([0.11904413, 0.35331674]),
        pressure_ratio=np.array([0.06697191, 2.10697331]),
        density_ratio=np.array([0.56258049823, 5.96341220006]),
        total_pressure_ratio=np.array([90.0, 1.2]),
        total_temperature_ratio=np.array([0.39545734, 0.30725879]),
        entropy_ratio=np.array([10.0660836, 4.93834072]),
        specific_heat_ratio=np.array([1.2, 1.33]),
    )
    compare_tables(actual, expected)


def test_lookup_table_by_total_temperature():
    actual = rayleigh.lookup_table_by_total_temperature(
        total_temperature_ratio=np.array([0.6, 0.1]),
        specific_heat_ratio=np.array([1.5, 1.5]),
        flow_regime=np.array(
            [FlowSpeedRegime.supersonic, FlowSpeedRegime.subsonic],
        ),
    )
    expected = RayleighFlowTable(
        mach=np.array([5.64015853, 0.14552929]),
        temperature_ratio=np.array([0.08377223, 0.12434164]),
        pressure_ratio=np.array([0.05131670, 2.42302494]),
        density_ratio=np.array([0.61257411404, 19.4868337208]),
        total_pressure_ratio=np.array([18.8543774, 1.26039890]),
        total_temperature_ratio=np.array([0.6, 0.1]),
        entropy_ratio=np.array([4.46922198, 7.13918354]),
        specific_heat_ratio=np.array([1.5, 1.5]),
    )
    compare_tables(actual, expected)


def test_lookup_table_by_pressure_compare_direct():
    gam = 1.4
    m1 = 1.5
    m2 = 3.0
    p1 = 10.0

    p1_pstar = rayleigh.lookup_table_by_mach(
        mach=m1,
        specific_heat_ratio=gam,
    ).pressure_ratio
    p2_pstar = rayleigh.lookup_table_by_mach(
        mach=m2,
        specific_heat_ratio=gam,
    ).pressure_ratio
    p2_from_ref = p2_pstar * (1 / p1_pstar) * p1

    p2_p1 = rayleigh.pressure_ratio_by_mach(
        mach_initial=m1,
        mach_final=m2,
        specific_heat_ratio=gam,
    )
    p2_direct = p2_p1 * p1
    np.testing.assert_allclose(p2_from_ref, 3.05, rtol=1e-3)
    np.testing.assert_allclose(p2_from_ref, p2_direct)


def test_lookup_table_by_temperature_compare_direct():
    gam = 1.4
    m1 = 1.5
    m2 = 3.0

    tt1_ttstar = rayleigh.lookup_table_by_mach(
        mach=m1,
        specific_heat_ratio=gam,
    ).total_temperature_ratio
    tt2_ttstar = rayleigh.lookup_table_by_mach(
        mach=m2,
        specific_heat_ratio=gam,
    ).total_temperature_ratio
    tt2_tt1_from_ref = tt2_ttstar / tt1_ttstar

    tt2_tt1_direct = rayleigh.total_temperature_ratio_by_mach(
        mach_initial=m1,
        mach_final=m2,
        specific_heat_ratio=gam,
    )
    np.testing.assert_allclose(tt2_tt1_direct, 0.719, rtol=1e-3)
    np.testing.assert_allclose(tt2_tt1_direct, tt2_tt1_from_ref)


def test_lookup_table_by_entropy():
    actual = rayleigh.lookup_table_by_entropy(
        entropy_ratio=np.array([2.0, 4.0]),
        specific_heat_ratio=np.array([1.4, 1.4]),
        flow_regime=np.array(
            [FlowSpeedRegime.subsonic, FlowSpeedRegime.supersonic],
        ),
    )
    expected = RayleighFlowTable(
        mach=np.array([0.43346186, 4.03619180]),
        temperature_ratio=np.array([0.67840136, 0.16555796]),
        pressure_ratio=np.array([1.90017003, 0.10080991]),
        density_ratio=np.array([2.80095254794, 0.6089100727]),
        total_pressure_ratio=np.array([1.14216936, 8.48494372]),
        total_temperature_ratio=np.array([0.58657852, 0.58747815]),
        entropy_ratio=np.array([2.0, 4.0]),
        specific_heat_ratio=np.array([1.4, 1.4]),
    )
    compare_tables(actual, expected)


@pytest.mark.parametrize(
    ("func", "val"),
    [
        (rayleigh.lookup_table_by_mach, 0.0),
        (rayleigh.lookup_table_by_pressure, 0.0),
        (rayleigh.lookup_table_by_pressure, 2.4),
        (rayleigh.lookup_table_by_density, 0.583),
        (rayleigh.lookup_table_by_temperature, 0.0),
        (rayleigh.lookup_table_by_temperature, 1.03),
        (rayleigh.lookup_table_by_entropy, -1.0),
    ],
)
def test_out_of_bounds(func, val):
    with pytest.raises(OutOfBoundsError):
        func(val, 1.4)


@pytest.mark.parametrize(
    ("func", "val", "fr"),
    [
        (
            rayleigh.lookup_table_by_total_pressure,
            0.999,
            FlowSpeedRegime.supersonic,
        ),
        (
            rayleigh.lookup_table_by_total_pressure,
            0.9,
            FlowSpeedRegime.subsonic,
        ),
        (
            rayleigh.lookup_table_by_total_pressure,
            1.27,
            FlowSpeedRegime.subsonic,
        ),
        (
            rayleigh.lookup_table_by_total_temperature,
            1.01,
            FlowSpeedRegime.supersonic,
        ),
        (
            rayleigh.lookup_table_by_total_temperature,
            0.488,
            FlowSpeedRegime.supersonic,
        ),
        (
            rayleigh.lookup_table_by_total_temperature,
            0.0,
            FlowSpeedRegime.subsonic,
        ),
        (
            rayleigh.lookup_table_by_total_temperature,
            1.1,
            FlowSpeedRegime.subsonic,
        ),
    ],
)
def test_out_of_bounds_by_regime(func, val, fr):
    with pytest.raises(OutOfBoundsError):
        func(val, 1.4, flow_regime=fr)
