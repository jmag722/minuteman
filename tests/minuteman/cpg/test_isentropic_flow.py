import numpy as np

import minuteman.cpg.isentropic_flow as isentropic_flow
from minuteman.cpg import FlowSpeedRegime


def compare_tables(actual, expected, **kwargs):
    np.testing.assert_allclose(actual.mach, expected.mach, **kwargs)
    np.testing.assert_allclose(
        actual.temperature, expected.temperature, **kwargs
    )
    np.testing.assert_allclose(
        actual.temperature, expected.temperature, **kwargs
    )
    np.testing.assert_allclose(actual.pressure, expected.pressure, **kwargs)
    np.testing.assert_allclose(actual.density, expected.density, **kwargs)
    np.testing.assert_allclose(
        actual.speed_of_sound, expected.speed_of_sound, **kwargs
    )
    np.testing.assert_allclose(
        actual.specific_heat_ratio, expected.specific_heat_ratio, **kwargs
    )


def test_lookup_table_by_mach():
    actual = isentropic_flow.lookup_table_by_mach(2.0, 1.4)
    expected = isentropic_flow.IsentropicFlowTable(
        mach=np.array([2.0]),
        temperature=np.array([1.8]),
        pressure=np.array([7.824]),
        density=np.array([4.347]),
        speed_of_sound=np.array([1.342]),
        area_ratio=np.array([1.687]),
        specific_heat_ratio=np.array([1.4]),
    )
    compare_tables(actual, expected, rtol=1e-3)


def test_lookup_table_by_temperature():
    actual = isentropic_flow.lookup_table_by_temperature(1.008, 1.4)
    expected = isentropic_flow.IsentropicFlowTable(
        mach=np.array([0.2]),
        temperature=np.array([1.008]),
        pressure=np.array([1.028]),
        density=np.array([1.02]),
        speed_of_sound=np.array([1.00399203184]),
        area_ratio=np.array([2.964]),
        specific_heat_ratio=np.array([1.4]),
    )
    compare_tables(actual, expected, rtol=1e-3)


def test_lookup_table_by_area_supersonic():
    actual = isentropic_flow.lookup_table_by_area_ratio(
        1.094, 1.4, flow_regime=FlowSpeedRegime.supersonic
    )
    expected = isentropic_flow.IsentropicFlowTable(
        mach=np.array([1.36]),
        temperature=np.array([1.3698630137]),
        pressure=np.array([3.00932891965]),
        density=np.array([2.19669178218]),
        speed_of_sound=np.array([1.17041147196]),
        area_ratio=np.array([1.094]),
        specific_heat_ratio=np.array([1.4]),
    )
    compare_tables(actual, expected, rtol=1e-4)


def test_lookup_table_by_area_subsonic():
    actual = isentropic_flow.lookup_table_by_area_ratio(
        3.1, 1.3, flow_regime=FlowSpeedRegime.subsonic
    )
    expected = isentropic_flow.IsentropicFlowTable(
        mach=np.array([0.193]),
        temperature=np.array([1.0060362173]),
        pressure=np.array([1.02438024995]),
        density=np.array([1.01871377199]),
        speed_of_sound=np.array([1.00301356785]),
        area_ratio=np.array([3.1]),
        specific_heat_ratio=np.array([1.3]),
    )
    compare_tables(actual, expected, rtol=1e-3)


def test_lookup_table_by_pressure():
    actual = isentropic_flow.lookup_table_by_pressure(39.59, 1.4)
    expected = isentropic_flow.IsentropicFlowTable(
        mach=np.array([3.05]),
        temperature=np.array([2.860]),
        pressure=np.array([39.59]),
        density=np.array([13.84]),
        speed_of_sound=np.array([1.69115345253]),
        area_ratio=np.array([4.441]),
        specific_heat_ratio=np.array([1.4]),
    )
    compare_tables(actual, expected, rtol=1e-3)
