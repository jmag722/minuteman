import numpy as np
import minuteman.cpg.fanno as fanno
from minuteman.cpg.fanno import FannoFlowTable
from minuteman import FlowSpeedRegime


def compare_tables(actual: FannoFlowTable, expected: FannoFlowTable, **kwargs):
    np.testing.assert_allclose(actual.mach, expected.mach, **kwargs)
    np.testing.assert_allclose(
        actual.temperature_ratio, expected.temperature_ratio, **kwargs
    )
    np.testing.assert_allclose(
        actual.pressure_ratio, expected.pressure_ratio, **kwargs
    )
    np.testing.assert_allclose(
        actual.density_ratio, expected.density_ratio, **kwargs
    )
    np.testing.assert_allclose(
        actual.velocity_ratio, expected.velocity_ratio, **kwargs
    )
    np.testing.assert_allclose(
        actual.total_pressure_ratio, expected.total_pressure_ratio, **kwargs
    )
    np.testing.assert_allclose(
        actual.fanno_parameter,
        expected.fanno_parameter,
        **kwargs,
    )
    np.testing.assert_allclose(
        actual.entropy_ratio, expected.entropy_ratio, **kwargs
    )
    np.testing.assert_allclose(
        actual.specific_heat_ratio, expected.specific_heat_ratio, **kwargs
    )


def test_lookup_table_by_mach():
    actual = fanno.lookup_table_by_mach(mach=0.4, specific_heat_ratio=1.3)
    expected = FannoFlowTable(
        mach=np.array([0.4]),
        pressure_ratio=np.array([2.64934764]),
        temperature_ratio=np.array([1.12304687]),
        density_ratio=np.array([2.35907132043]),
        entropy_ratio=np.array([0.47144997]),
        total_pressure_ratio=np.array([1.60231582]),
        fanno_parameter=np.array([2.51998734]),
        specific_heat_ratio=np.array([1.3]),
    )
    compare_tables(actual, expected)


def test_lookup_table_by_pressure():
    actual = fanno.lookup_table_by_pressure(0.9, specific_heat_ratio=1.3)
    expected = FannoFlowTable(
        mach=np.array([1.09670359]),
        pressure_ratio=np.array([0.9]),
        temperature_ratio=np.array([0.97423461]),
        density_ratio=np.array([0.92380212195]),
        entropy_ratio=np.array([0.00775303]),
        total_pressure_ratio=np.array([1.0077831]),
        fanno_parameter=np.array([0.01054915]),
        specific_heat_ratio=np.array([1.3]),
    )
    compare_tables(actual, expected, rtol=1e-6)


def test_mach2_by_pressure():
    actual = fanno.lookup_table_by_pressure(
        np.array([4.0, 0.5]), specific_heat_ratio=np.array([1.4, 1.3])
    ).mach
    expected = np.array([0.27185940, 1.76924837])
    np.testing.assert_allclose(actual, expected)


def test_lookup_table_by_density():
    actual = fanno.lookup_table_by_density(
        6.66666666667, specific_heat_ratio=1.2
    )
    expected = FannoFlowTable(
        mach=np.array([0.14316588]),
        pressure_ratio=np.array([7.31833333]),
        temperature_ratio=np.array([1.09775]),
        density_ratio=np.array([6.66666666667]),
        entropy_ratio=np.array([1.43080683]),
        total_pressure_ratio=np.array([4.18207206]),
        fanno_parameter=np.array([36.3460207]),
        specific_heat_ratio=np.array([1.2]),
    )
    compare_tables(actual, expected)


def test_mach2_by_density():
    actual = fanno.lookup_table_by_density(
        np.array([0.5, 5.0]), specific_heat_ratio=np.array([1.3, 1.5])
    ).mach
    expected = np.array([2.69679944, 0.17960530])
    np.testing.assert_allclose(actual, expected)


def test_lookup_table_by_temperature():
    actual = fanno.lookup_table_by_temperature(0.01, specific_heat_ratio=1.5)
    expected = FannoFlowTable(
        mach=np.array([22.2710574]),
        pressure_ratio=np.array([0.00449013]),
        temperature_ratio=np.array([0.01]),
        density_ratio=np.array([1 / 2.22710574]),
        entropy_ratio=np.array([8.40963750]),
        total_pressure_ratio=np.array([4490.13255]),
        fanno_parameter=np.array([0.66918220]),
        specific_heat_ratio=np.array([1.5]),
    )
    compare_tables(actual, expected, rtol=1e-6)


def test_mach2_by_temperature():
    actual = fanno.lookup_table_by_temperature(
        np.array([0.5, 1.1]), specific_heat_ratio=np.array([1.3, 1.5])
    ).mach
    expected = np.array([2.94392028, 0.73854894])
    np.testing.assert_allclose(actual, expected)


def test_lookup_table_by_entropy():
    actual = fanno.lookup_table_by_entropy(
        entropy_ratio=np.array([1.1, 1.5]),
        specific_heat_ratio=np.array([1.5, 1.1]),
        flow_regime=np.array(
            [FlowSpeedRegime.subsonic, FlowSpeedRegime.supersonic]
        ),
    )
    expected = FannoFlowTable(
        mach=np.array([0.19511262, 2.54543291]),
        pressure_ratio=np.array([5.70312323, 0.34986083]),
        temperature_ratio=np.array([1.23821561, 0.79307445]),
        density_ratio=np.array([1 / 0.21711184, 1 / 2.26682835]),
        entropy_ratio=np.array([1.1, 1.5]),
        total_pressure_ratio=np.array([3.00416602, 4.48168906]),
        fanno_parameter=np.array([14.2998521, 0.79358257]),
        specific_heat_ratio=np.array([1.5, 1.1]),
    )
    compare_tables(actual, expected)


def test_mach1_by_entropy():
    actual = fanno.lookup_table_by_entropy(
        np.array(
            [
                2.0,
                0.6,
                fanno.entropy_ratio_by_mach(
                    mach_initial=4.0, mach_final=1.0, specific_heat_ratio=1.4
                )[0]
                - 0.3,
            ]
        ),
        specific_heat_ratio=np.array([1.5, 1.3, 1.4]),
        flow_regime=np.array(
            [
                FlowSpeedRegime.subsonic,
                FlowSpeedRegime.supersonic,
                FlowSpeedRegime.supersonic,
            ]
        ),
    ).mach
    expected = np.array([0.07776356, 2.02884749, 3.66913016])
    np.testing.assert_allclose(actual, expected)


def test_lookup_table_by_total_pressure():
    actual = fanno.lookup_table_by_total_pressure(
        total_pressure_ratio=np.array([5.0, 1.5]),
        specific_heat_ratio=np.array([1.4, 1.1]),
        flow_regime=np.array(
            [FlowSpeedRegime.subsonic, FlowSpeedRegime.supersonic]
        ),
    )
    expected = FannoFlowTable(
        mach=np.array([0.11668889, 1.73162005]),
        pressure_ratio=np.array([9.37498439, 0.55183241]),
        temperature_ratio=np.array([1.19674096, 0.91310271]),
        density_ratio=np.array([1 / 0.12765258, 1 / 1.65467392]),
        entropy_ratio=np.array([1.60943791, 0.40546510]),
        total_pressure_ratio=np.array([5.0, 1.5]),
        fanno_parameter=np.array([48.2150982, 0.35551592]),
        specific_heat_ratio=np.array([1.4, 1.1]),
    )
    compare_tables(actual, expected)


def test_mach2_by_total_pressure():
    actual = fanno.lookup_table_by_total_pressure(
        np.array([2.0, 3.0]),
        specific_heat_ratio=np.array([1.3, 1.2]),
        flow_regime=np.array(
            [FlowSpeedRegime.subsonic, FlowSpeedRegime.supersonic]
        ),
    ).mach
    expected = np.array([0.30900860, 2.39713187])
    np.testing.assert_allclose(actual, expected)


def test_lookup_table_by_fanno_parameter():
    actual = fanno.lookup_table_by_fanno_parameter(
        fanno_parameter=np.array([33, 0.01]),
        specific_heat_ratio=np.array([1.4, 1.3]),
        flow_regime=np.array(
            [FlowSpeedRegime.subsonic, FlowSpeedRegime.supersonic]
        ),
    )
    expected = FannoFlowTable(
        mach=np.array([0.13904873, 1.09393234]),
        pressure_ratio=np.array([7.86294980, 0.90262819]),
        temperature_ratio=np.array([1.19537758, 0.97498675]),
        density_ratio=np.array([1 / 0.15202660, 1 / 1.08016431]),
        entropy_ratio=np.array([1.43754446, 0.00732479]),
        total_pressure_ratio=np.array([4.21034447, 1.00735168]),
        fanno_parameter=np.array([33, 0.01]),
        specific_heat_ratio=np.array([1.4, 1.3]),
    )
    compare_tables(actual, expected, rtol=1e-6)


def test_upstream_mach_by_fanno_parameter():
    actual = fanno.lookup_table_by_fanno_parameter(
        fanno_parameter=np.array([1.2, 0.4]),
        specific_heat_ratio=np.array([1.3, 1.35]),
        flow_regime=np.array(
            [FlowSpeedRegime.subsonic, FlowSpeedRegime.supersonic]
        ),
    ).mach
    expected = np.array([0.49693601, 2.23092883])
    np.testing.assert_allclose(actual, expected)

    # L1* - L2* = L
    actual = fanno.lookup_table_by_fanno_parameter(
        fanno_parameter=(
            0.2
            + fanno.fanno_parameter_by_mach(
                mach_initial=2.0, mach_final=1.0, specific_heat_ratio=1.4
            )
        ),
        specific_heat_ratio=1.4,
        flow_regime=FlowSpeedRegime.supersonic,
    ).mach
    np.testing.assert_allclose(actual, np.array([2.89064091]))


def test_temperature_ratio_by_mach():
    actual = fanno.temperature_ratio_by_mach(
        mach_initial=np.array([4.0, 1.0, 1.0]),
        mach_final=np.array([3.0, 0.5, 1.02]),
        specific_heat_ratio=np.array([1.35, 1.4, 1.4]),
    )
    expected = np.array([0.45631067 * 1 / 0.30921052, 1.14285714, 0.99331170])
    np.testing.assert_allclose(actual, expected)


def test_pressure_ratio_by_mach():
    actual = fanno.pressure_ratio_by_mach(
        mach_initial=np.array([1.0, 0.1, 1.0]),
        mach_final=np.array([5.0, 0.3, 0.9]),
        specific_heat_ratio=np.array([1.2, 1.35, 1.4]),
    )
    expected = np.array([0.11212238, 3.58512467 * 1 / 10.8302693, 1.12913286])
    np.testing.assert_allclose(actual, expected)


def test_density_ratio_by_mach():
    actual = fanno.density_ratio_by_mach(
        mach_initial=np.array([1.0, 1.0, 3.0]),
        mach_final=np.array([1.1, 0.5, 2.0]),
        specific_heat_ratio=np.array([1.6, 1.5, 1.3]),
    )
    expected = np.array(
        [1 / 1.07427738, 1 / 0.54232614, 2.09863177 / 1.69558249]
    )
    np.testing.assert_allclose(actual, expected)


def test_total_pressure_ratio_by_mach():
    np.testing.assert_allclose(
        fanno.total_pressure_ratio_by_mach(
            mach_initial=np.array([1.0, 2.5]),
            mach_final=2.0,
            specific_heat_ratio=1.4,
        ),
        np.array([1.6875, 0.64]),
    )


def test_entropy_ratio_by_mach():
    mach_initial = np.array([2.0, 0.3, 1.8])
    mach_final = np.array([1.0, 1.0, 1.5])
    specific_heat_ratio = 1.4
    actual = fanno.entropy_ratio_by_mach(
        mach_initial=mach_initial,
        mach_final=mach_final,
        specific_heat_ratio=specific_heat_ratio,
    )
    expected = np.array([0.52324814, 0.71052788, 0.20167506])
    np.testing.assert_allclose(actual, expected)


def test_fanno_parameter_by_mach():
    mach_initial = np.array([0.3, 0.47444776, 0.3, 3.0, 2.0, 3.0])
    mach_final = np.array([1.0, 1.0, 0.47444776, 1.0, 1.0, 2.0])
    specific_heat_ratio = np.array([1.4, 1.4, 1.4, 1.35, 1.35, 1.35])
    actual = fanno.fanno_parameter_by_mach(
        mach_initial=mach_initial,
        mach_final=mach_final,
        specific_heat_ratio=specific_heat_ratio,
    )
    expected = np.array(
        [5.29925, 1.29925, 4.0, 0.57108656, 0.32955389, 0.24153267]
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_duct_length():
    fanno_param = 0.5
    diameter = 1.2
    f = 0.002
    actual = fanno.duct_length(fanno_param, diameter, f)
    expected = np.array([75.0])
    np.testing.assert_allclose(actual, expected)
