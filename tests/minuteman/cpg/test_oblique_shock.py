import numpy as np
import pytest

import minuteman.cpg.oblique_shock as oblique_shock
from minuteman.cpg.oblique_shock import ObliqueShockTable, ObliqueShockType


def compare_tables(
    actual: ObliqueShockTable, expected: ObliqueShockTable, **kwargs
):
    np.testing.assert_allclose(
        actual.mach_upstream, expected.mach_upstream, **kwargs
    )
    np.testing.assert_allclose(
        actual.mach_downstream, expected.mach_downstream, **kwargs
    )
    np.testing.assert_allclose(
        actual.mach_upstream_normal, expected.mach_upstream_normal, **kwargs
    )
    np.testing.assert_allclose(
        actual.mach_downstream_normal,
        expected.mach_downstream_normal,
        **kwargs,
    )
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
        actual.deflection_angle, expected.deflection_angle, **kwargs
    )
    np.testing.assert_allclose(
        actual.total_pressure_ratio, expected.total_pressure_ratio, **kwargs
    )
    np.testing.assert_allclose(
        actual.shock_angle,
        expected.shock_angle,
        **kwargs,
    )
    np.testing.assert_allclose(
        actual.specific_heat_ratio, expected.specific_heat_ratio, **kwargs
    )


def test_lookup_table_by_deflection_angle():
    actual = oblique_shock.lookup_table_by_deflection_angle(
        deflection_angle=np.radians(20),
        mach_upstream=5.0,
        specific_heat_ratio=1.3,
        shock_type=np.array([ObliqueShockType.weak, ObliqueShockType.strong]),
    )
    expected = ObliqueShockTable(
        mach_upstream=np.array([5.0]),
        mach_downstream=np.array([3.30604887, 0.42056788]),
        mach_upstream_normal=np.array([2.40552416, 4.98615922]),
        mach_downstream_normal=np.array([0.50335993, 0.38341514]),
        deflection_angle=np.radians([20.0]),
        shock_angle=np.radians([28.7575852, 85.7358435]),
        temperature_ratio=np.array([1.79958742, 4.62723212]),
        pressure_ratio=np.array([6.41087864, 27.9741903]),
        density_ratio=np.array([3.56241578, 6.04555587]),
        total_pressure_ratio=np.array([0.50253707, 0.03661833]),
        specific_heat_ratio=np.array([1.3]),
    )
    compare_tables(actual, expected, rtol=1e-6)


def test_lookup_table_by_shock_angle():
    actual = oblique_shock.lookup_table_by_shock_angle(
        shock_angle=np.radians(15.0),
        mach_upstream=6.7,
        specific_heat_ratio=1.35,
    )
    expected = ObliqueShockTable(
        mach_upstream=np.array([6.7]),
        mach_downstream=np.array([5.45176471]),
        mach_upstream_normal=np.array([1.73408760]),
        mach_downstream_normal=np.array([0.62681804]),
        deflection_angle=np.radians([8.39780219]),
        shock_angle=np.radians([15.0]),
        temperature_ratio=np.array([1.42804635]),
        pressure_ratio=np.array([3.30598361]),
        density_ratio=np.array([2.31503942]),
        total_pressure_ratio=np.array([0.83644525]),
        specific_heat_ratio=np.array([1.35]),
    )
    compare_tables(actual, expected, rtol=1e-6)


def test_lookup_table_by_mach_upstream_normal():
    actual = oblique_shock.lookup_table_by_mach_upstream_normal(
        mach_upstream_normal=3.0,
        mach_upstream=5.0,
        specific_heat_ratio=1.3,
    )
    expected = ObliqueShockTable(
        mach_upstream=np.array([5.0]),
        mach_downstream=np.array([2.68696219]),
        mach_upstream_normal=np.array([3.0]),
        mach_downstream_normal=np.array([0.45106895]),
        deflection_angle=np.radians([27.2057121]),
        shock_angle=np.radians([36.8698976]),
        temperature_ratio=np.array([2.28040327]),
        pressure_ratio=np.array([10.0434782]),
        density_ratio=np.array([4.40425531]),
        total_pressure_ratio=np.array([0.28216315]),
        specific_heat_ratio=np.array([1.3]),
    )
    compare_tables(actual, expected, rtol=1e-6)


@pytest.mark.parametrize(
    "m1, beta, gam, expected",
    [
        # expected values from https://devenport.aoe.vt.edu/aoe3114/calc.html
        (2.5, np.radians(65.0), 1.35, 0.53010844),
        (30, np.radians(18.0), 1.67, 0.45612707),
    ],
)
def test_mach_upstream_normal_component(m1, beta, gam, expected):
    mn1 = oblique_shock.mach_upstream_normal_component(
        mach_upstream=m1, shock_angle=beta
    )
    actual = oblique_shock.mach_downstream_normal_component(
        mach_upstream_normal=mn1, specific_heat_ratio=gam
    )
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize(
    "m1, beta, gam, expected",
    [
        # expected values from https://devenport.aoe.vt.edu/aoe3114/calc.html
        (3.3, np.radians(35), 1.4, 2.21741000),
        (6.3, np.radians(87), 1.2, 0.35603772),
    ],
)
def test_mach_downstream_by_postshock(m1, beta, gam, expected):
    mn1 = oblique_shock.mach_upstream_normal_component(
        mach_upstream=m1, shock_angle=beta
    )
    mn2 = oblique_shock.mach_downstream_normal_component(
        mach_upstream_normal=mn1, specific_heat_ratio=gam
    )
    theta = oblique_shock.deflection_angle_by_shock_mach(
        mach_upstream=m1, shock_angle=beta, specific_heat_ratio=gam
    )
    actual = oblique_shock.mach_downstream_by_postshock(
        mach_downstream_normal=mn2, shock_angle=beta, deflection_angle=theta
    )
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize(
    "m1, beta, gam, expected",
    [
        # expected values from https://devenport.aoe.vt.edu/aoe3114/calc.html
        (2.0, np.radians(65.0), 1.4, np.radians(22.9704761)),
        (6.0, np.radians(15.0), 1.2, np.radians(7.85416103)),
    ],
)
def test_deflection_angle(m1, beta, gam, expected):
    assert oblique_shock.deflection_angle_by_shock_mach(
        mach_upstream=m1, shock_angle=beta, specific_heat_ratio=gam
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    "m1, theta, shock_type, gam, expected",
    [
        # expected values from https://devenport.aoe.vt.edu/aoe3114/calc.html
        (
            6.0,
            np.radians(15.0),
            ObliqueShockType.weak,
            1.2,
            np.radians(21.2816779),
        ),
        (
            1.2,
            np.radians(3.0),
            ObliqueShockType.weak,
            1.45,
            np.radians(64.6226700),
        ),
        (
            3,
            np.radians(15.0),
            ObliqueShockType.strong,
            1.45,
            np.radians(83.9497031),
        ),
        (
            1.1,
            np.radians(5.0),
            ObliqueShockType.strong,
            1.5,
            oblique_shock.InvalidDeflectionAngle,
        ),
        (
            4,
            np.radians(89.0),
            ObliqueShockType.weak,
            1.3,
            oblique_shock.InvalidDeflectionAngle,
        ),
    ],
)
def test_shock_angle(m1, theta, shock_type, gam, expected):
    if isinstance(expected, float):
        assert oblique_shock.shock_angle_by_deflection_mach(
            mach_upstream=m1,
            deflection_angle=theta,
            shock_type=shock_type,
            specific_heat_ratio=gam,
        ) == pytest.approx(expected)
    else:
        with pytest.raises(expected):
            oblique_shock.shock_angle_by_deflection_mach(
                mach_upstream=m1,
                deflection_angle=theta,
                shock_type=shock_type,
                specific_heat_ratio=gam,
            )


@pytest.mark.parametrize(
    "theta, m1, gam",
    [
        (30.0, 3.0, 1.4),
        (20.0, 2.0, 1.4),
        (0.0, 15.0, 1.4),
        (10.0, 25.0, 1.5),
        (25.0, 10.0, 1.3),
    ],
)
def test_check_deflection_angle(theta, m1, gam):
    theta = np.radians(theta)
    oblique_shock.check_deflection_angle(theta, m1, gam)
    with pytest.raises(oblique_shock.InvalidDeflectionAngle):
        oblique_shock.check_deflection_angle(np.radians(-5), 1.05, 1.3)
    with pytest.raises(oblique_shock.InvalidDeflectionAngle):
        oblique_shock.check_deflection_angle(np.radians(90.1), 2, 1.4)


@pytest.mark.parametrize("beta, m", [(45, 2), (35, 4), (65, 1.2)])
def test_check_shock_angle(beta, m):
    beta = np.radians(beta)
    oblique_shock.check_shock_angle(shock_angle=beta, mach=m)
    with pytest.raises(oblique_shock.InvalidShockAngle):
        oblique_shock.check_shock_angle(np.radians(5), mach=2)
    with pytest.raises(oblique_shock.InvalidShockAngle):
        oblique_shock.check_shock_angle(np.radians(90.1), mach=1.1)


@pytest.mark.parametrize(
    "m1, gam", [(3, 1.35), (11, 1.45), (1.3, 1.6), (5, 1.4), (25, 1.25)]
)
def test_max_shock_deflection_angle(m1, gam):
    actual = oblique_shock.shock_angle_max(
        mach_upstream=m1, specific_heat_ratio=gam
    )

    # check that max_shock_angle agrees with shock_angle prediction
    #  (to numerical precision)
    theta_max = oblique_shock.deflection_angle_max(
        mach_upstream=m1, specific_heat_ratio=gam
    )
    expected = oblique_shock.shock_angle_by_deflection_mach(
        mach_upstream=m1,
        deflection_angle=theta_max - 1e-16,
        specific_heat_ratio=gam,
    )
    assert actual == pytest.approx(expected)

    # check that shock_angle errors out just above max
    with pytest.raises(oblique_shock.InvalidDeflectionAngle):
        oblique_shock.shock_angle_by_deflection_mach(
            mach_upstream=m1,
            deflection_angle=theta_max + 1e-10,
            specific_heat_ratio=gam,
        )


@pytest.mark.parametrize(
    "m1, gam, expected",
    [
        # expected values from https://www.pdas.com/flowcalc.html
        (3, 1.4, 63.76658),
        (11, 1.45, None),
        (1.3, 1.6, None),
        (5, 1.4, 66.08399),
        (25, 1.25, None),
    ],
)
def test_shock_angle_sonic(m1, gam, expected):
    beta = oblique_shock.shock_angle_sonic(
        mach_upstream=m1, specific_heat_ratio=gam
    )
    if expected is not None:
        expected = expected * np.pi / 180
        assert beta == pytest.approx(expected)
    # in case online calculator wrong or not specific_heat_ratio=1.4,
    # check downstream Mach
    mn1 = oblique_shock.mach_upstream_normal_component(
        mach_upstream=m1, shock_angle=beta
    )
    mn2 = oblique_shock.mach_downstream_normal_component(
        mach_upstream_normal=mn1, specific_heat_ratio=gam
    )
    theta = oblique_shock.deflection_angle_by_shock_mach(
        mach_upstream=m1, shock_angle=beta, specific_heat_ratio=gam
    )
    m2 = oblique_shock.mach_downstream_by_postshock(
        mach_downstream_normal=mn2, shock_angle=beta, deflection_angle=theta
    )
    assert m2 == pytest.approx(1.0)
