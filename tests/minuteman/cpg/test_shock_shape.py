# Copyright (c) 2022-2026 Jared Magnusson
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from minuteman.cpg import shock_shape
from minuteman.utils.bounds_check import OutOfBoundsError


def compare_soln(
    actual: shock_shape.ShockShapeSolution,
    expected: shock_shape.ShockShapeSolution,
    **kwargs,
):
    np.testing.assert_allclose(actual.x, expected.x, **kwargs)
    np.testing.assert_equal(actual.y, expected.y)
    assert actual.mach == expected.mach
    assert actual.nose_radius == expected.nose_radius
    assert actual.shock_angle == pytest.approx(
        expected.shock_angle, abs=kwargs["atol"]
    )
    assert actual.half_angle == expected.half_angle
    assert actual.standoff_distance == pytest.approx(
        expected.standoff_distance, abs=kwargs["atol"]
    )
    assert actual.curvature_radius > 0  # no validation data yet
    assert actual.body_type == expected.body_type


def test_estimate_coords_cylinder_wedge():
    """This test comes from Figure 5.36 (Billig correlation) in
    Anderson, J. D. (1989). Hypersonic and high temperature gas
    dynamics. AIAA.
    """
    mach = 8.0
    nose_radius = 1.0
    wedge_angle = np.atan(1.0 / 4.0)
    # note that unlike Billig, Anderson defines x as positive downstream
    # as we do
    x = np.array(
        [
            -1.397570995,
            -1.343488757,
            -1.274225236,
            -1.180930927,
            -1.013758217,
            -0.820405718,
            -0.498821794,
        ]
    )
    # change x based on API convention
    # (zero at nose tip, not center of curvature)
    x = x + nose_radius
    y = np.array(
        [
            0.003572066,
            0.509779972,
            0.760406524,
            1.01405781,
            1.305538431,
            1.612062062,
            2.005692261,
        ]
    )
    actual = shock_shape.estimate_coords_cylinder_wedge(
        y=y,
        mach=mach,
        nose_radius=nose_radius,
        half_angle=wedge_angle,
    )
    expected = shock_shape.ShockShapeSolution(
        x=x,
        y=y,
        mach=mach,
        nose_radius=nose_radius,
        # shock angle computed from VTT calculator
        shock_angle=np.radians(19.7950834),
        half_angle=wedge_angle,
        standoff_distance=-x[0],
        # curvature radius unknown
        curvature_radius=-1,
        body_type=shock_shape.BluntBodyType.cylinder_wedge,
    )
    compare_soln(actual, expected, atol=1.8e-2)


@pytest.mark.parametrize(
    # note that unlike Billig, Anderson defines x as positive downstream
    # as we do
    ("mach", "nose_radius", "x", "y", "shock_angle"),
    [
        # Mach 8 case
        (
            8.0,
            1.0,
            np.array(
                [
                    -1.129762712,
                    -1.133083733,
                    -1.114112454,
                    -1.07879991,
                    -1.023088005,
                    -0.926359608,
                    -0.853762009,
                    -0.744695999,
                    -0.617232455,
                    -0.482946788,
                    -0.328098433,
                ]
            ),
            np.array(
                [
                    0.008794633,
                    0.13294972,
                    0.269869588,
                    0.411262155,
                    0.559263261,
                    0.712340126,
                    0.844460638,
                    0.997839033,
                    1.155740382,
                    1.289368545,
                    1.423499259,
                ]
            ),
            np.radians(16.9840636),
        ),
        # Mach 4 case from figure
        (
            4.0,
            1.0,
            np.array(
                [
                    -1.170942533,
                    -1.172261728,
                    -1.142954662,
                    -1.117869019,
                    -1.072547344,
                    -0.988047675,
                    -0.909607927,
                    -0.80438224,
                    -0.697372499,
                    -0.567471585,
                    -0.424960843,
                ]
            ),
            np.array(
                [
                    0.009824861,
                    0.136065533,
                    0.271201347,
                    0.41030731,
                    0.562127799,
                    0.710832475,
                    0.855315727,
                    0.998416966,
                    1.151745106,
                    1.295449406,
                    1.429278589,
                ]
            ),
            np.radians(20.9863127),
        ),
    ],
)
def test_estimate_coords_sphere_cone(mach, nose_radius, x, y, shock_angle):
    """This test comes from Figure 5.35 (Billig correlation) in
    Anderson, J. D. (1989). Hypersonic and high temperature gas
    dynamics. AIAA.
    """
    cone_angle = np.atan(1.0 / 4.0)
    # change x based on API convention
    # (zero at nose tip, not center of curvature)
    x = x + nose_radius
    actual = shock_shape.estimate_coords_sphere_cone(
        y=y,
        mach=mach,
        nose_radius=nose_radius,
        half_angle=cone_angle,
    )
    expected = shock_shape.ShockShapeSolution(
        x=x,
        y=y,
        mach=mach,
        nose_radius=nose_radius,
        # shock angle computed from VTT calculator
        shock_angle=shock_angle,
        half_angle=cone_angle,
        standoff_distance=-x[0],
        # curvature radius unknown
        curvature_radius=-1,
        body_type=shock_shape.BluntBodyType.sphere_cone,
    )
    compare_soln(actual, expected, atol=2.1e-2)


@pytest.mark.parametrize(
    ("y", "mach", "r", "theta"),
    [
        # negative y
        ([-1], 5, 1, 0.3),
        # Mach too low
        ([0.0], 1, 1, 0.3),
        # zero radius
        ([0.0], 2, 0, 0.3),
        # 90 deg cone
        ([0.0], 2, 0, np.pi / 2),
        # negative angle
        ([0.0], 2, 0, -np.pi / 6),
    ],
)
def test_out_of_bounds_surface_mach(y, mach, r, theta):
    with pytest.raises(OutOfBoundsError):
        shock_shape.estimate_coords_sphere_cone(y, mach, r, theta)
    with pytest.raises(OutOfBoundsError):
        shock_shape.estimate_coords_cylinder_wedge(y, mach, r, theta)
