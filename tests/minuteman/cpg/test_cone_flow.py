import numpy as np
import pytest

import minuteman.cpg.cone_flow as cf

"""
Unit tests taken from the following sources

- Sims, Joseph L. "Tables for supersonic flow around right circular cones at
    zero angle of attack." (1964).

- Fluid Dynamics Flow Calculators - Prof. S. A. E. Miller:
    https://saemiller.com/flow/SAEMiller_Comp_Calc.html

- Compressible Aerodynamics Calculator v3.2:
    https://devenport.aoe.vt.edu/aoe3114/calc.html

I get better agreement with SAEMiller calculator than VT calculator.
"""


@pytest.mark.parametrize(
    "M1, shock_angle, gam, theta_c, mach_c, rtol_theta, rtol_mach",
    [
        # VT
        (1.5, 42.0, 1.4, 6.67715193, 1.43307833, 1e-3, None),
        (5, 20.0, 1.3, 15.269259373069906, 4.002902231709606, None, None),
        # SAEMiller
        (1.5, 42.0, 1.4, 6.680016719152055, 1.4330783951611243, 1e-5, None),
        # Sims
        (20.0, np.degrees(0.06734690), 1.4, 2.5, 17.637850, 1e-5, 1e-5),
        (15, np.degrees(0.58391003), 1.4, 30.0, 3.3037471, None, None),
        (1.75, np.degrees(0.87135164), 1.4, 27.5, 1.1495789, 1e-5, None),
        (7.0, np.degrees(0.23632593), 1.4, 10, 5.6318010, None, None),
        (2.0, 31, 1.5, 9.274242549862922, 1.83629035381199, 1e-5, None),
    ],
)
def test_solve_taylor_maccoll_by_shock_angle(
    M1, shock_angle, gam, theta_c, mach_c, rtol_theta, rtol_mach
):
    rtol_theta = rtol_theta if rtol_theta is not None else 5e-6
    rtol_mach = rtol_mach if rtol_mach is not None else 5e-6
    theta, vr, vtheta = cf.solve_taylor_maccoll_by_shock_angle(
        mach_upstream=M1,
        shock_angle=np.radians(shock_angle),
        specific_heat_ratio=gam,
    )
    theta = np.degrees(theta)
    v = cf.nondimensional_velocity_from_components(vr, vtheta)
    mach = cf.mach_from_nondimensional_velocity(v, specific_heat_ratio=gam)
    assert theta[-1] == pytest.approx(theta_c, rel=rtol_theta)
    assert mach[-1] == pytest.approx(mach_c, rel=rtol_mach)


@pytest.mark.parametrize(
    "M1, cone_angle, gam, theta_shock, mach_c, rtol_theta, rtol_mach",
    [
        # SAEMiller
        (5.0, 10.0, 1.4, 15.608275334274234, 4.292164349604961, None, None),
        (19.0, 1.0, 1.25, 3.0892857358022963, 18.60504230165579, None, None),
        (3.5, 11.0, 1.5, 20.190166783821734, 3.0152597231607237, None, None),
        (10.0, 10.0, 1.3, 12.104996198134035, 7.686454681916647, None, None),
        # Sims
        (10.0, 2.5, 1.4, np.degrees(0.10660326), 9.5046400, None, 5e-5),
        (10.0, 20.0, 1.4, np.degrees(0.39698293), 4.6723717, None, 1e-5),
        (20.0, 30.0, 1.4, np.degrees(0.58182628), 3.3673365, 1e-5, 1e-5),
        (1.5, 25.0, 1.4, np.degrees(0.95578504), 1.0100703, 1e-5, 5e-5),
    ],
)
def test_solve_taylor_maccoll_by_cone_angle(
    M1, cone_angle, gam, theta_shock, mach_c, rtol_theta, rtol_mach
):
    theta, vr, vtheta = cf.solve_taylor_maccoll_by_cone_angle(
        cone_angle=np.radians(cone_angle),
        mach_upstream=M1,
        specific_heat_ratio=gam,
        shock_type=cf.ObliqueShockType.weak,
    )
    theta = np.degrees(theta)
    v = cf.nondimensional_velocity_from_components(vr, vtheta)
    mach = cf.mach_from_nondimensional_velocity(v, specific_heat_ratio=gam)
    assert theta[0] == pytest.approx(theta_shock, rel=rtol_theta)
    assert mach[-1] == pytest.approx(mach_c, rel=rtol_mach)


@pytest.mark.parametrize(
    "m1, cone_mach, gam, theta_shock, theta_cone, rtol_shock, rtol_cone",
    [
        # VT
        (5.0, 3.0, 1.4, 29.3616760, 24.2337037, 1e-3, 1e-3),
        (15.0, 5.0, 1.3, 25.8012147, 23.6526230, 1e-3, 1e-3),
        (15.0, 1.4, 1.3, 61.5824290, 54.3345129, 1e-3, 1e-3),
        # Sims
        (20.0, 7.0145241, 1.4, np.degrees(0.29125730), 15.0, 1e-5, 1e-5),
        (10.0, 3.1391940, 1.4, np.degrees(0.58984179), 30.0, 1e-5, 1e-5),
        (3.0, 2.9627600, 1.4, np.degrees(0.34011057), 2.5, 1e-5, 1e-3),
    ],
)
def test_solve_taylor_maccoll_by_surface_mach(
    m1, cone_mach, gam, theta_shock, theta_cone, rtol_shock, rtol_cone
):
    theta, vr, vtheta = cf.solve_taylor_maccoll_by_surface_mach(
        surface_mach=cone_mach, mach_upstream=m1, specific_heat_ratio=gam
    )
    theta = np.degrees(theta)
    v = cf.nondimensional_velocity_from_components(vr, vtheta)
    _ = cf.mach_from_nondimensional_velocity(v, specific_heat_ratio=gam)
    assert theta[0] == pytest.approx(theta_shock, rel=rtol_shock)
    assert theta[-1] == pytest.approx(theta_cone, rel=rtol_cone)
