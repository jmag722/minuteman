import numpy as np
import pytest
import minuteman.compressible.cone_flow as cf


@pytest.mark.parametrize("M1, shock_angle, gam, theta_c, mach_c, rtol_theta, rtol_mach", [
    (1.5, 42.0, 1.4, 6.67715193, 1.43307833, 1e-3, None),  # VT
    (1.5, 42.0, 1.4, 6.680016719152055, 1.4330783951611243, 1e-5, None),  # SAEMiller
    (5, 20.0, 1.3, 15.269259373069906, 4.002902231709606, None, None),  # VT
    (20.0, np.degrees(0.06734690), 1.4, 2.5, 17.637850, 1e-5, 1e-5),  # Sims
    (15, np.degrees(0.58391003), 1.4, 30.0, 3.3037471, None, None),  # Sims
    (1.75, np.degrees(0.87135164), 1.4, 27.5, 1.1495789, 1e-5, None),  # Sims
    (7.0, np.degrees(0.23632593), 1.4, 10, 5.6318010, None, None),  # Sims
    (2.0, 31, 1.5, 9.274242549862922, 1.83629035381199,
     1e-5, None),  # SAEMiller
])
def test_taylor_maccoll_from_shock(M1, shock_angle, gam, theta_c, mach_c,
                                   rtol_theta, rtol_mach):
    """Check Taylor-Maccoll relations with input shock angle.

    Unit tests taken from the following sources:

    Sims, Joseph L. "Tables for supersonic flow around right circular cones at
        zero angle of attack." (1964).
    Fluid Dynamics Flow Calculators - Prof. S. A. E. Miller:
        https://saemiller.com/flow/SAEMiller_Comp_Calc.html
    Compressible Aerodynamics Calculator v3.2:
        https://devenport.aoe.vt.edu/aoe3114/calc.html

    I get better agreement with SAEMiller calculator than VT calculator.
    """
    rtol_theta = rtol_theta if rtol_theta is not None else 5e-6
    rtol_mach = rtol_mach if rtol_mach is not None else 5e-6
    theta, vr, vtheta = cf.taylor_maccoll_from_shock(
        M1=M1, shock_angle=np.radians(shock_angle), gam=gam)
    theta = np.degrees(theta)
    v = cf.nondimensional_velocity_from_components(vr, vtheta)
    mach = cf.mach_from_nondimensional_velocity(v, gam=gam)
    assert theta[-1] == pytest.approx(theta_c, rel=rtol_theta)
    assert mach[-1] == pytest.approx(mach_c, rel=rtol_mach)


@pytest.mark.parametrize("M1, cone_angle, gam, theta_shock, mach_c, rtol_theta, rtol_mach", [
    (5.0, 10.0, 1.4, 15.608275334274234, 4.292164349604961, None, None),  # SAEMiller
    (10.0, 2.5, 1.4, np.degrees(.10660326), 9.5046400, None, None),  # Sims
    (10.0, 20.0, 1.4, np.degrees(.39698293), 4.6723717, None, 1e-5),  # Sims
])
def test_taylor_maccoll_from_cone(M1, cone_angle, gam, theta_shock, mach_c,
                                  rtol_theta, rtol_mach):
    """Check Taylor-Maccoll relations with input cone angle.

    Unit tests taken from the following sources:

    Sims, Joseph L. "Tables for supersonic flow around right circular cones at
        zero angle of attack." (1964).
    Fluid Dynamics Flow Calculators - Prof. S. A. E. Miller:
        https://saemiller.com/flow/SAEMiller_Comp_Calc.html
    Compressible Aerodynamics Calculator v3.2:
        https://devenport.aoe.vt.edu/aoe3114/calc.html

    I get better agreement with SAEMiller calculator than VT calculator.
    """
    theta, vr, vtheta = cf.taylor_maccoll_from_cone(
        M1=M1, cone_angle=np.radians(cone_angle), gam=gam)
    theta = np.degrees(theta)
    v = cf.nondimensional_velocity_from_components(vr, vtheta)
    mach = cf.mach_from_nondimensional_velocity(v, gam=gam)
    assert theta[0] == pytest.approx(theta_shock, rel=rtol_theta)
    assert mach[-1] == pytest.approx(mach_c, rel=rtol_mach)
