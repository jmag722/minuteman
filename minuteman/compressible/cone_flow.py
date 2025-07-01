import numpy as np
from scipy.integrate import solve_ivp
import minuteman.compressible.oblique_shock as obs


def compute_vprime(m: float, gam: float):
    return (2 / ((gam-1) * m**2) + 1) ** -0.5


def compute_vtheta_shock(vprime, shock_angle, deflection_angle):
    return -np.sin(shock_angle - deflection_angle) * vprime


def compute_vr_shock(vprime, shock_angle, deflection_angle):
    return np.cos(shock_angle - deflection_angle) * vprime


def fun(theta, y, gam):
    vrprime, vthetaprime = y
    x1 = vrprime
    x2 = vthetaprime  # vtheta_prime = dvrprime/dtheta
    dx1prime = x2

    _term1 = (1-x1**2-x2**2)
    dx2prime = (
        ((gam-1)/2 * (2*x1*_term1 + x2/np.tan(theta)*_term1) - x1*x2**2) /
        (x2**2 - (gam-1)/2 * _term1))
    return dx1prime, dx2prime


def taylor_maccoll_from_cone(M1: float, cone_angle: float, gam: float = 1.4):
    """Compute the solution for a right cone at zero AoA.

    This follows the solution method to the Taylor-Maccoll analysis, as presented
    in Anderson Ch 10.

    Args:
        M1 (float): freestream Mach number [-]
        shock_angle (float): oblique shock angle [radians]
        gam (float, optional): ratio of specific heats. Defaults to 1.4.
    """
    initial_shock_guess = obs.shock_angle(
        M1=M1, theta=cone_angle, gam=gam)
    initial_shock_guess = obs.mach_angle(M=M1)
    initial_shock_guess = np.radians(34)

    print(f"Freestream Mach at gamma: {M1} at gam={gam}")

    solution = _tay_mac_cone(
        shock_angle_guess=initial_shock_guess,
        cone_angle=cone_angle,
        M1=M1,
        gam=gam)
    vr, vtheta = solution.y
    theta = solution.t
    sol_vprime = (vr**2 + vtheta**2)**0.5
    sol_mach = ((sol_vprime**-2 - 1) * (gam-1)/2) ** -0.5

    print(f"Mach at surface: {sol_mach[-1]}")
    print(f"Shock Angle: {np.degrees(theta[0])} deg")
    print(
        f"Deflection Angle: {np.degrees(obs.deflection_angle(M1=M1, beta=theta[0], gam=gam))} deg")
    print(f"Cone angle = {np.degrees(theta[-1])} deg")
    print(f"Vtheta at cone = {np.degrees(vtheta[-1])}")


def _tay_mac_cone(shock_angle_guess: float, M1: float,
                  cone_angle: float, gam: float,
                  epsilon: float = 1e-15, ntheta: int = 100):
    mn1 = obs.mach1_normal(M1=M1, beta=shock_angle_guess)
    mn2 = obs.mach2_normal(Mn1=mn1, gam=gam)
    deflection_angle = obs.deflection_angle(
        M1=M1, beta=shock_angle_guess, gam=gam)
    m2 = obs.mach2(Mn2=mn2, beta=shock_angle_guess, theta=deflection_angle)
    v_shock = compute_vprime(m=m2, gam=gam)
    vtheta_shock = compute_vtheta_shock(vprime=v_shock, shock_angle=shock_angle_guess,
                                        deflection_angle=deflection_angle)
    vr_shock = compute_vr_shock(vprime=v_shock, shock_angle=shock_angle_guess,
                                deflection_angle=deflection_angle)

    theta_arr = np.linspace(shock_angle_guess, cone_angle, ntheta)
    solution = solve_ivp(fun=fun, t_span=(
        shock_angle_guess, cone_angle), y0=(vr_shock, vtheta_shock), args=(gam,), t_eval=theta_arr)

    sol_vr, sol_vtheta = solution.y
    cone_angle_new = (cone_angle - sol_vtheta[-1] /
                      fun(theta=theta_arr[-1], y=(sol_vr[-1], sol_vtheta[-1]), gam=gam)[-1])
    # we know the actual cone angle (it's cone_angle), so adjust the new shock
    # angle by the amount needed to get vtheta(theta_c)=0
    shock_angle_new = shock_angle_guess - (cone_angle_new - cone_angle)
    if np.abs(cone_angle_new - cone_angle) < epsilon:
        return solution
    else:
        return _tay_mac_cone(shock_angle_guess=shock_angle_new, M1=M1,
                             cone_angle=cone_angle, gam=gam)


def taylor_maccoll_from_shock(M1: float, shock_angle: float, gam: float = 1.4):
    """Compute the solution for a right cone at zero AoA.

    This follows the solution method to the Taylor-Maccoll analysis, as presented
    in Anderson Ch 10.

    Args:
        M1 (float): freestream Mach number [-]
        shock_angle (float): oblique shock angle [radians]
        gam (float, optional): ratio of specific heats. Defaults to 1.4.
    """
    obs.check_shock_angle(beta=shock_angle, M=M1)
    mn1 = obs.mach1_normal(M1=M1, beta=shock_angle)
    mn2 = obs.mach2_normal(Mn1=mn1, gam=gam)
    deflection_angle = obs.deflection_angle(M1=M1, beta=shock_angle, gam=gam)
    m2 = obs.mach2(Mn2=mn2, beta=shock_angle, theta=deflection_angle)
    v_shock = compute_vprime(m=m2, gam=gam)
    vtheta_shock = compute_vtheta_shock(vprime=v_shock, shock_angle=shock_angle,
                                        deflection_angle=deflection_angle)
    vr_shock = compute_vr_shock(vprime=v_shock, shock_angle=shock_angle,
                                deflection_angle=deflection_angle)

    print(f"Freestream Mach at gamma: {M1} at gam={gam}")
    print(f"Shock Angle: {np.degrees(shock_angle)} deg")
    print(f"Deflection Angle: {np.degrees(deflection_angle)} deg")

    solution = _tay_mac(
        cone_angle_guess=deflection_angle*2,
        vr_shock=vr_shock,
        vtheta_shock=vtheta_shock, shock_angle=shock_angle,
        gam=gam)
    vr, vtheta = solution.y
    theta = solution.t
    sol_vprime = (vr**2 + vtheta**2)**0.5
    sol_mach = ((sol_vprime**-2 - 1) * (gam-1)/2) ** -0.5

    print(f"Mach at surface: {sol_mach[-1]}")
    print(f"Cone angle = {np.degrees(theta[-1])} deg")
    print(f"Vtheta at cone = {np.degrees(vtheta[-1])}")


def _tay_mac(cone_angle_guess: float, vr_shock: float, vtheta_shock: float,
             shock_angle: float, gam: float,
             epsilon: float = 1e-15, ntheta: int = 100):
    """Recursive formula to solve the Taylor-Maccoll problem

    Args:
        cone_angle_guess (float): estimate for the cone angle [radians]
        vr_shock (float): nondimensional velocity in the radial direction
            immediately behind the shock
        vtheta_shock (float): nondimensional velocity in the angular direction
            immediately behind the shock
        shock_angle (float): shock angle [radians]
        gam (float): ratio of specific heats
        epsilon (float, optional): minimum required difference in cone angle.
            Defaults to 1e-15.
        ntheta (int, optional): number of angular points in numerical integration
            from the shock to the surface. Defaults to 100.

    Raises:
        ValueError: vtheta_shock is non-negative

    Returns:
        _type_: solution of numerical integral
    """
    if vtheta_shock >= 0.0:
        raise ValueError("Angular velocity at the shock must be negative!")
    theta_arr = np.linspace(shock_angle, cone_angle_guess, ntheta)
    solution = solve_ivp(fun=fun, t_span=(
        shock_angle, cone_angle_guess), y0=(vr_shock, vtheta_shock), args=(gam,), t_eval=theta_arr)

    sol_vr, sol_vtheta = solution.y
    vrc = sol_vr[-1]
    vthetac = sol_vtheta[-1]
    cone_angle_new = (cone_angle_guess - vthetac /
                      fun(theta=theta_arr[-1], y=(vrc, vthetac), gam=gam)[-1])
    if np.abs(cone_angle_new - cone_angle_guess) < epsilon:
        return solution
    else:
        return _tay_mac(cone_angle_guess=cone_angle_new, vr_shock=vr_shock,
                        vtheta_shock=vtheta_shock, shock_angle=shock_angle, gam=gam)


if __name__ == "__main__":
    # taylor_maccoll_from_shock(M1=3.0, shock_angle=np.radians(25), gam=1.4)
    # print("\n")
    # taylor_maccoll_from_shock(M1=7.0, shock_angle=np.radians(80), gam=1.4)
    # print("\n")
    # taylor_maccoll_from_shock(M1=4.0, shock_angle=np.radians(33), gam=1.5)
    # print("\n")
    # taylor_maccoll_from_shock(M1=1.5, shock_angle=np.radians(50), gam=1.5)

    taylor_maccoll_from_cone(M1=11.0, cone_angle=np.radians(25), gam=1.6)
    print("\n")
    # taylor_maccoll_from_shock(M1=7.0, shock_angle=np.radians(80), gam=1.4)
    # print("\n")
    # taylor_maccoll_from_shock(M1=4.0, shock_angle=np.radians(33), gam=1.5)
    # print("\n")
    # taylor_maccoll_from_shock(M1=1.5, shock_angle=np.radians(50), gam=1.5)
