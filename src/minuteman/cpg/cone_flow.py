import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import minuteman.cpg.oblique_shock as obs


def nondimensional_velocity_from_mach(M: float, gam: float):
    """Compute the nondimensional velocity useful to nondimensionalizing the
    Taylor Maccoll equations.

    V'=V/Vmax from Anderson chapter 10, where Vmax is a a max theoretical velocity
    if the flow expanded to 0 K (ho=h+V^2/2=Vmax^2/2). It is a function of Mach number only.

    Args:
        M (float): Mach number [-]
        gam (float): ratio of specific heats [-]

    Returns:
        float: nondimensional velocity V'
    """
    return (2.0 / ((gam - 1.0) * M**2) + 1.0) ** -0.5


def nondimensional_velocity_from_components(velocity_radial: float, velocity_azimuth: float):
    """Compute the nondimensional velocity V' useful to nondimensionalizing the
    Taylor Maccoll equations from its radial and azimuthal components,
    V_r' and V_theta'.

    Args:
        velocity_radial (float): nondimensional radial velocity [-]
        velocity_azimuth (float): nondimensional azimuth velocity [-]

    Returns:
        float: nondimensional velocity V' [-]
    """
    return np.linalg.norm((velocity_radial, velocity_azimuth), axis=0)


def mach_from_nondimensional_velocity(velocity: float, gam: float):
    """Compute Mach number from nondimensional velocity

    Args:
        velocity (float): nondimensional velocity, V' [-]
        gam (float): ratio of specific heats [-]

    Returns:
        float: Mach number
    """
    return (0.5 * (gam-1) * (velocity ** -2 - 1.0)) ** -0.5


def nondimensional_velocity_azimuth(velocity_nondim, shock_angle, deflection_angle):
    """Compute the nondimensional azimuthal velocity behind a conic flow.

    This is V_theta' from Anderson chapter 10. The quantity is negative, as the
    positive V_theta axis in the coordinate system is positive away from the
    body.

    Args:
        velocity_nondim (Any): nondimensional velocity, V'
        shock_angle (Any): shock angle [radians]
        deflection_angle (Any): deflection angle [radians]

    Returns:
        Any: azimuth component of nondimensional velocity
    """
    return -np.sin(shock_angle - deflection_angle) * velocity_nondim


def nondimensional_velocity_radial(velocity_nondim, shock_angle, deflection_angle):
    """Compute the nondimensional radial velocity behind a conic flow.

    This is V_r' from Anderson chapter 10. The quantity is positive in the
    downstream direction

    Args:
        velocity_nondim (Any): nondimensional velocity, V'
        shock_angle (Any): shock angle [radians]
        deflection_angle (Any): deflection angle [radians]

    Returns:
        Any: radial component of nondimensional velocity
    """
    return np.cos(shock_angle - deflection_angle) * velocity_nondim


def taylor_maccoll_odes(azimuth_angles: np.ndarray, nondim_velocity_components: tuple,
                        gam: float):
    """Return the Taylor-Maccoll 2nd-order ODE as a system of first-order ODE,
    evaluated for a set of azimuth angles and nondimensional velocities.

    Let y1 = V_r',

    y2 = dV_r'/dtheta = V_theta'

    Then,

    dy1/dtheta = dV_r'/dtheta = y2

    dy2/dtheta = d^2V_r'/dtheta^2

    Args:
        azimuth_angles (np.ndarray): azimuth angles at which to evaluate
            ODE [radians]
        nondim_velocity_components (tuple): radial, then azimuthal velocity components [-]
        gam (float): ratio of specific heats [-]

    Returns:
        tuple: (dy1/dtheta, dy2/dtheta)
    """
    y1, y2 = nondim_velocity_components
    dy1_dtheta = y2  # irrotational condition

    gamma_term = 0.5 * (gam - 1)
    # nondim. speed of sound squared, a^2
    a2 = gamma_term * (1.0 - y1**2 - y2**2)
    dy2_dtheta = (y1 * y2**2 - a2 * (2 * y1 + y2 / np.tan(azimuth_angles))
                  ) / (a2 - y2**2)
    return dy1_dtheta, dy2_dtheta


def taylor_maccoll_from_cone(M1: float, cone_angle: float, gam: float = 1.4):
    """Compute the solution for a right cone at zero AoA.

    This follows the solution method to the Taylor-Maccoll analysis, as presented
    in Anderson Ch 10.

    Args:
        M1 (float): freestream Mach number [-]
        shock_angle (float): oblique shock angle [radians]
        gam (float, optional): ratio of specific heats. Defaults to 1.4.

    Returns:
        tuple: see taylor_maccoll_from_shock return
    """
    max_defl_ang = obs.max_deflection_angle(M1=M1, gam=gam)
    if cone_angle >= max_defl_ang:
        max_angle = obs.sonic_shock_angle(M1=M1, gam=gam)
    else:
        max_angle = obs.shock_angle(
            M1=M1, theta=cone_angle, gam=gam)

    def func(shock_angle, M1, cone_angle, gam):
        theta, _, _ = taylor_maccoll_from_shock(
            M1=M1, shock_angle=shock_angle, gam=gam)
        return theta[-1] - cone_angle
    # multiply by 1+small decimal to avoid numerical issues with edge case
    actual_shock_angle = brentq(
        func, a=obs.mach_angle(M=M1)*1.00001, b=max_angle,
        args=(M1, cone_angle, gam))
    return taylor_maccoll_from_shock(
        M1=M1, shock_angle=actual_shock_angle, gam=gam)


def taylor_maccoll_from_surface_mach(M1: float, cone_mach: float, gam: float = 1.4):
    """_summary_

    Args:
        M1 (float): freestream Mach number [-]
        cone_mach (float): Mach number at the cone surface [-]
        gam (float, optional): ratio of specific heats. Defaults to 1.4.

    Returns:
        tuple: see taylor_maccoll_from_shock return
    """
    if cone_mach > M1:
        raise ValueError("Surface mach number should be less than freestream")
    max_angle = obs.sonic_shock_angle(M1=M1, gam=gam)

    def func(shock_angle, M1, cone_mach, gam):
        _, vr, vtheta = taylor_maccoll_from_shock(
            M1=M1, shock_angle=shock_angle, gam=gam)
        vprime = nondimensional_velocity_from_components(vr, vtheta)
        mach = mach_from_nondimensional_velocity(vprime, gam=gam)
        return mach[-1] - cone_mach
    actual_shock_angle = brentq(
        func, a=obs.mach_angle(M=M1)*1.00001, b=max_angle,
        args=(M1, cone_mach, gam))
    return taylor_maccoll_from_shock(
        M1=M1, shock_angle=actual_shock_angle, gam=gam)


def taylor_maccoll_from_shock(M1: float, shock_angle: float, gam: float = 1.4):
    """Compute the solution for a right cone at zero AoA.

    This follows the solution method to the Taylor-Maccoll analysis, as presented
    in Anderson Ch 10.

    Args:
        M1 (float): freestream Mach number [-]
        shock_angle (float): oblique shock angle [radians]
        gam (float, optional): ratio of specific heats. Defaults to 1.4.

    Returns:
        tuple: theta, radial velocity, azimuthal velocity

    """
    obs.check_shock_angle(beta=shock_angle, M=M1)
    mn1 = obs.mach1_normal(M1=M1, beta=shock_angle)
    mn2 = obs.mach2_normal(Mn1=mn1, gam=gam)
    deflection_angle = obs.deflection_angle(M1=M1, beta=shock_angle, gam=gam)
    m2 = obs.mach2(Mn2=mn2, beta=shock_angle, theta=deflection_angle)
    v_shock = nondimensional_velocity_from_mach(M=m2, gam=gam)
    vr_shock = nondimensional_velocity_radial(velocity_nondim=v_shock,
                                              shock_angle=shock_angle,
                                              deflection_angle=deflection_angle)
    vtheta_shock = nondimensional_velocity_azimuth(velocity_nondim=v_shock,
                                                   shock_angle=shock_angle,
                                                   deflection_angle=deflection_angle)
    if vtheta_shock >= 0.0:
        raise ValueError("Angular velocity at the shock must be negative!")

    def vtheta_zero_event(t, y, gam: float):
        return y[1]
    vtheta_zero_event.terminal = True
    vtheta_zero_event.direction = 1.0
    solution = solve_ivp(fun=taylor_maccoll_odes, t_span=(shock_angle, 0.0),
                         y0=(vr_shock, vtheta_shock), args=(gam,),
                         events=vtheta_zero_event, atol=1e-18, rtol=1e-13)

    if not solution.success:
        raise AssertionError("RK integration step failed, check inputs.")

    vr, vtheta = solution.y
    theta = solution.t
    return theta, vr, vtheta
