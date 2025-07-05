import numpy as np
from scipy.integrate import solve_ivp
import minuteman.compressible.oblique_shock as obs


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
    """
    initial_shock_guess = obs.shock_angle(
        M1=M1, theta=cone_angle, gam=gam)

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
                  cone_angle: float, gam: float, x1=None, x2=None, fx1=None, fx2=None,
                  epsilon: float = 1e-15, ntheta: int = 100):
    mn1 = obs.mach1_normal(M1=M1, beta=shock_angle_guess)
    mn2 = obs.mach2_normal(Mn1=mn1, gam=gam)
    deflection_angle = obs.deflection_angle(
        M1=M1, beta=shock_angle_guess, gam=gam)
    m2 = obs.mach2(Mn2=mn2, beta=shock_angle_guess, theta=deflection_angle)
    v_shock = nondimensional_velocity_from_mach(M=m2, gam=gam)
    vtheta_shock = nondimensional_velocity_azimuth(velocity_nondim=v_shock, shock_angle=shock_angle_guess,
                                                   deflection_angle=deflection_angle)
    vr_shock = nondimensional_velocity_radial(velocity_nondim=v_shock, shock_angle=shock_angle_guess,
                                              deflection_angle=deflection_angle)

    theta_arr = np.linspace(shock_angle_guess, cone_angle, ntheta)
    solution = solve_ivp(fun=taylor_maccoll_odes, t_span=(
        shock_angle_guess, cone_angle), y0=(vr_shock, vtheta_shock), args=(gam,), t_eval=theta_arr)

    v_r, v_theta = solution.y

    if fx1 is None or fx2 is None:
        cone_angle_new = (cone_angle - v_theta[-1] /
                          taylor_maccoll_odes(azimuth_angles=theta_arr[-1],
                                              nondim_velocity_components=(
                                              v_r[-1], v_theta[-1]),
                                              gam=gam)[-1])
        # we know the actual cone angle (it's cone_angle), so adjust the new shock
        # angle by the amount needed to get vtheta(theta_c)=0
        shock_angle_new = shock_angle_guess - (cone_angle_new - cone_angle)
        shock_angle_new = np.maximum(shock_angle_new, obs.mach_angle(M=M1))
    else:
        shock_angle_new = x1 - fx1 * (x1 - x2) / (fx1 - fx2)
        shock_angle_new = np.maximum(shock_angle_new, obs.mach_angle(M=M1))
    print(f"Vtheta at surf: {v_theta[-1]}")
    print(f"Cone angle: {np.degrees(shock_angle_new)}")
    fx2 = fx1
    fx1 = v_theta[-1]
    x2 = x1
    x1 = shock_angle_guess
    if np.abs(v_theta[-1]) < epsilon:
        return solution
    else:
        return _tay_mac_cone(shock_angle_guess=shock_angle_new, M1=M1,
                             cone_angle=cone_angle, gam=gam, x1=x1, x2=x2, fx1=fx1, fx2=fx2)


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
    v_shock = nondimensional_velocity_from_mach(M=m2, gam=gam)
    vr_shock = nondimensional_velocity_radial(velocity_nondim=v_shock, shock_angle=shock_angle,
                                              deflection_angle=deflection_angle)
    vtheta_shock = nondimensional_velocity_azimuth(velocity_nondim=v_shock, shock_angle=shock_angle,
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
    return (theta, vr, vtheta)
