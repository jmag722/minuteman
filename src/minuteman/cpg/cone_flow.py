from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, fminbound

import minuteman.cpg.isentropic_flow as isentropic_flow
import minuteman.cpg.oblique_shock as oblique_shock
from minuteman.cpg.oblique_shock import ObliqueShockType
from minuteman.utils.types import (
    Floatlike,
    ndarray_f,
    DeveloperError,
)

_delta: float = 1e-9
"""value used to add/subtract from max/min values"""


@dataclass
class ConeFlowSolution:
    """Flowfield solution for cone flow for a calorically perfect gas.

    Arraylike quantities vary as a function of the normal angle."""

    mach_upstream: float
    r"""Upstream mach number, $M_1$"""

    normal_angle: ndarray_f
    r"""Normal angle decreasing from the shock to the cone surface,
        $\theta$ [radians]"""

    @property
    def shock_angle(self) -> float:
        r"""Shock angle, $\theta_s$ [radians]"""
        return self.normal_angle[0]

    @property
    def cone_angle(self) -> float:
        r"""Cone half-angle, $\theta_c$ [radians]"""
        return self.normal_angle[-1]

    specific_heat_ratio: float
    r"""Ratio of specific heats, $\gamma$"""

    flow_angle: ndarray_f
    r"""Flow angle w.r.t. the cone axis, $\psi$ [radians]"""

    velocity_radial: ndarray_f
    r"""Nondimensional radial velocity, $V'_r$"""

    velocity_normal: ndarray_f
    r"""Nondimensional normal velocity, $V'_{\theta}$"""

    velocity: ndarray_f
    r"""Nondimensional velocity magnitude, $V'$"""

    mach: ndarray_f
    r"""Downstream mach number, $M$"""

    pressure_ratio: ndarray_f
    r"""Pressure ratio, $p / p_1$"""

    temperature_ratio: ndarray_f
    r"""Temperature ratio, $T / T_1$"""

    density_ratio: ndarray_f
    r"""Density ratio, $\rho / \rho_1$"""

    total_pressure_ratio: ndarray_f
    r"""Total pressure ratio, $p_0 / p_01$"""


def nondimensional_velocity_from_mach(mach, specific_heat_ratio):
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
    return (2.0 / ((specific_heat_ratio - 1.0) * mach**2) + 1.0) ** -0.5


def nondimensional_velocity_from_components(velocity_radial, velocity_azimuth):
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


def mach_from_nondimensional_velocity(velocity, specific_heat_ratio):
    """Compute Mach number from nondimensional velocity

    Args:
        velocity (float): nondimensional velocity, V' [-]
        gam (float): ratio of specific heats [-]

    Returns:
        float: Mach number
    """
    return (0.5 * (specific_heat_ratio - 1) * (velocity**-2 - 1.0)) ** -0.5


def nondimensional_velocity_azimuth(
    velocity_nondim, shock_angle, deflection_angle
):
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


def nondimensional_velocity_radial(
    velocity_nondim, shock_angle, deflection_angle
):
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


def taylor_maccoll_odes(
    azimuth_angles: np.ndarray, nondim_velocity_components: tuple, gam: float
):
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
    dy2_dtheta = (y1 * y2**2 - a2 * (2 * y1 + y2 / np.tan(azimuth_angles))) / (
        a2 - y2**2
    )
    return dy1_dtheta, dy2_dtheta


def cone_shock_angle_maxes(
    mach_upstream: Floatlike, specific_heat_ratio: Floatlike
) -> tuple[float, float]:
    r"""Compute the max cone angle $\theta_{c,max}$ for a given upstream
    condition before the shock detaches, as well as the shock angle at that
    max cone angle condition, $\theta_{s,max}$


    Args:
        mach_upstream: Upstream Mach number, $M_1$
        specific_heat_ratio: Ratio of specific heats, $\gamma$

    Returns:
        tuple[float, float]: (max cone angle $\theta_{c,max}$,
            shock angle for the max cone angle $\theta_{s,max}$)

    Raises:
        DeveloperError: Solver did not converge
    """
    m1 = float(mach_upstream)
    gam = float(specific_heat_ratio)

    def shock_angle_for_max_defl(_theta_shock, _m1, _g):
        _theta, _, _ = solve_taylor_maccoll_by_shock_angle(
            _theta_shock, mach_upstream=_m1, specific_heat_ratio=_g
        )
        # we want to maximixe the cone angle
        return -_theta[-1]

    mu = isentropic_flow.mach_angle(mach=m1)[0]
    xopt, fval, ierr, _ = fminbound(
        shock_angle_for_max_defl,
        x1=mu * (1 + _delta),
        x2=0.5 * np.pi - _delta,
        args=(m1, gam),
        xtol=1e-12,
        maxfun=100,
        full_output=True,
    )
    if ierr:
        raise DeveloperError(f"fminbound failed with exit code: {ierr}")
    theta_s_at_cone_max = xopt
    theta_c_max = -fval
    return theta_c_max, theta_s_at_cone_max


def solve_taylor_maccoll_by_cone_angle(
    cone_angle: Floatlike,
    mach_upstream: Floatlike,
    specific_heat_ratio: Floatlike,
    shock_type: ObliqueShockType,
) -> tuple[ndarray_f, ndarray_f, ndarray_f]:
    r"""Solve the Taylor-Maccoll equations for a given cone angle, $\theta_c$

    Args:
        cone_angle (Floatlike): cone angle, $\theta_c$ [radians]
        mach_upstream (Floatlike): Upstream Mach number, $M_1$
        specific_heat_ratio (Floatlike): ratio of specific heats, $\gamma$
        shock_type(ObliqueShockType): shock type, strong or weak

    Returns:
        tuple[3*float]: normal angle $\theta$, nondimensional radial velocity
            $V'_r$, and normal velocity $V'_{\theta}$
    """
    theta_c = float(cone_angle)
    m1 = float(mach_upstream)
    gam = float(specific_heat_ratio)

    class InvalidConeAngleError(Exception):
        pass

    if theta_c < 0.0 or theta_c >= 0.5 * np.pi:
        raise InvalidConeAngleError(
            "Must be nonnegative but less than 90 deg."
        )

    theta_c_max, theta_s_at_cone_max = cone_shock_angle_maxes(
        mach_upstream=m1, specific_heat_ratio=gam
    )
    if theta_c > theta_c_max:
        raise InvalidConeAngleError(
            f"Max cone angle for an attached shock is {np.degrees(theta_c_max)}"
            + " at this flight condition"
        )

    mu = isentropic_flow.mach_angle(mach=m1)[0]
    if shock_type == ObliqueShockType.weak:
        min_shock_angle = mu * (1 + _delta)
        max_shock_angle = theta_s_at_cone_max - _delta
    else:
        min_shock_angle = theta_s_at_cone_max + _delta
        max_shock_angle = np.radians(90.0 - _delta)

    def find_matching_cone_angle(_theta_shock, _theta_cone, _m1, _g):
        theta, _, _ = solve_taylor_maccoll_by_shock_angle(
            shock_angle=_theta_shock, mach_upstream=_m1, specific_heat_ratio=_g
        )
        return theta[-1] - _theta_cone

    # multiply by 1+small decimal to avoid numerical issues with edge case
    shock_angle, result = brentq(
        find_matching_cone_angle,
        a=min_shock_angle,
        b=max_shock_angle,
        args=(theta_c, m1, gam),
        full_output=True,
    )
    if not result.converged:
        raise DeveloperError(f"Root-finding failed: {result.flag}")
    return solve_taylor_maccoll_by_shock_angle(
        shock_angle=shock_angle, mach_upstream=m1, specific_heat_ratio=gam
    )


def solve_taylor_maccoll_by_surface_mach(
    surface_mach: Floatlike,
    mach_upstream: Floatlike,
    specific_heat_ratio: Floatlike,
):
    r"""Compute the solution to the Taylor-Maccoll equations for a given
    Mach number at the surface of the cone, $M_c$.

    Args:
        surface_mach (Floatlike): Mach number at the surface of the cone, $M_c$
        mach_upstream (Floatlike): upstream Mach number, $M_1$
        specific_heat_ratio (Floatlike): ratio of specific heats, $\gamma$

    Returns:
        tuple[3*float]: normal angle $\theta$, nondimensional radial velocity
            $V'_r$, and normal velocity $V'_{\theta}$

    Raises:
        InvalidSurfaceMachError: Surface Mach number is not possible for the
            given freestream condition
    """
    mc = float(surface_mach)
    m1 = float(mach_upstream)
    gam = float(specific_heat_ratio)

    class InvalidSurfaceMachError(Exception):
        pass

    # check Mc
    if mc >= m1:
        raise InvalidSurfaceMachError("Must be less than freestream Mach")
    # Mc cannot be lower than Mc for a shock angle of 90 degrees
    _, vr90, vtheta90 = solve_taylor_maccoll_by_shock_angle(
        shock_angle=0.5 * np.pi, mach_upstream=m1, specific_heat_ratio=gam
    )
    v90 = nondimensional_velocity_from_components(
        velocity_radial=vr90, velocity_azimuth=vtheta90
    )
    m90 = mach_from_nondimensional_velocity(v90, specific_heat_ratio=gam)[-1]
    if mc < m90:
        raise InvalidSurfaceMachError(
            f"Must be greater than {m90:.3f} for this freestream condition"
        )

    min_shock_angle = isentropic_flow.mach_angle(mach=m1)[0] * (1 + _delta)
    max_shock_angle = np.radians(90.0 - _delta)

    def find_matching_surface_mach(theta_s, _mc, _m1, _gam):
        _, vr, vtheta = solve_taylor_maccoll_by_shock_angle(
            shock_angle=theta_s, mach_upstream=_m1, specific_heat_ratio=_gam
        )
        vprime = nondimensional_velocity_from_components(
            velocity_radial=vr, velocity_azimuth=vtheta
        )
        mach = mach_from_nondimensional_velocity(
            velocity=vprime, specific_heat_ratio=gam
        )[-1]
        return mach - _mc

    shock_angle, result = brentq(
        find_matching_surface_mach,
        a=min_shock_angle,
        b=max_shock_angle,
        args=(mc, m1, gam),
        full_output=True,
    )
    if not result.converged:
        raise DeveloperError(f"Root-finding failed: {result.flag}")
    return solve_taylor_maccoll_by_shock_angle(
        shock_angle=shock_angle, mach_upstream=m1, specific_heat_ratio=gam
    )


def solve_taylor_maccoll_by_shock_angle(
    shock_angle: Floatlike,
    mach_upstream: Floatlike,
    specific_heat_ratio: Floatlike,
) -> tuple[ndarray_f, ndarray_f, ndarray_f]:
    r"""Compute the solution to the Taylor Maccoll equations for a given
    shock angle, $\theta_s$.

    Args:
        shock_angle (Floatlike): shock angle, $\theta_s$ [radians]
        mach_upstream (Floatlike): upstream Mach number, $M_1$
        specific_heat_ratio (Floatlike): ratio of specific heats, $\gamma$

    Returns:
        tuple[3*float]: normal angle $\theta$, nondimensional radial velocity
            $V'_r$, and normal velocity $V'_{\theta}$

    Raises:
        InvalidNormalVelocity: Normal velocity has the wrong sign, check inputs
        RungeKuttaIntegrationError: IVP solver failed, check inputs
    """
    theta_s = float(shock_angle)
    m1 = float(mach_upstream)
    gam = float(specific_heat_ratio)
    oblique_shock.check_shock_angle(shock_angle=theta_s, mach=m1)
    mn1 = oblique_shock.mach_upstream_normal_component(
        mach_upstream=m1, shock_angle=theta_s
    )[0]
    # compute quantities immediately behind oblique shock
    mn2 = oblique_shock.mach_downstream_normal_component(
        mach_upstream_normal=mn1, specific_heat_ratio=gam
    )[0]
    deflection_angle = oblique_shock.deflection_angle_by_shock_mach(
        shock_angle=theta_s, mach_upstream=m1, specific_heat_ratio=gam
    )[0]
    m2 = oblique_shock.mach_downstream_by_postshock(
        mach_downstream_normal=mn2,
        shock_angle=theta_s,
        deflection_angle=deflection_angle,
    )[0]
    v_shock = nondimensional_velocity_from_mach(
        mach=m2, specific_heat_ratio=gam
    )
    vr_shock = nondimensional_velocity_radial(
        velocity_nondim=v_shock,
        shock_angle=theta_s,
        deflection_angle=deflection_angle,
    )
    vtheta_shock = nondimensional_velocity_azimuth(
        velocity_nondim=v_shock,
        shock_angle=theta_s,
        deflection_angle=deflection_angle,
    )

    class InvalidNormalVelocity(Exception):
        pass

    if vtheta_shock >= 0.0:
        raise InvalidNormalVelocity("Velocity must be negative")

    @dataclass
    class VthetaEqualsZeroEvent:
        terminal: bool = True
        direction: float = 1.0

        # pyrefly: ignore
        def __call__(
            self, t: float, y: tuple[float, float], gam: float
        ) -> float:
            return y[1]

    solution = solve_ivp(
        fun=taylor_maccoll_odes,
        t_span=(theta_s, 0.0),
        y0=(vr_shock, vtheta_shock),
        args=(gam,),
        events=VthetaEqualsZeroEvent(),
        atol=1e-18,
        rtol=1e-13,
    )

    class RungeKuttaIntegrationError(Exception):
        pass

    if not solution.success:
        msg = f"solve_ivp status({solution.status}): {solution.message}"
        raise RungeKuttaIntegrationError(msg)

    vr, vtheta = solution.y
    theta = solution.t
    return theta, vr, vtheta
