from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, fminbound

import minuteman.cpg.isentropic_flow as isentropic_flow
import minuteman.cpg.oblique_shock as oblique_shock
from minuteman.cpg.oblique_shock import ObliqueShockType
from minuteman.utils.types import (
    DeveloperError,
    Floatlike,
    ndarray_f,
)

_delta: float = 1e-9
"""value used to add/subtract from max/min values"""


@dataclass
class ConeFlowSolution:
    r"""Flowfield solution for cone flow for a calorically perfect gas.

    Arraylike quantities vary as a function of the polar angle, $\theta$."""

    mach_upstream: float
    r"""Upstream mach number, $M_1$"""

    polar_angle: ndarray_f
    r"""Polar or cross-flow angle decreasing from the shock to the
        cone surface, $\theta$ [radians]"""

    @property
    def shock_angle(self) -> float:
        r"""Shock angle, $\theta_s$ [radians]"""
        return self.polar_angle[0]

    @property
    def cone_angle(self) -> float:
        r"""Cone half-angle, $\theta_c$ [radians]"""
        return self.polar_angle[-1]

    specific_heat_ratio: float
    r"""Ratio of specific heats, $\gamma$"""

    flow_angle: ndarray_f
    r"""Flow angle w.r.t. the cone axis, $\psi$ [radians]"""

    velocity_radial: ndarray_f
    r"""Nondimensional radial velocity, $V'_r$"""

    velocity_polar: ndarray_f
    r"""Nondimensional polar velocity, $V'_{\theta}$"""

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

    total_pressure_ratio: float
    r"""Total pressure ratio, $p_0 / p_01$. This value is a constant"""


def nondimensional_velocity_from_mach(
    mach: Any, specific_heat_ratio: Any
) -> Any:
    r"""Compute the nondimensional velocity $V'$ useful to nondimensionalizing
    the Taylor Maccoll equations.

    $V' = V / V_{max}$ in chapter 10 of [1]. $V_{max}$ is a max theoretical
    velocity if the flow were expanded to 0 K.

    Args:
        mach (Any): Mach number $M$
        specific_heat_ratio (Any): ratio of specific heats, $\gamma$

    Returns:
        Any: nondimensional velocity $V'$
    """
    return (2.0 / ((specific_heat_ratio - 1.0) * mach**2) + 1.0) ** -0.5


def nondimensional_velocity_from_components(
    velocity_radial: Any, velocity_polar: Any
) -> Any:
    r"""Compute the nondimensional velocity $V'$ useful to nondimensionalizing
    the Taylor Maccoll equations from its radial and polar components.

    Args:
        velocity_radial (Any): nondimensional radial velocity $V'_r$
        velocity_polar (Any): nondimensional polar velocity $V'_{\theta}$

    Returns:
        Any: nondimensional velocity $V'$
    """
    return np.linalg.norm((velocity_radial, velocity_polar), axis=0)


def mach_from_nondimensional_velocity(
    velocity: Any, specific_heat_ratio: Any
) -> Any:
    r"""Compute Mach number $M$ from nondimensional velocity $V'$

    Args:
        velocity (Any): nondimensional velocity, $V'$
        specific_heat_ratio (Any): ratio of specific heats, $\gamma$

    Returns:
        Any: Mach number $M$
    """
    return (0.5 * (specific_heat_ratio - 1) * (velocity**-2 - 1.0)) ** -0.5


def nondimensional_velocity_polar(
    velocity: Any, shock_angle: Any, deflection_angle: Any
) -> Any:
    r"""Compute the nondimensional polar velocity $V'_{\theta}$ for conic flow

    The quantity is negative, as the $+V'_{\theta}$ axis in the
    coordinate system is positive pointing away from the body.

    Args:
        velocity (Any): nondimensional velocity, $V'$
        shock_angle (Any): shock angle $\theta_s$ [radians]
        deflection_angle (Any): flow deflection angle, $\theta$ [radians]

    Returns:
        Any: polar component of nondimensional velocity, $V'_{\theta}$
    """
    return -np.sin(shock_angle - deflection_angle) * velocity


def nondimensional_velocity_radial(
    velocity: Any, shock_angle: Any, deflection_angle: Any
) -> Any:
    r"""Compute the nondimensional radial velocity $V'_r$ for conic flow

    The quantity is positive in the downstream direction

    Args:
        velocity (Any): nondimensional velocity, $V'$
        shock_angle (Any): shock angle $\theta_s$ [radians]
        deflection_angle (Any): flow deflection angle, $\theta$ [radians]

    Returns:
        Any: radial component of nondimensional velocity, $V'_r$
    """
    return np.cos(shock_angle - deflection_angle) * velocity


def deflection_angle_by_velocity_components(
    polar_angle: ndarray_f,
    velocity_radial: ndarray_f,
    velocity_polar: ndarray_f,
) -> ndarray_f:
    r"""Compute the flow deflection angle $\psi$ at all polar angles $\theta$

    Args:
        polar_angle(ndarray_f): polar angle $\theta$ [radians]
        velocity_radial (ndarray_f): nondimensional radial velocity $V'_r$
        velocity_polar (ndarray_f): nondimensional polar velocity $V'_{\theta}$

    Returns:
        ndarray_f: flow deflection angle $\psi$ at all polar angles post-shock
    """
    return polar_angle + np.atan(velocity_polar / velocity_radial)


def _taylor_maccoll_odes(
    polar_angle: float,
    nondim_velocity_components: tuple[float, float],
    specific_heat_ratio: float,
) -> tuple[float, float]:
    r"""Return the Taylor-Maccoll 2nd-order ODE as a system of first-order ODE,
    evaluated for a polar angle $\theta$ and nondimensional velocity
    components $V'_r$ and $V'_{\theta}$.

    Let

    $y_1 = V'_r$,

    $y_2 = \frac{dV'_r}{d\theta} = V'_{\theta}$

    Then,

    $\frac{dy_1}{d\theta} = \frac{dV'_r}{d\theta} = y_2$

    $\frac{dy_2}{d\theta} = \frac{d^2V'_r}{d\theta^2}$

    Args:
        polar_angle (float): polar angle at which to evaluate
            ODE, $\theta$ [radians]
        nondim_velocity_components (tuple[float, float]): radial and polar
            velocity components, $V'_r$ and $V'_{\theta}$
        specific_heat_ratio (float): ratio of specific heats, $\gamma$

    Returns:
        tuple[float, float]: ($\frac{dy_1}{d\theta}$, $\frac{dy_2}{d\theta}$)
    """
    y1, y2 = nondim_velocity_components
    dy1_dtheta = y2  # irrotational condition

    gamma_term = 0.5 * (specific_heat_ratio - 1)
    # nondim. speed of sound squared, a^2
    a2 = gamma_term * (1.0 - y1**2 - y2**2)
    dy2_dtheta = (y1 * y2**2 - a2 * (2 * y1 + y2 / np.tan(polar_angle))) / (
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


def lookup_solution_by_cone_angle(
    cone_angle: Floatlike,
    mach_upstream: Floatlike,
    specific_heat_ratio: Floatlike = 1.4,
    shock_type: ObliqueShockType = ObliqueShockType.weak,
) -> ConeFlowSolution:
    r"""Solve a cone flow problem with a known surface Mach number, $M_c$

    Args:
        cone_angle (Floatlike): cone angle, $\theta_c$ [radians]
        mach_upstream (Floatlike): upstream Mach number, $M_1$
        specific_heat_ratio (Floatlike): ratio of specific heats, $\gamma$,
            defaults to 1.4
        shock_type(ObliqueShockType): shock type, strong or weak, defaults to
            a weak shock (almost always what you want)

    Returns:
        ConeFlowSolution: cone flow solution
    """
    m1 = float(mach_upstream)
    theta_c = float(cone_angle)
    gam = float(specific_heat_ratio)

    theta, v_r, v_theta = solve_taylor_maccoll_by_cone_angle(
        cone_angle=theta_c,
        mach_upstream=m1,
        specific_heat_ratio=gam,
        shock_type=shock_type,
    )

    v = nondimensional_velocity_from_components(
        velocity_radial=v_r, velocity_polar=v_theta
    )
    mach = mach_from_nondimensional_velocity(
        velocity=v, specific_heat_ratio=gam
    )
    flow_angle = deflection_angle_by_velocity_components(
        polar_angle=theta, velocity_radial=v_r, velocity_polar=v_theta
    )

    p02_p01, p_p1, t_t1, r_r1 = _get_thermo_qty(
        mach=mach,
        shock_angle=theta[0],
        mach_upstream=m1,
        specific_heat_ratio=gam,
    )
    return ConeFlowSolution(
        mach_upstream=m1,
        polar_angle=theta,
        specific_heat_ratio=gam,
        flow_angle=flow_angle,
        velocity_radial=v_r,
        velocity_polar=v_theta,
        velocity=v,
        mach=mach,
        pressure_ratio=p_p1,
        total_pressure_ratio=p02_p01,
        temperature_ratio=t_t1,
        density_ratio=r_r1,
    )


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
        tuple[3*float]: polar angle $\theta$, nondimensional radial velocity
            $V'_r$, and polar velocity $V'_{\theta}$
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
            "Max cone angle for an attached shock is"
            + f" {np.degrees(theta_c_max)} deg at this flight condition"
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


def lookup_solution_by_surface_mach(
    surface_mach: Floatlike,
    mach_upstream: Floatlike,
    specific_heat_ratio: Floatlike = 1.4,
) -> ConeFlowSolution:
    r"""Solve a cone flow problem with a known surface Mach number, $M_c$

    Args:
        surface_mach (Floatlike): Mach number at the surface of the cone, $M_c$
        mach_upstream (Floatlike): upstream Mach number, $M_1$
        specific_heat_ratio (Floatlike): ratio of specific heats, $\gamma$,
            defaults to 1.4

    Returns:
        ConeFlowSolution: cone flow solution
    """
    m1 = float(mach_upstream)
    m_c = float(surface_mach)
    gam = float(specific_heat_ratio)

    theta, v_r, v_theta = solve_taylor_maccoll_by_surface_mach(
        surface_mach=m_c,
        mach_upstream=m1,
        specific_heat_ratio=gam,
    )

    v = nondimensional_velocity_from_components(
        velocity_radial=v_r, velocity_polar=v_theta
    )
    mach = mach_from_nondimensional_velocity(
        velocity=v, specific_heat_ratio=gam
    )
    flow_angle = deflection_angle_by_velocity_components(
        polar_angle=theta, velocity_radial=v_r, velocity_polar=v_theta
    )

    p02_p01, p_p1, t_t1, r_r1 = _get_thermo_qty(
        mach=mach,
        shock_angle=theta[0],
        mach_upstream=m1,
        specific_heat_ratio=gam,
    )
    return ConeFlowSolution(
        mach_upstream=m1,
        polar_angle=theta,
        specific_heat_ratio=gam,
        flow_angle=flow_angle,
        velocity_radial=v_r,
        velocity_polar=v_theta,
        velocity=v,
        mach=mach,
        pressure_ratio=p_p1,
        total_pressure_ratio=p02_p01,
        temperature_ratio=t_t1,
        density_ratio=r_r1,
    )


def solve_taylor_maccoll_by_surface_mach(
    surface_mach: Floatlike,
    mach_upstream: Floatlike,
    specific_heat_ratio: Floatlike,
) -> tuple[ndarray_f, ndarray_f, ndarray_f]:
    r"""Compute the solution to the Taylor-Maccoll equations for a given
    Mach number at the surface of the cone, $M_c$.

    Args:
        surface_mach (Floatlike): Mach number at the surface of the cone, $M_c$
        mach_upstream (Floatlike): upstream Mach number, $M_1$
        specific_heat_ratio (Floatlike): ratio of specific heats, $\gamma$

    Returns:
        tuple[ndarray_f,ndarray_f,ndarray_f]: polar angle $\theta$,
            nondimensional radial velocity $V'_r$,
            and polar velocity $V'_{\theta}$

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
        velocity_radial=vr90, velocity_polar=vtheta90
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
            velocity_radial=vr, velocity_polar=vtheta
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


def lookup_solution_by_shock_angle(
    shock_angle: Floatlike,
    mach_upstream: Floatlike,
    specific_heat_ratio: Floatlike = 1.4,
) -> ConeFlowSolution:
    r"""Solve a cone flow problem with a known shock angle, $\theta_s$

    Args:
        shock_angle (Floatlike): shock angle, $\theta_s$ [radians]
        mach_upstream (Floatlike): upstream Mach number, $M_1$
        specific_heat_ratio (Floatlike): ratio of specific heats, $\gamma$,
            defaults to 1.4

    Returns:
        ConeFlowSolution: cone flow solution
    """
    m1 = float(mach_upstream)
    theta_s = float(shock_angle)
    gam = float(specific_heat_ratio)

    theta, v_r, v_theta = solve_taylor_maccoll_by_shock_angle(
        shock_angle=theta_s,
        mach_upstream=m1,
        specific_heat_ratio=gam,
    )

    v = nondimensional_velocity_from_components(
        velocity_radial=v_r, velocity_polar=v_theta
    )
    mach = mach_from_nondimensional_velocity(
        velocity=v, specific_heat_ratio=gam
    )
    flow_angle = deflection_angle_by_velocity_components(
        polar_angle=theta, velocity_radial=v_r, velocity_polar=v_theta
    )

    p02_p01, p_p1, t_t1, r_r1 = _get_thermo_qty(
        mach=mach,
        shock_angle=theta[0],
        mach_upstream=m1,
        specific_heat_ratio=gam,
    )
    return ConeFlowSolution(
        mach_upstream=m1,
        polar_angle=theta,
        specific_heat_ratio=gam,
        flow_angle=flow_angle,
        velocity_radial=v_r,
        velocity_polar=v_theta,
        velocity=v,
        mach=mach,
        pressure_ratio=p_p1,
        total_pressure_ratio=p02_p01,
        temperature_ratio=t_t1,
        density_ratio=r_r1,
    )


def _get_thermo_qty(
    mach: ndarray_f,
    shock_angle: float,
    mach_upstream: float,
    specific_heat_ratio: float,
) -> tuple[float, ndarray_f, ndarray_f, ndarray_f]:
    m1 = mach_upstream
    gam = specific_heat_ratio

    # get isentropic quantities post-shock (e.g., p0/p)
    isentropic_postshock = isentropic_flow.lookup_table_by_mach(
        mach=mach, specific_heat_ratio=gam
    )
    p0_p = isentropic_postshock.pressure
    t0_t = isentropic_postshock.temperature
    r0_r = isentropic_postshock.density
    # get isentropic quantities upstream (e.g., p01/p1)
    isentropic_upstream = isentropic_flow.lookup_table_by_mach(
        mach=m1, specific_heat_ratio=gam
    )
    p01_p1 = isentropic_upstream.pressure
    t01_t1 = isentropic_upstream.temperature
    r01_r1 = isentropic_upstream.density
    # get quantities across shock (e.g., p02/p01)
    across_shock_21 = oblique_shock.lookup_table_by_shock_angle(
        shock_angle=shock_angle, mach_upstream=m1, specific_heat_ratio=gam
    )
    p02_p01 = across_shock_21.total_pressure_ratio.item()
    t02_t01 = 1.0  # shocks are adiabatic
    r02_r01 = p02_p01  # from ideal gas law, temperature ratio=1
    # p02 constant post-shock so p0==p02 (same for all quantities)
    p_p1 = p01_p1 * p02_p01 / p0_p
    r_r1 = r01_r1 * r02_r01 / r0_r
    t_t1 = t01_t1 * t02_t01 / t0_t
    return p02_p01, p_p1, t_t1, r_r1


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
        tuple[3*float]: polar angle $\theta$, nondimensional radial velocity
            $V'_r$, and polar velocity $V'_{\theta}$

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
        velocity=v_shock,
        shock_angle=theta_s,
        deflection_angle=deflection_angle,
    )
    vtheta_shock = nondimensional_velocity_polar(
        velocity=v_shock,
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
        fun=_taylor_maccoll_odes,
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
