from dataclasses import dataclass

import numpy as np
from scipy.optimize import fsolve

import minuteman.cpg.thermo as thermo
from minuteman import cpg
from minuteman.utils.types import (
    Floatlike,
    ndarray_b,
    ndarray_f,
)

"""
Solve the 1D Sod shock tube problem
"""


@dataclass
class ShockTubeSolution:
    """Solution to the Sod Shock tube"""

    position: ndarray_f
    """Position in the shock tube, $x$"""
    time: Floatlike
    """Time, $t$"""
    pressure: ndarray_f
    """Pressure, $p$"""
    density: ndarray_f
    r"""Density, $\rho$"""
    temperature: ndarray_f
    """Temperature, $T$"""
    specific_heat_ratio: ndarray_f
    r"""Ratio of specific heats, $\gamma$"""
    gas_constant: ndarray_f
    """Specific gas constant, $R$"""
    mach: ndarray_f
    """Mach number, $M$"""
    velocity: ndarray_f
    """Velocity, $v$"""
    speed_of_sound: ndarray_f
    """Speed of sound, $a$"""
    entropy: ndarray_f
    """Entropy, $s$"""
    internal_energy: ndarray_f
    """Specific internal energy, $e$"""
    enthalpy: ndarray_f
    """Specific internal enthalpy, $h$"""
    total_energy: ndarray_f
    """Total energy per unit volume, $E_t$"""
    region_1: ndarray_b
    """Region 1 solution mask"""
    region_2: ndarray_b
    """Region 2 solution mask"""
    region_3: ndarray_b
    """Region 3 solution mask"""
    region_4: ndarray_b
    """Region 4 solution mask"""
    region_5: ndarray_b
    """Region 5 solution mask"""


def solve_sod(
    time: Floatlike,
    pressure_l: Floatlike,
    pressure_r: Floatlike,
    density_l: Floatlike,
    density_r: Floatlike,
    specific_heat_ratio_l: Floatlike = 1.4,
    specific_heat_ratio_r: Floatlike = 1.4,
    gas_constant_l: Floatlike = thermo.gas_constant_air_si,
    gas_constant_r: Floatlike = thermo.gas_constant_air_si,
    tube_length: Floatlike = 20.0,
    position: ndarray_f | None = None,
) -> ShockTubeSolution:
    r"""Computes Sod shock tube problem. Both gases initially stagnant, with
    the contact surface centered in the tube at x=0.

    Args:
        time (Floatlike): time, $t$ [seconds]
        pressure_l (Floatlike): pressure on left hand side (LHS), $p_L$ [Pa]
        pressure_r (Floatlike): pressure on right hand side (RHS), $p_R$ [Pa]
        density_l (Floatlike): density on LHS, $\rho_L$ [kg/$m^3$]
        density_r (Floatlike): density on RHS, $\rho_R$ [kg/$m^3$]
        specific_heat_ratio_l (Floatlike, optional): ratio of specific heats on
            LHS, $\gamma_L$. Defaults to 1.4.
        specific_heat_ratio_r (Floatlike, optional): ratio of specific heats on
            RHS, $\gamma_R$. Defaults to 1.4.
        gas_constant_l (Floatlike, optional): specific gas constant on LHS,
            $R_L$. Defaults to ``thermo.gas_constant_air_si``.
        gas_constant_r (Floatlike, optional): specific gas constant on RHS,
            $R_R$. Defaults to ``thermo.gas_constant_air_si``.
        tube_length (Floatlike, optional): Length of shock tube.
            Defaults to 20.0, the Sod problem #1 length
        position (ndarray_f | None, optional): explicit positions along
            the shock tube to evaluate. This is helpful when comparing to a
            CFD grid directly. Defaults to ``None``.

    Returns:
        ShockTubeSolution: Sod shock tube solution
    """
    # initialize driver gas (region 4) and driven gas (region 1)
    left_driver = pressure_l >= pressure_r
    p4 = pressure_l if left_driver else pressure_r
    p1 = pressure_r if left_driver else pressure_l
    r4 = density_l if left_driver else density_r
    r1 = density_r if left_driver else density_l
    gam4 = specific_heat_ratio_l if left_driver else specific_heat_ratio_r
    gam1 = specific_heat_ratio_r if left_driver else specific_heat_ratio_l
    gas_const4 = gas_constant_l if left_driver else gas_constant_r
    gas_const1 = gas_constant_r if left_driver else gas_constant_l

    # derived driver quantities
    p41 = p4 / p1
    a4 = cpg.speed_of_sound_from_pressure(
        specific_heat_ratio=gam4, pressure=p4, density=r4
    )[0]
    t4 = p4 / (r4 * gas_const4)
    u4 = 0.0

    # derived driven quantities
    a1 = cpg.speed_of_sound_from_pressure(
        pressure=p1, density=r1, specific_heat_ratio=gam1
    )[0]
    t1 = p1 / (r1 * gas_const1)
    u1 = 0.0

    # compute properties across shock
    p21 = moving_shock_pressure_ratio(
        pressure_ratio=p41,
        speed_of_sound_ratio=a4 / a1,
        specific_heat_ratio_driver=gam4,
        specific_heat_ratio_driven=gam1,
    )
    p2 = p21 * p1
    r2 = r1 * moving_shock_density_ratio(
        pressure_ratio=p21, specific_heat_ratio_driven=gam1
    )
    a2 = cpg.speed_of_sound_from_pressure(
        pressure=p2, density=r2, specific_heat_ratio=gam1
    )[0]
    t2 = t1 * moving_shock_temperature_ratio(
        pressure_ratio=p21, specific_heat_ratio_driven=gam1
    )
    # u2=u3=V=u_piston
    u2 = contact_surface_speed(
        pressure_ratio=p21,
        speed_of_sound_driven=a1,
        specific_heat_ratio_driven=gam1,
    )
    w = moving_shock_speed(
        pressure_ratio=p21,
        speed_of_sound_driven=a1,
        specific_heat_ratio_driven=gam1,
    )

    # compute expansion fan properties
    p34 = p21 / p41  # because p2/p1 = p3/p1
    expansion34 = thermo.isentropic_process_from_pressure(
        pressure_ratio=p34, specific_heat_ratio=gam4
    )
    p3 = p34 * p4
    r3 = expansion34.density_ratio[0] * r4
    a3 = expansion34.speed_of_sound_ratio[0] * a4
    t3 = expansion34.temperature_ratio[0] * t4
    u3 = u2

    # critical positions
    x45 = (u4 - a4) * time  # expansion fan head
    x53 = (u3 - a3) * time  # expansion fan tail
    x32 = u2 * time  # contact surface
    x21 = w * time  # shock
    # using set here removes multiple zeros @ time=0
    crit_pts = {x45, x53, x32, x21}
    if position is None:
        N = 500
        # adding crit pts so discont. always well resolved
        x_soln = np.zeros(N + len(crit_pts))
        x_soln[:N] = np.linspace(-0.5 * tube_length, 0.5 * tube_length, N)
        x_soln[-len(crit_pts):] = list(crit_pts)
        x_soln.sort()
    else:
        x_soln = np.zeros_like(position)
        x_soln[:] = position[:]

    # masks for each region
    region4 = x_soln < x45
    region5 = np.logical_and(x_soln >= x45, x_soln < x53)
    region3 = np.logical_and(x_soln >= x53, x_soln < x32)
    region2 = np.logical_and(x_soln >= x32, x_soln < x21)
    region1 = x_soln >= x21

    def assign_regions(_soln, _4, _5, _3, _2, _1):
        _soln[region4] = _4
        _soln[region5] = _5
        _soln[region3] = _3
        _soln[region2] = _2
        _soln[region1] = _1

    s_soln = np.empty_like(x_soln)  # entropy
    t_soln = np.empty_like(x_soln)  # temperature
    u_soln = np.empty_like(x_soln)  # velocity
    p_soln = np.empty_like(x_soln)  # pressure
    a_soln = np.empty_like(x_soln)  # speed of sound
    r_soln = np.empty_like(x_soln)

    m_soln = np.empty_like(x_soln)
    e_soln = np.empty_like(x_soln)
    h_soln = np.empty_like(x_soln)
    etot_soln = np.empty_like(x_soln)
    gam_soln = np.empty_like(x_soln)
    gas_const_soln = np.empty_like(x_soln)

    u5 = expansion_fan_velocity(a4, x_soln[region5], time, gam4)
    a5 = expansion_fan_speed_of_sound(a4, u5, gam4)
    expansion54 = thermo.isentropic_process_from_speed_of_sound(
        speed_of_sound_ratio=a5 / a4, specific_heat_ratio=gam4
    )
    p5 = expansion54.pressure_ratio[0] * p4
    r5 = expansion54.density_ratio[0] * r4
    t5 = expansion54.temperature_ratio[0] * t4

    assign_regions(u_soln, u4, u5, u3, u2, u1)
    assign_regions(a_soln, a4, a5, a3, a2, a1)
    assign_regions(p_soln, p4, p5, p3, p2, p1)
    assign_regions(r_soln, r4, r5, r3, r2, r1)
    assign_regions(t_soln, t4, t5, t3, t2, t1)
    assign_regions(gam_soln, gam4, gam4, gam4, gam1, gam1)
    assign_regions(
        gas_const_soln, gas_const4, gas_const4, gas_const4,
        gas_const1, gas_const1
    )
    s_soln[:] = thermo.entropy_state(
        pressure=p_soln,
        density=r_soln,
        specific_heat_ratio=gam_soln,
        gas_constant=gas_const_soln,
    )
    m_soln[:] = cpg.mach_number(velocity=u_soln, speed_of_sound=a_soln)
    # specific internal energy = cv * T
    e_soln[:] = t_soln * thermo.specific_heat_constant_volume(
        specific_heat_ratio=gam_soln, gas_constant=gas_const_soln
    )
    h_soln[:] = thermo.specific_enthalpy(
        specific_internal_energy=e_soln, pressure=p_soln, density=r_soln
    )
    etot_soln[:] = thermo.total_energy(
        pressure=p_soln,
        density=r_soln,
        speed=u_soln,
        specific_heat_ratio=gam_soln,
    )

    return ShockTubeSolution(
        position=x_soln,
        time=time,
        temperature=t_soln,
        pressure=p_soln,
        density=r_soln,
        specific_heat_ratio=gam_soln,
        gas_constant=gas_const_soln,
        mach=m_soln,
        velocity=u_soln,
        speed_of_sound=a_soln,
        entropy=s_soln,
        internal_energy=e_soln,
        enthalpy=h_soln,
        total_energy=etot_soln,
        region_1=region1,
        region_2=region2,
        region_3=region3,
        region_4=region4,
        region_5=region5,
    )


def expansion_fan_velocity(
    speed_of_sound_driver: Floatlike,
    position: ndarray_f,
    time: Floatlike,
    specfic_heat_ratio_driver: Floatlike,
) -> ndarray_f:
    r"""Compute the velocity within the expansion fan of a shock tube $u$[^1]

    [^1]: Eq. 7.89, Anderson, J. D., Jr. (2003). *Modern compressible flow: With
        historical perspective* (3rd ed.). McGraw-Hill.

    Args:
        speed_of_sound_driver (Floatlike): speed of sound of driver gas, $a_4$
        position (ndarray_f): position within expansion fan, $x$
        time (ArrayOrScalarFloat): time $t$
        specfic_heat_ratio_driver (Floatlike): ratio of specific heats of
            driver gas, $\gamma_4$

    Returns:
        ndarray_f: velocity within the expansion fan, $u$
    """
    a4 = speed_of_sound_driver
    x = position
    t = time
    gam4 = specfic_heat_ratio_driver
    return 2.0 / (gam4 + 1) * (a4 + x / t)


def expansion_fan_speed_of_sound(
    speed_of_sound_driver: Floatlike,
    velocity: ndarray_f,
    specific_heat_ratio_driver: Floatlike,
) -> ndarray_f:
    r"""Compute speed of sound within expansion fan $a$[^1]

    Args:
        speed_of_sound_driver (Floatlike): speed of sound of driver gas, $a_4$
        velocity (ndarray_f): velocity within the expansion fan
        specific_heat_ratio_driver (Floatlike): ratio of specific heats of
            driver gas, $\gamma_4$

    Returns:
        ndarray_f: speed of sound within expansion fan, $a$
    """
    a4 = speed_of_sound_driver
    u = velocity
    gam4 = specific_heat_ratio_driver
    return a4 - (gam4 - 1) / 2.0 * u


def moving_shock_density_ratio(
    pressure_ratio: Floatlike, specific_heat_ratio_driven: Floatlike
) -> Floatlike:
    r"""Compute the density ratio $\rho_2 / \rho_1$ ratio across a
    moving normal shock[^1]

    [^1]: Eq. 7.11, Anderson, J. D., Jr. (2003). *Modern compressible flow: With
        historical perspective* (3rd ed.). McGraw-Hill.

    Args:
        pressure_ratio (Floatlike): static pressure ratio across
            shock, $p_2 / p_1$
        specific_heat_ratio_driven (Floatlike): ratio of specific heats
            for the driven gas, $\gamma_1$

    Returns:
        Floatlike: density ratio, $\rho_2 / \rho_1$
    """
    p21 = pressure_ratio
    gam1 = specific_heat_ratio_driven
    return (1 + (gam1 + 1) / (gam1 - 1) * p21) / (
        (gam1 + 1) / (gam1 - 1) + p21
    )


def moving_shock_temperature_ratio(
    pressure_ratio: Floatlike, specific_heat_ratio_driven: Floatlike
) -> Floatlike:
    r"""Compute the static temperature $T_2 / T_1$ ratio across a
    moving normal shock[^1]

    [^1]: Eq. 7.10, Anderson, J. D., Jr. (2003). *Modern compressible flow: With
        historical perspective* (3rd ed.). McGraw-Hill.

    Args:
        pressure_ratio (Floatlike): static pressure ratio across
            shock, $p_2 / p_1$
        specific_heat_ratio_driven (Floatlike): ratio of specific heats
            for the driven gas, $\gamma_1$

    Returns:
        Floatlike: static temperature ratio, $T_2 / T_1$
    """
    p21 = pressure_ratio
    gam1 = specific_heat_ratio_driven
    return (
        p21
        * ((gam1 + 1) / (gam1 - 1) + p21)
        / (1 + (gam1 + 1) / (gam1 - 1) * p21)
    )


def moving_shock_speed(
    pressure_ratio: Floatlike,
    speed_of_sound_driven: Floatlike,
    specific_heat_ratio_driven: Floatlike,
) -> Floatlike:
    r"""Compute the shock speed $w$ of the moving shock[^1]

    [^1]: Eq. 7.14, Anderson, J. D., Jr. (2003). *Modern compressible flow: With
        historical perspective* (3rd ed.). McGraw-Hill.

    Args:
        pressure_ratio (Floatlike): static pressure ratio across
            shock, $p_2 / p_1$
        speed_of_sound_driven (Floatlike): speed of sound of driven
            gas, $a_1$
        specific_heat_ratio_driven (Floatlike): ratio of specific heats
            for the driven gas, $\gamma_1$

    Returns:
        Floatlike: wave velocity of the moving shock wave, $w$
    """
    p21 = pressure_ratio
    a1 = speed_of_sound_driven
    gam1 = specific_heat_ratio_driven
    return a1 * ((gam1 + 1) / (2 * gam1) * (p21 - 1) + 1) ** 0.5


def contact_surface_speed(
    pressure_ratio: Floatlike,
    speed_of_sound_driven: Floatlike,
    specific_heat_ratio_driven: Floatlike,
) -> Floatlike:
    r"""Compute the speed of the contact surface/piston $u_p$ in a shock tube,
    or the speed of the mass motion induced by the incident shock[^1].

    For $\gamma_1=1.4$, as the pressure ratio approaches infinity, the
    Mach number approaches 1.89.

    [^1]: Eq. 7.16, Anderson, J. D., Jr. (2003). *Modern compressible flow: With
        historical perspective* (3rd ed.). McGraw-Hill.

    Args:
        pressure_ratio (Floatlike): static pressure ratio across
            shock, $p_2 / p_1$
        speed_of_sound_driven (Floatlike): speed of sound of driven
            gas, $a_1$
        specific_heat_ratio_driven (Floatlike): ratio of specific heats
            for the driven gas, $\gamma_1$
    Returns:
        Floatlike: contact surface or piston speed, $u_p$
    """
    p21 = pressure_ratio
    a1 = speed_of_sound_driven
    gam1 = specific_heat_ratio_driven
    return (
        a1
        / gam1
        * (p21 - 1)
        * (2 * gam1 / (gam1 + 1) / (p21 + (gam1 - 1) / (gam1 + 1))) ** 0.5
    )


def moving_shock_pressure_ratio(
    pressure_ratio: Floatlike,
    speed_of_sound_ratio: Floatlike,
    specific_heat_ratio_driver: Floatlike,
    specific_heat_ratio_driven: Floatlike,
) -> Floatlike:
    r"""Solve for the pressure ratio across a moving normal shock in a shock
    tube, $p_2 / p_1$[^1]

    [^1]: Eq. 7.94, Anderson, J. D., Jr. (2003). *Modern compressible flow: With
        historical perspective* (3rd ed.). McGraw-Hill.

    Args:
        pressure_ratio (Floatlike): pressure ratio between the
            driver and driven gas, $p_4 / p_1$
        speed_of_sound_ratio (Floatlike): speed of sound ratio between the
            driver and driven gas, $a_4 / a_1$
        specific_heat_ratio_driver (Floatlike): ratio of specific heats
            for the driver gas, $\gamma_4$
        specific_heat_ratio_driven (Floatlike): ratio of specific heats
            for the driven gas, $\gamma_1$

    Returns:
        Floatlike: pressure ratio $p_2 / p_1$ across the moving normal shock
    """
    p41 = pressure_ratio
    a41 = speed_of_sound_ratio
    gam4 = specific_heat_ratio_driver
    gam1 = specific_heat_ratio_driven

    def _p21_func(
        p21: Floatlike,
        p41: Floatlike,
        a41: Floatlike,
        gam4: Floatlike,
        gam1: Floatlike,
    ):
        return p41 - p21 * (
            1
            - ((gam4 - 1) / a41 * (p21 - 1))
            / (2 * gam1 * (2 * gam1 + (gam1 + 1) * (p21 - 1))) ** 0.5
        ) ** (-2 * gam4 / (gam4 - 1))

    return fsolve(_p21_func, 0.5 * p41, (p41, a41, gam4, gam1))[0]
