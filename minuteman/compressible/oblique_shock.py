"""
This module computes flow parameters of a 2D, stationary, calorically perfect oblique shocks.
"""

import numpy as np


def mach_angle(M: float):
    """Compute the Mach angle.

    Source: Eqn 4.1 in Anderson

    Args:
        M (float): Mach number [-]

    Returns:
        float: Mach angle [radians]
    """
    return np.asin(1.0 / M)


def mach1_normal(M1: float, beta: float):
    """Compute the normal component of the upstream Mach number (Mn1).

    Args:
        M1 (float): upstream Mach number
        beta (float): shock angle [radians]

    Returns:
        float: normal component of the upstream Mach number, Mn1
    """
    check_shock_angle(M=M1, beta=beta)
    return M1 * np.sin(beta)


def density_ratio(Mn1: float, gam: float = 1.4):
    """Compute the density ratio `rho2/rho1` where 2 and 1 are downstream and
    upstream of the oblique shock, respectively.

    Args:
        Mn1 (float): normal component of upstream Mach number
        gam (float, optional): ratio of specific heats. Defaults to 1.4.

    Returns:
        float: density ratio `rho2/rho1`, where 2 and 1 are downstream and
            upstream of the shock, respectively.
    """
    return (gam + 1) * Mn1**2 / ((gam-1) * Mn1**2 + 2)


def pressure_ratio(Mn1: float, gam: float = 1.4):
    """Compute the static pressure ratio `p2/p1` across an oblique shock.

    Args:
        Mn1 (float): normal component of upstream Mach number
        gam (float, optional): ratio of specific heats. Defaults to 1.4.

    Returns:
        float: pressure ratio `p2/p1` where 2 and 1 are downstream and upstream
        of the shock, respectively.
    """
    return 1 + 2 * gam/(gam + 1) * (Mn1**2 - 1)


def mach2_normal(Mn1: float, gam: float = 1.4):
    """Compute the normal component of the downstream Mach number (Mn2).

    Args:
        Mn1 (float): normal component of upstream Mach number
        gam (float, optional): ratio of specific heats. Defaults to 1.4.

    Returns:
        float: normal component of downstream Mach number, Mn2
    """
    return np.sqrt(
        (Mn1**2 + (2 / (gam - 1))) / (2 * gam/(gam - 1) * Mn1**2 - 1)
    )


def temperature_ratio(Mn1: float, gam: float = 1.4):
    """Compute the static temperature ratio `T2/T1` across an oblique shock.

    Args:
        Mn1 (float): normal component of upstream Mach number
        gam (float, optional): ratio of specific heats. Defaults to 1.4.

    Returns:
        float: static temperature ratio `T2/T1` across the oblique shock
    """
    return pressure_ratio(Mn1=Mn1, gam=gam) / density_ratio(Mn1=Mn1, gam=gam)


def mach2(Mn2: float, beta: float, theta: float):
    """Compute Mach number `M2` downstream of oblique shock.

    Args:
        Mn2 (float): normal component of Mach number downstream of oblique shock.
        beta (float): oblique shock angle [radians]
        theta (float): flow deflection angle [radians]

    Returns:
        float: Mach number `M2` downstream of oblique shock
    """
    check_deflection_angle(theta)
    if theta >= beta:
        raise ValueError("Deflection angle must be smaller than shock angle.")
    return Mn2 / np.sin(beta - theta)


def check_shock_angle(beta: float, M: float):
    """Ensure shock angle is within bounds, no less than Mach angle `mu`
    or greater than 90 degrees.

    Args:
        beta (float): shock angle [radians]
        M (float): Mach number [-]

    Raises:
        ValueError: shock angle outside bounds
    """
    mu = mach_angle(M)
    if np.min(beta) < mu or np.max(beta) > np.radians(90):
        raise ValueError(
            f"Shock angle must be between [{np.degrees(mu)}, 90] degrees.")


def check_deflection_angle(theta: float):
    """Ensure deflection angle `theta` is within [0, 90] degrees.

    Args:
        theta (float): deflection angle [radians]

    Raises:
        ValueError: deflection angle outside bounds
    """
    if np.min(theta) < 0.0 or np.max(theta) > np.radians(90):
        raise ValueError(
            "Deflection angle must be between [0, 90] degrees.")


def deflection_angle(M1: float, beta: float, gam: float = 1.4):
    """Compute the deflection/wedge angle `theta` for a given upstream Mach number
    and shock angle `beta`.

    The Theta-Beta-M relation in Anderson, Eq. 4.17

    Args:
        M1 (float): upstream Mach number [-]
        beta (float): oblique shock angle [radians]
        gam (float, optional): ratio of specific heats. Defaults to 1.4.

    Returns:
        float: deflection angle [radians]
    """
    check_shock_angle(beta, M1)
    return np.atan(
        2.0 / np.tan(beta) * (((M1 * np.sin(beta)) ** 2 - 1.0) /
                              (M1**2 * (gam + np.cos(2 * beta)) + 2.0))
    )


def shock_angle(M1: float, theta: float, is_strong_shock: bool = False,
                gam: float = 1.4):
    """Compute the oblique shock angle `beta` for a given Mach and wedge angle Theta.

    The lesser-known Beta-Theta-M relation in Anderson, Eq. 4.19-4.21

    Args:
        M1 (float): upstream Mach number
        theta (float): deflection/wedge angle [radians]
        is_strong_shock (bool, optional): Compute the strong shock or weak
            shock solution. Defaults to False.
        gam (float, optional): ratio of specific heats. Defaults to 1.4.

    Raises:
        ValueError: no solution exists for Mach number, detached shock forms.

    Returns:
        float: shock angle [radians]
    """
    delta = 0.0 if is_strong_shock else 1.0
    check_deflection_angle(theta)
    # for valid solution (undetached shock), lam must be real value, and
    #   xi must be within (-1, 1)

    detachment_crit = ((M1**2 - 1)**2 - 3 * (1 + (gam - 1)/2 * M1**2) *
                       (1 + (gam + 1) / 2 * M1**2) * (np.tan(theta))**2)
    if detachment_crit <= 0.0:
        raise ValueError("No solution, detached shock.")
    lam = detachment_crit ** 0.5

    xi = ((M1**2-1)**3 - 9 * (1 + (gam-1) / 2 * M1**2)
          * (1 + (gam-1) / 2*M1**2 + (gam+1) / 4 * M1**4) * (np.tan(theta))**2) / lam**3
    if np.abs(xi) >= 1.0:
        raise ValueError("No solution, detached shock")

    return np.atan(
        (M1**2 - 1 + 2 * lam * np.cos((4 * np.pi * delta + np.acos(xi)) / 3)) / (
            3 * np.tan(theta) * (1 + (gam-1) / 2 * M1**2)
        )
    )


def max_shock_angle(M1: float, gam: float = 1.4):
    """Compute the shock angle where the the max deflection angle will be
    attained for a given Mach number, after which a detached shock forms.

    Computed by taking derivative of deflection angle wrt shock angle
     (dtheta/dbeta) and setting it equal to zero, and taking the (-) root.

    Source: https://mae-nas.eng.usu.edu/MAE_5420_Web/section8/section8.3.pdf

    Args:
        M1 (float): upstream Mach number
        gam (float, optional): ratio of specific heats. Defaults to 1.4.

    Returns:
        float: shock angle at max deflection for a given Mach
    """
    return 0.5 * np.arccos(
        (M1**2 * (gam - 1) + 4
         - np.sqrt(16*(gam + 1)
                   + 8*M1**2 * (gam**2 - 1) + M1**4 * (gam + 1)**2)) /
        (2 * gam * M1**2)
    )


def max_deflection_angle(M1: float, gam: float = 1.4):
    """Compute the maximum deflection/wedge angle for a given Mach number
    before a detached shock forms.

    Args:
        M1 (float): upstream Mach number [-]
        gam (float, optional): ratio of specific heats. Defaults to 1.4.

    Returns:
        float: max deflection/wedge angle [radians]
    """
    beta = max_shock_angle(M1=M1, gam=gam)
    return deflection_angle(M1=M1, beta=beta, gam=gam)


def sonic_shock_angle(M1: float, gam: float = 1.4):
    """Compute the oblique shock angle for flow that is sonic downstream of
    the oblique shock

    Source: https://mae-nas.eng.usu.edu/MAE_5420_Web/section8/section8.3.pdf

    Args:
        M1 (float): upstream Mach number [-]
        gam (float, optional): ratio of specific heats. Defaults to 1.4.

    Returns:
        float: sonic shock angle [radians]
    """
    return np.arcsin(np.sqrt(
        ((gam-3)*M1**2 + (gam+1)*M1**4 + np.sqrt(
            16*gam*M1**4 + ((gam-3)*M1**2 + (gam+1)*M1**4)**2
        )) / (4*gam*M1**4)
    ))


def sonic_deflection_angle(M1: float, gam: float = 1.4):
    """Compute the deflection angle such that, for a given Mach, flow has
    reached the sonic condition right behind an oblique shock.

    Args:
        M1 (float): upstream Mach number [-]
        gam (float, optional): ratio of specific heats. Defaults to 1.4.

    Returns:
        float: deflection angle yielding sonic flow [radians]
    """
    beta = sonic_shock_angle(M1=M1, gam=gam)
    return deflection_angle(M1=M1, beta=beta, gam=gam)
