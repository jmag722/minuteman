"""
This module computes flow parameters of 1D, stationary, calorically perfect normal shocks.

Enthalpy is constant across these shocks. For perfect (calorically+thermally) gases,
total temperature is also constant across the shock, and will not be output here.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import fsolve
import minuteman.compressible.isentropic_flow as isentropic_flow
import minuteman.utils.types as ut


@dataclass
class NormalShockTable:
    """Normal shock table, for calorically perfect gas"""

    mach_upstream: ut.ndarray
    """Upstream mach number, M1"""

    mach_downstream: ut.ndarray
    """Downstream mach number, M2"""

    temperature_ratio: ut.ndarray
    """Temperature ratio, T2/T1"""

    pressure_ratio: ut.ndarray
    """Static pressure ratio, p2/p1"""

    density_ratio: ut.ndarray
    """Density ratio, rho2/rho1"""

    total_pressure_ratio: ut.ndarray
    """Total pressure ratio, p02/p01"""

    pitot_pressure_ratio: ut.ndarray
    """Rayleigh Pitot tube pressure ratio, p02/p1"""

    specific_heat_ratio: ut.ndarray
    """Ratio of specific heats, gamma"""


def lookup_table_by_upstream_mach(
    mach_upstream: ut.ndarray | float,
    specific_heat_ratio: ut.ndarray | float = 1.4,
) -> NormalShockTable:
    """Look up a normal shock table result from the upstream Mach number, M1

    Args:
        mach_upstream (ut.ndarray | float): upstream Mach number, M1
        specific_heat_ratio (ut.ndarray | float, optional): Ratio of specific
            heats, gamma. Defaults to 1.4.

    Returns:
        NormalShockTable: normal shock table result
    """
    m1 = np.atleast_1d(mach_upstream)
    gam = np.atleast_1d(specific_heat_ratio)
    p02_p01 = total_pressure_ratio_by_mach(
        mach_upstream=m1, specific_heat_ratio=gam
    )
    p01_p1 = isentropic_flow.total_pressure_ratio(
        mach=m1, specific_heat_ratio=gam)
    return NormalShockTable(
        mach_upstream=m1,
        mach_downstream=mach_downstream(
            mach_upstream=m1, specific_heat_ratio=gam
        ),
        temperature_ratio=temperature_ratio_by_upstream_mach(
            mach_upstream=m1, specific_heat_ratio=gam
        ),
        pressure_ratio=pressure_ratio(
            mach_upstream=m1, specific_heat_ratio=gam
        ),
        density_ratio=density_ratio(mach_upstream=m1, specific_heat_ratio=gam),
        total_pressure_ratio=p02_p01,
        pitot_pressure_ratio=p02_p01 * p01_p1,
        specific_heat_ratio=gam,
    )


def lookup_table_by_temperature(
    temperature_ratio: ut.ndarray | float,
    specific_heat_ratio: ut.ndarray | float = 1.4,
) -> NormalShockTable:
    """Look up a normal shock table result from the temperature ratio, T2/T1

    Args:
        temperature_ratio (ut.ndarray | float): temperature ratio, T2/T1
        specific_heat_ratio (ut.ndarray | float, optional): Ratio of specific
            heats, gamma. Defaults to 1.4.

    Returns:
        NormalShockTable: normal shock table result
    """
    t21 = np.atleast_1d(temperature_ratio)
    gam = (
        ut.scalar2array(specific_heat_ratio, t21.shape)
        if np.isscalar(specific_heat_ratio)
        else np.atleast_1d(specific_heat_ratio)
    )
    # invert the T2/T1 -> M1 relationship

    def tfunc(mguess: float, _t: float, _g: float):
        return _t - temperature_ratio_by_upstream_mach(
            mach_upstream=mguess, specific_heat_ratio=_g
        )

    m1 = np.array(
        [
            fsolve(tfunc, 2.0, args=(t, g))[0]
            for t, g in zip(t21.flat, gam.flat, strict=True)
        ]
    )
    return lookup_table_by_upstream_mach(
        mach_upstream=m1, specific_heat_ratio=gam
    )


def lookup_table_by_pressure(
    pressure_ratio: ut.ndarray | float,
    specific_heat_ratio: ut.ndarray | float = 1.4,
) -> NormalShockTable:
    """Look up a normal shock table result from the static pressure ratio, p2/p1

    Args:
        pressure_ratio (ut.ndarray | float): static pressure ratio, p2/p1
        specific_heat_ratio (ut.ndarray | float, optional): Ratio of specific
            heats, gamma. Defaults to 1.4.

    Returns:
        NormalShockTable: normal shock table result
    """
    p21 = np.atleast_1d(pressure_ratio)
    gam = np.atleast_1d(specific_heat_ratio)
    # invert the p2/p1 -> M1 relationship
    m1 = ((p21 - 1) * (gam + 1) / (2 * gam) + 1) ** 0.5
    return lookup_table_by_upstream_mach(
        mach_upstream=m1, specific_heat_ratio=gam
    )


def lookup_table_by_density(
    density_ratio: ut.ndarray | float,
    specific_heat_ratio: ut.ndarray | float = 1.4,
) -> NormalShockTable:
    """Look up a normal shock table result from the density ratio, rho2/rho1

    Args:
        density_ratio (ut.ndarray | float): density ratio, rho2/rho1
        specific_heat_ratio (ut.ndarray | float, optional): Ratio of specific
            heats, gamma. Defaults to 1.4.

    Returns:
        NormalShockTable: normal shock table result
    """
    r21 = np.atleast_1d(density_ratio)
    gam = np.atleast_1d(specific_heat_ratio)
    # invert the rho2/rho1 -> M1 relationship
    m1 = (2.0 / ((gam + 1) / r21 - gam + 1)) ** 0.5
    return lookup_table_by_upstream_mach(
        mach_upstream=m1, specific_heat_ratio=gam
    )


def lookup_table_by_total_pressure(
    total_pressure_ratio: ut.ndarray | float,
    specific_heat_ratio: ut.ndarray | float = 1.4,
) -> NormalShockTable:
    """Look up a normal shock table result from the total pressure ratio, p02/p01

    Args:
        total_pressure_ratio (ut.ndarray | float): total pressure ratio, p02/p01
        specific_heat_ratio (ut.ndarray | float, optional): Ratio of specific
            heats, gamma. Defaults to 1.4.

    Returns:
        NormalShockTable: normal shock table result
    """
    p02_p01 = np.atleast_1d(total_pressure_ratio)
    gam = (
        ut.scalar2array(specific_heat_ratio, p02_p01.shape)
        if np.isscalar(specific_heat_ratio)
        else np.atleast_1d(specific_heat_ratio)
    )
    # invert the p02/p01 -> M1 relationship

    def pfunc(mguess: float, _p021: float, _g: float):
        return _p021 - total_pressure_ratio_by_mach(
            mach_upstream=mguess, specific_heat_ratio=_g
        )

    m1 = np.array(
        [
            fsolve(pfunc, 2.0, args=(p, g))[0]
            for p, g in zip(p02_p01.flat, gam.flat, strict=True)
        ]
    )
    return lookup_table_by_upstream_mach(
        mach_upstream=m1, specific_heat_ratio=gam
    )


def lookup_table_by_pitot_pressure(
    pitot_pressure_ratio: ut.ndarray | float,
    specific_heat_ratio: ut.ndarray | float = 1.4,
) -> NormalShockTable:
    """Look up a normal shock table result from the Rayleigh Pitot tube pressure ratio, p02/p1

    Args:
        total_pressure_ratio (ut.ndarray | float): Rayleigh Pitot tube pressure ratio, p02/p1
        specific_heat_ratio (ut.ndarray | float, optional): Ratio of specific
            heats, gamma. Defaults to 1.4.

    Returns:
        NormalShockTable: normal shock table result
    """
    p02_p1 = np.atleast_1d(pitot_pressure_ratio)
    gam = (
        ut.scalar2array(specific_heat_ratio, p02_p1.shape)
        if np.isscalar(specific_heat_ratio)
        else np.atleast_1d(specific_heat_ratio)
    )
    # invert the p02/p1 -> M1 relationship

    def pfunc(mguess: float, _p021: float, _g: float):
        return _p021 - total_pressure_ratio_by_mach(
            mach_upstream=mguess, specific_heat_ratio=gam
        ) * isentropic_flow.total_pressure_ratio(
            mach=mguess, specific_heat_ratio=gam
        )

    m1 = np.array(
        [
            fsolve(pfunc, 2.0, args=(p, g))[0]
            for p, g in zip(p02_p1.flat, gam.flat, strict=True)
        ]
    )
    return lookup_table_by_upstream_mach(
        mach_upstream=m1, specific_heat_ratio=gam
    )


def lookup_table_by_downstream_mach(
    mach_downstream: ut.ndarray | float,
    specific_heat_ratio: ut.ndarray | float = 1.4,
) -> NormalShockTable:
    """Look up a normal shock table result from the downstream Mach number, M2

    Args:
        mach_downstream (ut.ndarray | float): downstream Mach number, M2
        specific_heat_ratio (ut.ndarray | float, optional): Ratio of specific
            heats, gamma. Defaults to 1.4.

    Returns:
        NormalShockTable: normal shock table result
    """
    m2 = np.atleast_1d(mach_downstream)
    gam = np.atleast_1d(specific_heat_ratio)
    # invert the M1 -> M2 relationship
    m1 = (
        (1 + m2**2 * 0.5 * (gam - 1)) / (gam * m2**2 - 0.5 * (gam - 1))
    ) ** 0.5
    return lookup_table_by_upstream_mach(
        mach_upstream=m1, specific_heat_ratio=gam
    )


def mach_downstream(
    mach_upstream: ut.ndarray | float, specific_heat_ratio: ut.ndarray | float
) -> ut.ndarray:
    """Computes the Mach number downstream of a normal shock, M2.

    This form is valid for calorically perfect gases only, where `gamma`
    is constant (M1 < 5).

    Args:
        mach_upstream (ut.ndarray | float): upstream Mach number, M1
        specific_heat_ratio (ut.ndarray | float): ratio of specific heats, gamma

    Returns:
        ut.ndarray: downstream Mach number, M2
    """
    m1 = np.atleast_1d(mach_upstream)
    gam = np.atleast_1d(specific_heat_ratio)
    return (
        (1 + 0.5 * (gam - 1) * m1**2) / (gam * m1**2 - 0.5 * (gam - 1))
    ) ** 0.5


def density_ratio(
    mach_upstream: ut.ndarray | float, specific_heat_ratio: ut.ndarray | float
) -> ut.ndarray:
    """Computes the density ratio across a normal shock, rho2/rho1.

    This form is valid for calorically perfect gases only, where `gamma`
    is constant (M1 < 5).

    The density ratio rho2/rho1 is equivalent to the inverse velocity ratio
    u1/u2

    Args:
        mach_upstream (ut.ndarray | float): upstream Mach number, M1
        specific_heat_ratio (ut.ndarray | float): ratio of specific heats, gamma

    Returns:
        ut.ndarray: density ratio, rho2/rho1
    """
    m1 = np.atleast_1d(mach_upstream)
    gam = np.atleast_1d(specific_heat_ratio)
    return (gam + 1) * m1**2 / (2 + (gam - 1) * m1**2)


def pressure_ratio(
    mach_upstream: ut.ndarray | float, specific_heat_ratio: ut.ndarray | float
) -> ut.ndarray:
    """Computes the static pressure ratio across a normal shock, p2/p1.

    This form is valid for calorically perfect gases only, where `gamma`
    is constant (M1 < 5).

    Args:
        mach_upstream (ut.ndarray | float): upstream Mach number, M1
        specific_heat_ratio (ut.ndarray | float): ratio of specific heats, gamma

    Returns:
        ut.ndarray: static pressure ratio, p2/p1
    """
    m1 = np.atleast_1d(mach_upstream)
    gam = np.atleast_1d(specific_heat_ratio)
    return 1 + 2 * gam / (gam + 1) * (m1**2 - 1)


def total_pressure_ratio_by_mach(
    mach_upstream: ut.ndarray | float, specific_heat_ratio: ut.ndarray | float
) -> ut.ndarray:
    """Computes the total pressure ratio across a normal shock, p02/p01.

    This form is valid for calorically perfect gases only, where `gamma`
    is constant (M1 < 5).

    Args:
        mach_upstream (ut.ndarray | float): upstream Mach number, M1
        specific_heat_ratio (ut.ndarray | float): ratio of specific heats, gamma

    Returns:
        ut.ndarray: total pressure ratio across shock, p02/p01
    """
    m1 = np.atleast_1d(mach_upstream)
    gam = np.atleast_1d(specific_heat_ratio)
    return ((gam + 1) / (2 * gam * m1**2 - gam + 1)) ** (1.0 / (gam - 1)) * (
        (gam + 1) * m1**2 / ((gam - 1) * m1**2 + 2)
    ) ** (gam / (gam - 1))


def temperature_ratio_by_upstream_mach(
    mach_upstream: ut.ndarray | float, specific_heat_ratio: ut.ndarray | float
) -> ut.ndarray:
    """Computes the temperature ratio across a normal shock, T2/T1.

    This form is valid for calorically perfect gases only, where `gamma`
    is constant (M1 < 5). This is also equivalent to the enthalpy ratio
    across the normal shock, h2/h1

    Args:
        mach_upstream (ut.ndarray | float): upstream Mach number, M1
        specific_heat_ratio (ut.ndarray | float): ratio of specific heats, gamma

    Returns:
        ut.ndarray: temperature ratio across shock, T2/T1
    """
    m1 = np.atleast_1d(mach_upstream)
    gam = np.atleast_1d(specific_heat_ratio)
    return (
        (1 + 2 * gam / (gam + 1) * (m1**2 - 1))
        * (2 + (gam - 1) * m1**2)
        / ((gam + 1) * m1**2)
    )


def entropy_change(
    total_pressure_ratio: ut.ndarray | float, gas_constant: ut.ndarray | float
) -> ut.ndarray:
    """Compute the change in specific entropy across a normal shock, s2-s1

    This form is valid for calorically and thermally perfect gases over
    a stationary shock.

    Args:
        total_pressure_ratio (ut.ndarray | float): total pressure ratio, p02/p01
        gas_constant (ut.ndarray | float): specific gas constant, R

    Returns:
        ut.ndarray: change in specific entropy, s2-s1
    """
    p02_p01 = np.atleast_1d(total_pressure_ratio)
    return -gas_constant * np.log(p02_p01)


def internal_energy_change(
    pressure_upstream: ut.ndarray | float,
    pressure_downstream: ut.ndarray | float,
    density_upstream: ut.ndarray | float,
    density_downstream: ut.ndarray | float,
) -> ut.ndarray:
    """Computes the change in specific internal energy across a normal shock.
    This is the Hugoniot relation.

    This relates only thermodynamic quantities across a normal shock.
    This relation is valid for perfect, chemically reacting, and real gases.

    Args:
        pressure_upstream (ut.ndarray | float): upstream static pressure
        pressure_downstream (ut.ndarray | float): downstream static pressure
        density_upstream (ut.ndarray | float): upstream static density
        density_downstream (ut.ndarray | float): downstream static density

    Returns:
        ut.ndarray: change in specific internal energy
    """
    p1 = np.atleast_1d(pressure_upstream)
    # compute specific volumes
    v1 = 1.0 / density_upstream
    v2 = 1.0 / density_downstream
    return 0.5 * (p1 + pressure_downstream) * (v1 - v2)
