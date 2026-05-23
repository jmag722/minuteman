"""
Compressible, inviscid flow relations.

These functions are basic definitions pertaining to 1-,2-, and 3-D
compressible, inviscid flow.

Total or stagnation quantities (total temperature, for instance) are the
values that would exist if the flow were isentropically brought to rest.
Total quantities remain constant where flowfield is isentropic
(adiabatic and reversible).
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import fsolve

import minuteman.thermodynamics.caloric_perfect as calp
import minuteman.utils.types as ut


def mach_number(
    velocity: ut.ndarray | float, speed_of_sound: ut.ndarray | float
) -> ut.ndarray:
    """Compute Mach number

    Args:
        velocity (ut.ndarray | float): velocity
        speed_of_sound (ut.ndarray | float): speed of sound

    Returns:
        ut.ndarray: Mach number
    """
    return np.atleast_1d(velocity) / speed_of_sound


def speed_of_sound_from_temperature(
    gas_constant: ut.ndarray | float,
    temperature: ut.ndarray | float,
    specific_heat_ratio: ut.ndarray | float,
) -> ut.ndarray:
    """Compute the speed of sound from the gas constant (R) and temperature (T)

    Args:
        gas_constant (ut.ndarray | float): gas constant
        temperature (ut.ndarray | float): temperature
        specific_heat_ratio (ut.ndarray | float): ratio of specific heats

    Returns:
        ut.ndarray: speed of sound
    """
    gam = np.atleast_1d(specific_heat_ratio)
    return (gam * gas_constant * temperature) ** 0.5


def speed_of_sound_from_pressure(
    pressure: ut.ndarray | float,
    density: ut.ndarray | float,
    specific_heat_ratio: ut.ndarray | float,
) -> ut.ndarray:
    """Compute the speed of sound from the pressure (p) and density (r)

    Args:
        pressure (ut.ndarray | float): pressure
        density (ut.ndarray | float): density
        specific_heat_ratio (ut.ndarray | float): ratio of specific heats

    Returns:
        ut.ndarray: speed of sound
    """
    gam = np.atleast_1d(specific_heat_ratio)
    return (gam * pressure / density) ** 0.5


@dataclass
class IsentropicFlowTable:
    """Isentropic flow table, containing various isentropic parameters
    for a given Mach number and specific heat ratio
    """

    mach: ut.ndarray
    """Mach number"""
    temperature: ut.ndarray
    """total temperature ratio, T0/T"""
    pressure: ut.ndarray
    """total pressure ratio, p0/p"""
    density: ut.ndarray
    """total density ratio, rho0/rho"""
    speed_of_sound: ut.ndarray
    """total speed of sound ratio, a0/a"""
    area_ratio: ut.ndarray
    """area ratio, A/A*"""
    specific_heat_ratio: ut.ndarray
    """ratio of specific heats, gamma"""


def lookup_table_by_mach(
    mach: ut.ndarray | float, specific_heat_ratio: ut.ndarray | float
) -> IsentropicFlowTable:
    """Lookup the isentropic flow table based on Mach number

    Args:
        mach (ut.ndarray | float): Mach number, M
        specific_heat_ratio (ut.ndarray | float): ratio of specific heats, gamma

    Returns:
        IsentropicFlowTable: isentropic flow table result
    """
    gam = np.atleast_1d(specific_heat_ratio)
    p0_ratio = total_pressure_ratio(mach=mach, specific_heat_ratio=gam)
    r0_ratio = total_density_ratio(mach=mach, specific_heat_ratio=gam)
    t0_ratio = total_temperature_ratio(mach=mach, specific_heat_ratio=gam)
    a0_ratio = total_speed_of_sound_ratio(mach=mach, specific_heat_ratio=gam)
    area_ratio = area_mach_relation(mach=mach, specific_heat_ratio=gam)
    return IsentropicFlowTable(
        mach=mach,
        temperature=t0_ratio,
        pressure=p0_ratio,
        density=r0_ratio,
        speed_of_sound=a0_ratio,
        area_ratio=area_ratio,
        specific_heat_ratio=gam,
    )


def lookup_table_by_temperature(
    temperature_ratio: ut.ndarray | float,
    specific_heat_ratio: ut.ndarray | float,
) -> IsentropicFlowTable:
    """Lookup the isentropic flow table based on total temperature ratio

    Args:
        temperature_ratio (ut.ndarray | float): total temperature ratio, T0/T
        specific_heat_ratio (ut.ndarray | float): ratio of specific heats, gamma

    Returns:
        IsentropicFlowTable: isentropic flow table result
    """
    gam = np.atleast_1d(specific_heat_ratio)
    mach = mach_from_temperature(
        temperature_ratio=temperature_ratio, specific_heat_ratio=gam
    )
    return lookup_table_by_mach(mach=mach, specific_heat_ratio=gam)


def lookup_table_by_pressure(
    pressure_ratio: ut.ndarray | float, specific_heat_ratio: ut.ndarray | float
) -> IsentropicFlowTable:
    """Lookup the isentropic flow table based on total pressure ratio

    Args:
        pressure_ratio (ut.ndarray | float): total pressure ratio, p0/p
        specific_heat_ratio (ut.ndarray | float): ratio of specific heats, gamma

    Returns:
        IsentropicFlowTable: isentropic flow table result
    """
    gam = np.atleast_1d(specific_heat_ratio)
    t0_ratio = calp.isentropic_process_from_pressure(
        pressure_ratio=pressure_ratio, specific_heat_ratio=gam
    ).temperature_ratio
    return lookup_table_by_temperature(
        temperature_ratio=t0_ratio, specific_heat_ratio=specific_heat_ratio
    )


def lookup_table_by_density(
    density_ratio: ut.ndarray | float, specific_heat_ratio: ut.ndarray | float
) -> IsentropicFlowTable:
    """Lookup the isentropic flow table based on total density ratio

    Args:
        density_ratio (ut.ndarray | float): total density ratio, rho0/rho
        specific_heat_ratio (ut.ndarray | float): ratio of specific heats, gamma

    Returns:
        IsentropicFlowTable: isentropic flow table result
    """
    gam = np.atleast_1d(specific_heat_ratio)
    t0_ratio = calp.isentropic_process_from_density(
        density_ratio=density_ratio, specific_heat_ratio=gam
    ).temperature_ratio
    return lookup_table_by_temperature(
        temperature_ratio=t0_ratio, specific_heat_ratio=specific_heat_ratio
    )


def lookup_table_by_speed_of_sound(
    speed_of_sound_ratio: ut.ndarray | float,
    specific_heat_ratio: ut.ndarray | float,
) -> IsentropicFlowTable:
    """Lookup the isentropic flow table based on total speed of sound ratio

    Args:
        speed_of_sound_ratio (ut.ndarray | float): total speed of sound ratio,
            a0/a
        specific_heat_ratio (ut.ndarray | float): ratio of specific heats, gamma

    Returns:
        IsentropicFlowTable: isentropic flow table result
    """
    gam = np.atleast_1d(specific_heat_ratio)
    t0_ratio = calp.isentropic_process_from_speed_of_sound(
        speed_of_sound_ratio=speed_of_sound_ratio, specific_heat_ratio=gam
    ).temperature_ratio
    return lookup_table_by_temperature(
        temperature_ratio=t0_ratio, specific_heat_ratio=specific_heat_ratio
    )


def lookup_table_by_area_ratio(
    area_ratio: ut.ndarray | float,
    specific_heat_ratio: ut.ndarray | float,
    mach_guess: ut.ndarray | float,
) -> IsentropicFlowTable:
    """Lookup the isentropic flow table based on area ratio

    Args:
        area_ratio (ut.ndarray | float): area ratio, A/A*
        specific_heat_ratio (ut.ndarray | float): ratio of specific heats, gamma
        mach_guess (ut.ndarray | float): Guess for the Mach number.
            For supersonic flow: choose value > 1 (2.0 is good)
            For subsonic flow: choose value > 1 (0.2 is good)

    Returns:
        IsentropicFlowTable: isentropic flow table result
    """
    aratios = np.atleast_1d(area_ratio)
    mguess = (
        np.zeros_like(area_ratio) + mach_guess
        if np.isscalar(mach_guess)
        else mach_guess
    )
    gam = (
        np.zeros_like(area_ratio) + specific_heat_ratio
        if np.isscalar(specific_heat_ratio)
        else specific_heat_ratio
    )

    mach = np.empty_like(aratios)
    for i in range(aratios.size):
        mach.flat[i] = mach_from_area_ratio(
            area_ratio=aratios.flat[i],
            specific_heat_ratio=gam.flat[i],
            mach_guess=mguess.flat[i],
        )
    return lookup_table_by_mach(mach=mach, specific_heat_ratio=gam)


def total_temperature_ratio(
    mach: ut.ndarray | float, specific_heat_ratio: ut.ndarray | float
) -> ut.ndarray:
    """Computes the stagnation or total temperature ratio, T0/T

    Args:
        mach (ut.ndarray | float): Mach number
        specific_heat_ratio (ut.ndarray | float): ratio of specific heats

    Returns:
        ut.ndarray: total temperature ratio
    """
    gam = np.atleast_1d(specific_heat_ratio)
    return 1 + 0.5 * (gam - 1) * mach**2


def mach_from_temperature(
    temperature_ratio: ut.ndarray | float,
    specific_heat_ratio: ut.ndarray | float,
) -> ut.ndarray:
    """Computes the Mach number from total temperature ratio T0/T

    Args:
        temperature_ratio (ut.ndarray | float): total temperature ratio
        specific_heat_ratio (ut.ndarray | float): ratio of specific heats

    Returns:
        ut.ndarray | float: _description_
    """
    gam = np.atleast_1d(specific_heat_ratio)
    return (2.0 / (gam - 1) * (temperature_ratio - 1)) ** 0.5


def total_pressure_ratio(
    mach: ut.ndarray | float, specific_heat_ratio: ut.ndarray | float
) -> ut.ndarray:
    """Computes the stagnation or total pressure ratio, p0/p

    Args:
        mach (ut.ndarray | float): Mach number
        specific_heat_ratio (ut.ndarray | float): ratio of specific heats

    Returns:
        ut.ndarray: total pressure ratio
    """
    gam = np.atleast_1d(specific_heat_ratio)
    return total_temperature_ratio(mach=mach, specific_heat_ratio=gam) ** (
        gam / (gam - 1)
    )


def total_density_ratio(
    mach: ut.ndarray | float, specific_heat_ratio: ut.ndarray | float
) -> ut.ndarray:
    """Computes the stagnation or total density ratio, rho0/rho

    Args:
        mach (ut.ndarray | float): Mach number
        specific_heat_ratio (ut.ndarray | float): ratio of specific heats

    Returns:
        ut.ndarray: total density ratio
    """
    gam = np.atleast_1d(specific_heat_ratio)
    return total_temperature_ratio(mach=mach, specific_heat_ratio=gam) ** (
        1.0 / (gam - 1)
    )


def total_speed_of_sound_ratio(
    mach: ut.ndarray | float, specific_heat_ratio: ut.ndarray | float
) -> ut.ndarray:
    """Computes the stagnation or total speed of sound ratio, a0/a

    Args:
        mach (ut.ndarray | float): Mach number
        specific_heat_ratio (ut.ndarray | float): ratio of specific heats

    Returns:
        ut.ndarray: total speed of sound ratio
    """
    gam = np.atleast_1d(specific_heat_ratio)
    return (total_temperature_ratio(mach=mach, specific_heat_ratio=gam)) ** 0.5


def _area_mach_relation_sqr(m, gam):
    return m**-2 * (
        2.0
        / (gam + 1)
        * total_temperature_ratio(mach=m, specific_heat_ratio=gam)
    ) ** ((gam + 1) / (gam - 1))


def area_mach_relation(
    mach: ut.ndarray | float, specific_heat_ratio: ut.ndarray | float
) -> ut.ndarray:
    """Computes the area ratio A/A* for an isentropic nozzle.

    This is the standard area-Mach number relation

    Args:
        mach (ut.ndarray | float): Mach number
        specific_heat_ratio (ut.ndarray | float): ratio of specific heats

    Returns:
        ut.ndarray: area ratio A/A*
    """
    return _area_mach_relation_sqr(m=mach, gam=specific_heat_ratio) ** 0.5


def mach_from_area_ratio(
    area_ratio: float, specific_heat_ratio: float, mach_guess: float
) -> float:
    """Compute the Mach number for a known area ratio, A/A*

    Args:
        area_ratio (float): area ratio
        specific_heat_ratio (float): ratio of specific heats
        mach_guess (float): Guess for the Mach number.
            A value above unity is suitable for supersonic Mach (2.0 is good).
            If a subsonic Mach solution is desired, set to a value below
            unity (0.2 is a good default)

    Returns:
        float: supersonic or subsonic Mach solution for a given area ratio
    """

    def compute_mach(m, aratio, gam):
        return aratio**2 - _area_mach_relation_sqr(m=m, gam=gam)

    return fsolve(compute_mach, mach_guess, (area_ratio, specific_heat_ratio))[
        0
    ]
