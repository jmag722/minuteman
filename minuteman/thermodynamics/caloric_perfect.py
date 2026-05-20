"""
Calorically perfect gases are those where gases are chemically unreactive
and intermolecular forces are neglected. Internal energy and enthalpy are
functions of temperature only AND the specific heats are constant.

This is the case for atmospheric air below ~1000 K. However, at higher
temperatures where O2 and N2 vibrational motion/excitation becomes
important, the gas is no longer calorically perfect.
"""

import numpy as np
import scipy.constants as scc

import minuteman.utils.types as ut

avogadro = scc.Avogadro
"""Avogadro constant [#/mol]"""

boltzmann_si = scc.Boltzmann  # J/K, Boltzmann constant
"""Boltzmann constant in SI units [J/K]"""

boltzmann_imperial = (
    scc.Boltzmann /
    (scc.foot * scc.pound_force * scc.convert_temperature(1.0, "K", "R"))
)
"""Boltzmann constant in Imperial units [ft-lb/R]"""

universal_gas_constant_si = scc.R * scc.kilo
"""Universal gas constant in SI units [J/(kg-mol K)]"""

universal_gas_constant_imperial_lbm = (
    scc.R * scc.kilo * scc.pound /
    (scc.foot * scc.pound_force * scc.convert_temperature(1.0, "K", "R"))
)
"""Universal gas constant in Imperial units [ft-lbf/(lbm-mol R)]"""

universal_gas_constant_imperial_slug = (
    scc.R * scc.kilo * scc.slug /
    (scc.foot * scc.pound_force * scc.convert_temperature(1.0, "K", "R"))
)
"""Universal gas constant in Imperial units [ft-lbf/(slug-mol R)]"""

molecular_weight_air = 28.9647
"""Molecular weight for dry air [kg/kg-mol]"""
# https://www.engineeringtoolbox.com/molecular-mass-air-d_679.html

gas_constant_air_si = universal_gas_constant_si / molecular_weight_air
"""Specific gas constant for air in SI units [J/(kg K)]"""

gas_constant_air_imperial_lbm = (
    universal_gas_constant_imperial_lbm / molecular_weight_air
)
"""Specific gas constant for air in Imperial units [ft-lbf/(lbm R)]"""

gas_constant_air_imperial_slug = (
    universal_gas_constant_imperial_slug / molecular_weight_air
)
"""Specific gas constant for air in Imperial units [ft-lbf/(slug R)]"""


def specific_heat_constant_pressure(
        specific_heat_ratio: ut.ndarray | float,
        gas_constant: ut.ndarray | float) -> ut.ndarray:
    """Computes the specific heat at constant pressure.

    Valid for perfect (thermally & calorically) gases.

    Args:
        specific_heat_ratio (ndarray | float): ratio of specific heats (gamma)
        gas_constant (ndarray | float): specific gas constant

    Returns:
        ndarray: specific heat at constant pressure
    """
    gam = np.atleast_1d(specific_heat_ratio)
    return gam * specific_heat_constant_volume(
        specific_heat_ratio=gam,
        gas_constant=gas_constant,
    )


def specific_heat_constant_volume(
        specific_heat_ratio: ut.ndarray | float,
        gas_constant: ut.ndarray | float) -> ut.ndarray:
    """Computes the specific heat at constant volume.

    Valid for perfect (thermally & calorically) gases.

    Args:
        specific_heat_ratio (ndarray | float): ratio of specific heats
        gas_constant (ndarray | float): specific gas constant

    Returns:
        ndarray: specific heat at constant volume
    """
    gam = np.atleast_1d(specific_heat_ratio)
    return gas_constant / (gam - 1.0)


def entropy_state(pressure: ut.ndarray | float,
                  density: ut.ndarray | float,
                  specific_heat_ratio: ut.ndarray | float,
                  gas_constant: ut.ndarray | float) -> ut.ndarray:
    """Compute entropy state of a calorically perfect gas.

    Args:
        pressure (ut.ndarray | float): pressure
        density (ut.ndarray | float): density
        specific_heat_ratio (ut.ndarray | float): ratio of specific heats
        gas_constant (ut.ndarray | float): specific gas constant

    Returns:
        ut.ndarray: entropy
    """
    gam = np.atleast_1d(specific_heat_ratio)
    return np.log(pressure / density**gam) * specific_heat_constant_volume(
        specific_heat_ratio=gam, gas_constant=gas_constant)


def entropy_change_tp(temperature_ratio: ut.ndarray | float,
                      pressure_ratio: ut.ndarray | float,
                      specific_heat_constant_pressure: ut.ndarray | float,
                      gas_constant: ut.ndarray | float) -> ut.ndarray:
    """Compute the change in entropy for a known change in temperature and
    pressure.

    Args:
        temperature_ratio (ut.ndarray | float): temperature ratio, T2/T1
        pressure_ratio (ut.ndarray | float): pressure ratio, p2/p1
        specific_heat_constant_pressure (ut.ndarray | float): specific heat at
            constant pressure, cp
        gas_constant (ut.ndarray | float): specific gas constant R

    Returns:
        ut.ndarray: change in entropy, s2-s1
    """
    return (
        specific_heat_constant_pressure *
        np.log(temperature_ratio) - gas_constant * np.log(pressure_ratio)
    )


def entropy_change_tv(temperature_ratio: ut.ndarray | float,
                      specific_volume_ratio: ut.ndarray | float,
                      specific_heat_constant_volume: ut.ndarray | float,
                      gas_constant: ut.ndarray | float) -> ut.ndarray:
    """Compute the change in entropy for a known change in temperature and
    specific volume.

    Args:
        temperature_ratio (ut.ndarray | float): temperature ratio, T2/T1
        specific_volume_ratio (ut.ndarray | float): specific volume ratio, v2/v1
        specific_heat_constant_volume (ut.ndarray | float): specific heat at
            constant volume, cv
        gas_constant (ut.ndarray | float): specific gas constant, R

    Returns:
        ut.ndarray: change in entropy, s2-s1
    """
    return (
        specific_heat_constant_volume *
        np.log(temperature_ratio) + gas_constant *
        np.log(specific_volume_ratio)
    )


def entropy_change_pv(
        pressure_ratio: ut.ndarray | float,
        specific_volume_ratio: ut.ndarray | float,
        specific_heat_constant_pressure: ut.ndarray | float,
        specific_heat_constant_volume: ut.ndarray | float) -> ut.ndarray:
    """Compute the change in entropy for a known change in pressure and
    specific volume.

    Args:
        pressure_ratio (ut.ndarray | float): pressure ratio, p2/p1
        specific_volume_ratio (ut.ndarray | float): specific volume ratio, v2/v1
        specific_heat_constant_pressure (ut.ndarray | float): specific heat
            of constant pressure, cp
        specific_heat_constant_volume (ut.ndarray | float): specific heat
            of constant volume, cv

    Returns:
        ut.ndarray: change in entropy, s2-s1
    """
    return (
        specific_heat_constant_volume *
        np.log(pressure_ratio) + specific_heat_constant_pressure *
        np.log(specific_volume_ratio)
    )


def isentropic_process(p21: float = None, t21: float = None, r21: float = None,
                       a21: float = None, gam: float = 1.4):
    """
    Returns the state of an isentropic process.

    Parameters
    ----------
    p21 : float, optional
        pressure ratio p2/p1, by default None
    t21 : float, optional
        temperature ratio T2/T1, by default None
    r21 : float, optional
        density ratio rho2/rho1, by default None
    a21 : float, optional
        speed of sound ratio a2/a1, by default None
    gam : float, optional
        ratio of specific heats gam, by default 1.4

    Returns
    -------
    dict
        table of p2/p1, t2/t1, rho2/rho1, a2/a1, and gam

    Raises
    ------
    ValueError
        Incorrect or insufficient inputs supplied.
    """
    if p21 is not None:
        t21 = p21**((gam-1)/gam)
        r21 = p21**(1/gam)
        a21 = p21**((gam-1)/gam/2)
    elif t21 is not None:
        p21 = t21**(gam/(gam-1))
        r21 = t21**(1/(gam-1))
        a21 = t21**0.5
    elif r21 is not None:
        p21 = r21**gam
        t21 = r21**(gam-1)
        a21 = r21**((gam-1)/2)
    elif a21 is not None:
        p21 = a21**(2*gam/(gam-1))
        t21 = a21*a21
        r21 = a21**(2/(gam-1))
    else:
        raise ValueError("Supply either p2/p1, T2/T1, rho2/rho1, or a2/a1.")
    return {"p21": p21, "t21": t21, "r21": r21, "a21": a21, "gam": gam}


def total_energy(pressure: ut.ndarray | float,
                 density: ut.ndarray | float,
                 speed: ut.ndarray | float,
                 specific_heat_ratio: ut.ndarray | float) -> ut.ndarray:
    """Compute total energy per unit volume

    Args:
        pressure (ut.ndarray | float): pressure
        density (ut.ndarray | float): density
        speed (ut.ndarray | float): velocity magnitude (sign irrelevant)
        specific_heat_ratio (ut.ndarray | float): ratio of specific heats

    Returns:
        ut.ndarray: total energy per unit volume
    """
    pressure = np.atleast_1d(pressure)
    return pressure / (specific_heat_ratio - 1) + 0.5 * density * speed**2


def specific_enthalpy(
        specific_internal_energy: ut.ndarray | float,
        pressure: ut.ndarray | float,
        density: ut.ndarray | float) -> ut.ndarray:
    """Compute specific enthalpy (per unit mass)

    Args:
        specific_internal_energy (ut.ndarray | float): specific internal
            energy
        pressure (ut.ndarray | float): pressure
        density (ut.ndarray | float): density

    Returns:
        ut.ndarray: specific enthalpy
    """
    e = np.atleast_1d(specific_internal_energy)
    return e + pressure / density
