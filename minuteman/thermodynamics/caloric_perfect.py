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
        gas_constant: ut.ndarray | float) -> ut.ndarray | float:
    """Computes the specific heat at constant pressure.

    Valid for perfect (thermally & calorically) gases.

    Args:
        specific_heat_ratio (ndarray | float): ratio of specific heats (gamma)
        gas_constant (ndarray | float): specific gas constant

    Returns:
        ndarray | float: specific heat at constant pressure
    """
    return specific_heat_ratio * specific_heat_constant_volume(
        specific_heat_ratio=specific_heat_ratio,
        gas_constant=gas_constant
    )


def specific_heat_constant_volume(
        specific_heat_ratio: ut.ndarray | float,
        gas_constant: ut.ndarray | float) -> ut.ndarray | float:
    """Computes the specific heat at constant volume.

    Valid for perfect (thermally & calorically) gases.

    Args:
        specific_heat_ratio (ndarray | float): ratio of specific heats
        gas_constant (ndarray | float): specific gas constant

    Returns:
        ndarray | float: specific heat at constant volume
    """
    return gas_constant / (specific_heat_ratio - 1.0)


def entropy_state(p, rho, gam, R):
    """
    Compute entropy state of a calorically perfect gas.

    Parameters
    ----------
    p : float | ArrayLike
        pressure
    rho : float | ArrayLike
        density
    gam : float | ArrayLike
        ratio of specific heats
    R : float | ArrayLike
        gas constant

    Returns
    -------
    float | ArrayLike
        entropy
    """
    return np.log(p / rho**gam) * specific_heat_constant_volume(
        specific_heat_ratio=gam, gas_constant=R)


def entropy(t21: float = None, p21: float = None, v21: float = None,
            cp: float = None, cv: float = None, R: float = gas_constant_air_si, s1: float = 0.0):
    """
    Computes the change in entropy for a calorically perfect gas.

    Parameters
    ----------
    t21 : float, optional
        temperature ratio T2/T1, by default None
    p21 : float, optional
        pressure ratio p2/p1, by default None
    v21 : float, optional
        specific volume ratio v2/v1, by default None
    cp : float, optional
        specific heat (constant pressure), by default None
    cv : float, optional
        specific heat (constant volume), by default None
    R : float, optional
        specific gas constant, by default gas_constant_air_si
    s1 : float, optional
        entropy at state 1, by default 0.0

    Returns
    -------
    float
        entropy change ds if s1==0.0 else entropy at state 2 (s2)

    Raises
    ------
    ValueError
        Incorrect or insufficient inputs supplied
    """
    if all(var is not None for var in [t21, p21, cp, R]):
        expr = cp*np.log(t21) - R * np.log(p21)
    elif all(var is not None for var in [t21, v21, cv, R]):
        expr = cv*np.log(t21) + R * np.log(v21)
    elif all(var is not None for var in [p21, v21, cp, cv]):
        expr = cv*np.log(p21) + cp*np.log(v21)
    else:
        error_msg = "Must specify one of the following: \n\
                1) T2/T1, v2/v1, cp, R, OR\n\
                2) T2/T1, v2/v1, cv, R, OR\n\
                3) p2/p1, v2/v1, cp, cv"
        raise ValueError(error_msg)
    return s1 + expr  # if s1==0, returns (s2-s1), else s2


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


def total_energy(p, rho, v, gam=1.4):
    """
    Compute total energy, in units of energy/unit volume.

    Parameters
    ----------
    p : float | ArrayLike
        pressure [Pa]
    rho : float | ArrayLike
        density [kg/m3]
    v : float | ArrayLike
        speed
    gam : float, float | ArrayLike
        ratio of specific heats, by default 1.4

    Returns
    -------
    float | ArrayLike
        total energy in volumetric units
    """
    return p/(gam-1) + 0.5*rho*v**2


def specific_enthalpy(e, p, rho):
    """
    Compute specific enthalpy, in units of energy per unit mass

    Parameters
    ----------
    e : float | ArrayLike
        specific internal energy [energy/mass]
    p : float | ArrayLike
        pressure
    rho : float | ArrayLike
        density [mass/volume]

    Returns
    -------
    float | ArrayLike
        specific enthalpy
    """
    return e + p/rho
