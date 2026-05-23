"""
This module computes 1D, calorically perfect flow with heat addition (Rayleigh flow).
Online calculator for comparison at: http://www.dept.aoe.vt.edu/~devenpor/aoe3114/calc.html
Additional notes: https://kyleniemeyer.github.io/gas-dynamics-notes/compressible-flows/heat-transfer.html
"""

import numpy as np
from scipy.optimize import fsolve

import minuteman.cpg.isentropic_flow as isentropic_flow
import minuteman.utils.arg_checks as ac


def lookup_table(p21: float = None, r21: float = None, T21: float = None,
                 ds21_R: float = None, p02_p01: float = None, T02_T01: float = None,
                 M2: float = None, is_supersonic: bool = True, M1: float = 1.0,
                 gam: float = 1.4):
    """
    Computes Rayleigh flow parameters for a given input.

    This form is valid for calorically perfect gases only, where gamma
    is constant (M1 < 5).

    Parameters
    ----------
    p21 : float, optional
        pressure ratio p2/p1 and p/p* if `M1`==1.0, by default None
    r21 : float, optional
        density ratio rho2/rho1 and rho/rho* if `M1`==1.0, by default None
    T21 : float, optional
        temperature ratio T2/T1 and T/T* if `M1`==1.0, by default None
    ds21_R : float, optional
        entropy change (s2-s1)/R and (s-s*)/R if `M1`==1.0, by default None
    p02_p01 : float, optional
        total pressure ratio p02/p01 and p0/p0* if `M1`==1.0, by default None
    T02_T01 : float, optional
        total pressure ratio T02/T01 and T0/T0* if `M1`==1.0, by default None
    M2 : float, optional
        Mach number at station 2 of Rayleigh flow and M if `M1`==1.0, by default None
    is_supersonic : bool, optional
        is flow supersonic or subsonic. Must be specified for all inputs
        except p2/p1 and rho2/rho1, by default True
    M1 : float, optional
        Mach number at station 1 of Rayleigh flow, by default 1.0
    gam : float, optional
        ratio of specific heats gamma, by default 1.4

    Returns
    -------
    dict
        Rayleigh flow table parameters

    Raises
    ------
    ValueError
        Inconsistent or incorrect inputs supplied
    """
    if ac.is1known(M2, [p21, r21, T21, ds21_R, p02_p01, T02_T01]):
        return {
            "M1": M1,
            "M2": M2,
            "p21": pressure2(M2=M2, M1=M1, p1=1.0, gam=gam),
            "r21": density2(M2=M2, M1=M1, rho1=1.0, gam=gam),
            "T21": temperature2(M2=M2, M1=M1, T1=1.0, gam=gam),
            "ds21_R": entropy2(M2=M2, M1=M1, R=1.0, s1=0.0, gam=gam),
            "p02_p01": total_pressure2(M2=M2, M1=M1, p01=1.0, gam=gam),
            "T02_T01": total_temperature2(M2=M2, M1=M1, T01=1.0, gam=gam)
        }

    elif ac.is1known(p21, [M2, r21, T21, ds21_R, p02_p01, T02_T01]):
        M2 = mach2(M1=M1, p21=p21, gam=gam, is_supersonic=is_supersonic)

    elif ac.is1known(r21, [p21, M2, T21, ds21_R, p02_p01, T02_T01]):
        M2 = mach2(M1=M1, r21=r21, gam=gam, is_supersonic=is_supersonic)

    elif ac.is1known(T21, [p21, r21, M2, ds21_R, p02_p01, T02_T01]):
        M2 = mach2(M1=M1, T21=T21, gam=gam, is_supersonic=is_supersonic)

    elif ac.is1known(ds21_R, [p21, r21, T21, M2, p02_p01, T02_T01]):
        M2 = mach2(M1=M1, ds21_R=ds21_R, gam=gam, is_supersonic=is_supersonic)

    elif ac.is1known(p02_p01, [p21, r21, T21, ds21_R, M2, T02_T01]):
        M2 = mach2(M1=M1, p02_p01=p02_p01, gam=gam,
                   is_supersonic=is_supersonic)

    elif ac.is1known(T02_T01, [p21, r21, T21, ds21_R, p02_p01, M2]):
        M2 = mach2(M1=M1, T02_T01=T02_T01, gam=gam,
                   is_supersonic=is_supersonic)

    else:
        raise ValueError("Specify only p2/p1, rho2/rho1, T2/T1, p02/p01, "
                         "T02/T01, or M2.")
    return lookup_table(M2=M2, M1=M1, gam=gam)


def mach2(M1: float = 1.0, is_supersonic: bool = True, p21: float = None,
          r21: float = None, T21: float = None, ds21_R: float = None,
          p02_p01: float = None, T02_T01: float = None, gam: float = 1.4):
    """
    Computes the Mach number at region 2 for Rayleigh flow.

    This form is valid for calorically perfect gases only, where gamma
    is constant (M1 < 5).

    Parameters
    ----------
    M1 : float, optional
        Mach number for region 1 of Rayleigh flow, by default 1.0
    is_supersonic : bool, optional
        is flow supersonic or subsonic. Must be specified for all inputs
        except `p21` and `r21`, by default True
    p21 : float, optional
        p2/p1, and p/p* if `M1`=1.0, by default None
    r21 : float, optional
        rho2/rho1, and rho/rho* if `M1`=1.0, by default None
    T21 : float, optional
        T2/T1, and T/T* if `M1`=1.0, by default None
    ds21_R : float, optional
        (s2-s1)/R and (s-s*)/R if `M1`=1.0, by default None
    p02_p01 : float, optional
        p02/p01, and p0/p0* if `M1`=1.0, by default None
    T02_T01 : float, optional
        T02/T01 and T0/T0* if `M1`=1.0, by default None
    gam : float, optional
        ratio of specific heats gamma, by default 1.4

    Returns
    -------
    float
        Mach number for region 2 of heat addition AND

        mach number M if `M1`=`M*`=1.0

    Raises
    ------
    ValueError
        Incorrect or inconsistent inputs supplied
    """
    M2_guess = 3.0 if is_supersonic else 0.5

    if ac.is1known(p21, [r21, T21, ds21_R, p02_p01, T02_T01]):
        return (((1+gam*M1*M1)/p21 - 1)/gam)**0.5

    elif ac.is1known(T21, [r21, p21, ds21_R, p02_p01, T02_T01]):
        def func(M2, M1, T21, gam): return (
            T21 - temperature2(M2=M2, M1=M1, T1=1.0, gam=gam)
        )
        return fsolve(func, M2_guess, args=(M1, T21, gam))[0]

    elif ac.is1known(r21, [p21, T21, ds21_R, p02_p01, T02_T01]):
        return ((r21*(1+gam*M1*M1))/(M1*M1) - gam)**(-0.5)

    elif ac.is1known(ds21_R, [r21, T21, p21, p02_p01, T02_T01]):
        def func(M2, M1, ds21_R, gam): return (
            ds21_R - entropy2(M2=M2, M1=M1, R=1.0, s1=0.0, gam=gam)
        )
        return fsolve(func, M2_guess, args=(M1, ds21_R, gam))[0]

    elif ac.is1known(p02_p01, [r21, T21, ds21_R, p21, T02_T01]):
        def func(M2, M1, p02_p01, gam): return (
            p02_p01 - total_pressure2(M2=M2, M1=M1, p01=1.0, gam=gam)
        )
        return fsolve(func, M2_guess, args=(M1, p02_p01, gam))[0]

    elif ac.is1known(T02_T01, [r21, T21, ds21_R, p02_p01, p21]):
        def func(M2, M1, T02_T01, gam): return (
            T02_T01 - total_temperature2(M2=M2, M1=M1, T01=1.0, gam=gam)
        )
        return fsolve(func, M2_guess, args=(M1, T02_T01, gam))[0]

    else:
        raise ValueError("Specify exactly 1 of the following: p21, "
                         "T21, r21, ds21_R, T02_T01, or p02_p01.")


def entropy2(M2, M1=1.0, R: float = 1.0, s1=0.0, gam: float = 1.4):
    """
    `entropy2` computes the change in entropy for Rayleigh flow.

    This form is valid for calorically perfect gases only, where `gamma`
    is constant (M1 < 5).

    Parameters:
    M2 (float): Mach number for region 2 of Rayleigh flow.
    M1 (float): Mach number for region 1 of Rayleigh flow.
    R (float): specific gas constant.
    s1 (float): entropy for region 1 of Rayleigh flow.
    gam (float): gamma, ratio of specific heats.

    Returns:
    s2: entropy for region 2 of heat addition.
    ds: change in entropy when `s1`==0.0
    ds/R: change in entropy for specific gas constant when `s1`==0.0 and `R`=1.0
    """
    return (
        s1 + R * np.log(
            ((1+gam*M1*M1)/(1+gam*M2*M2))**(gam+1) * (M2*M2/(M1*M1))**(gam)
        ) / (gam-1)
    )


def total_pressure2(M2, M1=1.0, p01=1.0, gam: float = 1.4):
    """
    `total_pressure2` computes the total pressure `p02` for Rayleigh flow.

    This form is valid for calorically perfect gases only, where `gamma`
    is constant (M1 < 5).

    Parameters:
    M2 (float): Mach number for region 2 of Rayleigh flow.
    M1 (float): Mach number for region 1 of Rayleigh flow.
    p01 (float): total pressure of region 1. Equal to the sonic
        reference total pressure p0* if `M1`==1.0
    gam (float): gamma, ratio of specific heats.

    Returns:
    p02: total pressure for region 2 of heat addition.
    p02/p01: total pressure ratio when `p1`==1.0.
    p0: total pressure when `M1`==1.0
    p0/p0*: reference total pressure ratio when `p01`==1.0 and `M1`==1.0
    """
    return p01 * (
        pressure2(M2=M2, M1=M1, p1=1.0, gam=gam)
        * isentropic_flow.total_pressure_ratio(mach=M2, specific_heat_ratio=gam)
        / isentropic_flow.total_pressure_ratio(mach=M1, specific_heat_ratio=gam)
    )


def total_temperature2(M2, M1=1.0, T01=1.0, gam: float = 1.4):
    """
    `total_temperature2` computes the total temperature `T02` for Rayleigh flow.

    This form is valid for calorically perfect gases only, where `gamma`
    is constant (M1 < 5).

    Parameters:
    M2 (float): Mach number for region 2 of Rayleigh flow.
    M1 (float): Mach number for region 1 of Rayleigh flow.
    T01 (float): total temperature of region 1. Equal to the sonic
        reference total temperature t0* if `M1`==1.0
    gam (float): gamma, ratio of specific heats.

    Returns:
    T02: total temperature for region 2 of heat addition.
    T02/T01: total temperature ratio when `p1`==1.0.
    t0: total temperature when `M1`==1.0
    t0/t0*: reference total temperature ratio when `T01`==1.0 and `M1`==1.0
    """
    return T01 * (
        pressure2(M2=M2, M1=M1, p1=1.0, gam=gam)**2
        * (M2*M2/M1/M1)
        * isentropic_flow.total_temperature_ratio(mach=M2, specific_heat_ratio=gam)
        / isentropic_flow.total_temperature_ratio(mach=M1, specific_heat_ratio=gam)
    )


def pressure2(M2, M1=1.0, p1=1.0, gam: float = 1.4):
    """
    `pressure2` computes the pressure `p2` for Rayleigh flow.

    This form is valid for calorically perfect gases only, where `gamma`
    is constant (M1 < 5).

    Parameters:
    M2 (float): Mach number for region 2 of Rayleigh flow.
    M1 (float): Mach number for region 1 of Rayleigh flow.
    p1 (float): static pressure of region 1. Equal to the sonic
        reference pressure p* if `M1`==1.0
    gam (float): gamma, ratio of specific heats.

    Returns:
    p2: static pressure for region 2 of heat addition.
    p2/p1: pressure ratio when `p1`==1.0.
    p: temperature when `M1`==1.0
    p/p*: reference pressure ratio when `p1`==1.0 and `M1`==1.0
    """
    return p1*(1+gam*M1*M1)/(1+gam*M2*M2)


def temperature2(M2, M1=1.0, T1=1.0, gam: float = 1.4):
    """
    `temperature2` computes the temperature `T2` for Rayleigh flow.

    This form is valid for calorically perfect gases only, where `gamma`
    is constant (M1 < 5).

    Parameters:
    M2 (float): Mach number for region 2 of Rayleigh flow.
    M1 (float): Mach number for region 1 of Rayleigh flow.
    T1 (float): static temperature of region 1. Equal to the sonic
        reference temperature T* if `M1`==1.0
    gam (float): gamma, ratio of specific heats.

    Returns:
    T2: static temperature for region 2 of heat addition.
    T2/T1: temperature ratio when `T1`==1.0.
    t: temperature when `M1`==1.0
    t/t*: reference temperature ratio when `T1`==1.0 and `M1`==1.0
    """
    return T1*((1+gam*M1*M1)/(1+gam*M2*M2))**2 * (M2/M1)**2


def density2(M2, M1=1.0, rho1=1.0, gam: float = 1.4):
    """
    `density2` computes the density `rho2` for Rayleigh flow.

    This form is valid for calorically perfect gases only, where `gamma`
    is constant (M1 < 5).

    Parameters:
    M2 (float): Mach number for region 2 of Rayleigh flow.
    M1 (float): Mach number for region 1 of Rayleigh flow.
    rho1 (float): density of region 1. Equal to the sonic
        reference density rho* if `M1`==1.0
    gam (float): gamma, ratio of specific heats.

    Returns:
    rho2: density of region 2 for Rayleigh flow.
    rho2/rho1: density ratio when `rho1`==1.0.
    rho: density when `M1`==1.0
    rho/rho*: reference density ratio when `rho1`==1.0 and `M1`==1.0
    """
    return rho1*(1+gam*M2*M2)/(1+gam*M1*M1) * (M1/M2)**2
