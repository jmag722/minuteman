"""
This module computes 1D, calorically perfect flow with heat addition (Rayleigh flow).
Online calculator for comparison at: http://www.dept.aoe.vt.edu/~devenpor/aoe3114/calc.html
Additional notes: https://kyleniemeyer.github.io/gas-dynamics-notes/compressible-flows/heat-transfer.html
"""

from scipy.optimize import fsolve
import numpy as np
import phypy.thermodynamics.caloric_perfect as calp
import phypy.compressible.isentropic as isc

def lookup_table(p21=None,r21=None,t21=None,s21=None,p02_p01=None,t02_t01=None,m2=None,supersonic=True,m1=1.0,gam=1.4):
    """
    `lookup_table` computes Rayleigh flow parameters for a given input.

    This form is valid for calorically perfect gases only, where gamma
    is constant (M1 < 5). 

    Parameters:
    p21 (float): pressure ratio p2/p1. p/p* if `m1`==1.0
    r21 (float): density ratio rho2/rho1. rho/rho* if `m1`==1.0
    t21 (float): temperature ratio T2/T1. T/T* if `m1`==1.0
    s21 (float): entropy change (s2-s1)/R. (s-s*)/R if `m1`==1.0
    p02_p01 (float): total pressure ratio p02/p01. p0/p0* if `m1`==1.0
    t02_t01 (float): total pressure ratio T02/T01. T0/T0* if `m1`==1.0
    m2 (float): Mach number at station 2 of Rayleigh flow. m if `m1`==1.0
    m1 (float): Mach number at station 1 of Rayleigh flow. Default 1.0
    supersonic (bool): is flow supersonic or subsonic. Needed to compute M2
        from thermodynamic variables.
    gam (float): ratio of specific heats gamma

    Returns:
    dict: Rayleigh Flow table parameters.
    """
    if (m2 is not None and  # M2 (or M if M1=1.0) known
        all(var is None for var in [p21, r21, t21, s21, p02_p01, t02_t01])):
        return {"m1":m1,
                "m2":m2,
                "p21":pressure_2(m2=m2,m1=m1,p1=1.0,gam=gam),
                "r21":density_2(m2=m2,m1=m1,rho1=1.0,gam=gam),
                "t21":temperature_2(m2=m2,m1=m1,t1=1.0,gam=gam),
                "s21":entropy_2(m2=m2,m1=m1,R=1.0,s1=0.0,gam=gam),
                "p02_p01":total_pressure_2(m2=m2,m1=m1,p01=1.0,gam=gam),
                "t02_t01":total_temperature_2(m2=m2,m1=m1,t01=1.0,gam=gam)
                }

    elif (p21 is not None and  # p2/p1 (or p/p* if m1=1.0) known
          all(var is None for var in [m2, r21, t21, s21, p02_p01, t02_t01])):
        m2 = mach_2(m1=m1,p21=p21,gam=gam,supersonic=supersonic)
        return lookup_table(m2=m2,m1=m1,gam=gam)

    elif (r21 is not None and  # rho2/rho1 (or rho/rho* if m1=1.0) known
          all(var is None for var in [p21, m2, t21, s21, p02_p01, t02_t01])):
        m2 = mach_2(m1=m1,r21=r21,gam=gam,supersonic=supersonic)
        return lookup_table(m2=m2,m1=m1,gam=gam)

    elif (t21 is not None and  # T2/T1 (or T/T* if m1=1.0) known
          all(var is None for var in [p21, r21, m2, s21, p02_p01, t02_t01])):
        m2 = mach_2(m1=m1,t21=t21,gam=gam,supersonic=supersonic)
        return lookup_table(m2=m2,m1=m1,gam=gam)

    elif (s21 is not None and  # (s2-s1)/R (or (s-s*)/R if m1=1.0) known
          all(var is None for var in [p21, r21, t21, m2, p02_p01, t02_t01])):
        m2 = mach_2(m1=m1,s21=s21,gam=gam,supersonic=supersonic)
        return lookup_table(m2=m2,m1=m1,gam=gam)

    elif (p02_p01 is not None and  # p02/p01 (or p0/p0* if m1=1.0) known
          all(var is None for var in [p21, r21, t21, s21, m2, t02_t01])):
        m2 = mach_2(m1=m1,p02_p01=p02_p01,gam=gam,supersonic=supersonic)
        return lookup_table(m2=m2,m1=m1,gam=gam)
    
    elif (t02_t01 is not None and  # T02/T01 (or T0/T0* if m1=1.0) known
          all(var is None for var in [p21, r21, t21, s21, p02_p01, m2])):
        m2 = mach_2(m1=m1,t02_t01=t02_t01,gam=gam,supersonic=supersonic)
        return lookup_table(m2=m2,m1=m1,gam=gam)

    else:
        raise Exception("Specify only p2/p1, rho2/rho1, T2/T1, p02/p01, T02/T01, or M2.")


def mach_2(m1=1.0,supersonic=True,p21=None,r21=None,t21=None,s21=None,
            p02_p01=None,t02_t01=None,gam=1.4):
    """
    `mach_2` computes the Mach number at region 2 for Rayleigh flow.

    This form is valid for calorically perfect gases only, where `gamma`
    is constant (M1 < 5). 

    Parameters:
    m1 (float): Mach number for region 1 of Rayleigh flow.
    supersonic (bool): is flow supersonic or subsonic.
    p21 (float): p2/p1, or p/p* if m1=1.0
    r21 (float): rho2/rho1, or rho/rho* if m1=1.0
    t21 (float): T2/T1, or T/T* if m1=1.0
    s21 (float): (s2-s1)/R, or (s-s*)/R if m1=1.0
    p02_p01 (float): p02/p01, or p0/p0* if m1=1.0
    t02_t01 (float): T02/T01, or T0/T0* if m1=1.0
    gam (float): gamma, ratio of specific heats.

    Returns:
    m2: Mach number for region 2 of heat addition.
    m: mach number if `m1`=M*=1.0
    """
    if supersonic:
        m2_guess = 3.0
    else:
        m2_guess = 0.5
    if ( p21 is not None and  # p2/p1 known
         all(var is None for var in [r21, t21, s21, p02_p01, t02_t01]) ):
        return ( ((1+gam*m1*m1)/p21 - 1)/gam )**0.5

    elif ( t21 is not None and  # T2/T1
           all(var is None for var in [r21, p21, s21, p02_p01, t02_t01])):
        func = lambda m2,m1,t21,gam: t21 - temperature_2(m2,m1,t1=1.0,gam=gam)
        return fsolve(func, m2_guess,args=(m1,t21,gam))[0]

    elif ( r21 is not None and  # rho2/rho1
           all(var is None for var in [p21, t21, s21, p02_p01, t02_t01])):
        return 1/np.sqrt( (r21*(1+gam*m1*m1))/(m1*m1) - gam)

    elif ( s21 is not None and  # ds/R
           all(var is None for var in [r21, t21, p21, p02_p01, t02_t01])):
        func = lambda m2,m1,s21,gam: s21 - entropy_2(m2,m1,R=1.0,s1=0.0,gam=gam)
        return fsolve(func, m2_guess,args=(m1,s21,gam))[0]

    elif ( p02_p01 is not None and  # p02/p01
           all(var is None for var in [r21, t21, s21, p21, t02_t01])):
        func = lambda m2,m1,p02_p01,gam: p02_p01 - total_pressure_2(m2,m1,p01=1.0,gam=gam)
        return fsolve(func, m2_guess,args=(m1,p02_p01,gam))[0]

    elif ( t02_t01 is not None and  # T02/T01
           all(var is None for var in [r21, t21, s21, p02_p01, p21])):
        func = lambda m2,m1,t02_t01,gam: t02_t01 - total_temperature_2(m2,m1,t01=1.0,gam=gam)
        return fsolve(func, m2_guess,args=(m1,t02_t01,gam))[0]

    else:
        raise Exception("Specify exactly 1 of the following: p21, t21, r21, s21, t02_t01, or p02_p01.")

def entropy_2(m2,m1=1.0,R=1.0,s1=0.0,gam=1.4):
    """
    `entropy_2` computes the change in entropy for Rayleigh flow.

    This form is valid for calorically perfect gases only, where `gamma`
    is constant (M1 < 5). 

    Parameters:
    m2 (float): Mach number for region 2 of Rayleigh flow.
    m1 (float): Mach number for region 1 of Rayleigh flow.
    R (float): specific gas constant.
    s1 (float): entropy for region 1 of Rayleigh flow.
    gam (float): gamma, ratio of specific heats.

    Returns:
    s2: entropy for region 2 of heat addition.
    ds: change in entropy when `s1`==0.0
    ds/R: change in entropy for specific gas constant when `s1`==0.0 and `R`=1.0
    """
    return ( s1 + R * np.log( ( (1+gam*m1*m1) / (1+gam*m2*m2))**(gam+1)
                            * (m2*m2/(m1*m1))**(gam) ) 
                    / (gam-1) )



def total_pressure_2(m2,m1=1.0,p01=1.0,gam=1.4):
    """
    `total_pressure_2` computes the total pressure `p02` for Rayleigh flow.

    This form is valid for calorically perfect gases only, where `gamma`
    is constant (M1 < 5). 

    Parameters:
    m2 (float): Mach number for region 2 of Rayleigh flow.
    m1 (float): Mach number for region 1 of Rayleigh flow.
    p01 (float): total pressure of region 1. Equal to the sonic 
        reference total pressure p0* if `m1`==1.0
    gam (float): gamma, ratio of specific heats.

    Returns:
    p02: total pressure for region 2 of heat addition.
    p02/p01: total pressure ratio when `p1`==1.0.
    p0: total pressure when `m1`==1.0
    p0/p0*: reference total pressure ratio when `p01`==1.0 and `m1`==1.0
    """
    return p01 * ( 
        pressure_2(m2=m2,m1=m1,p1=1.0,gam=gam)
        * isc.total_pressure(M=m2,p=1.0,gam=gam)
        / isc.total_pressure(M=m1,p=1.0,gam=gam) 
    )


def total_temperature_2(m2,m1=1.0,t01=1.0,gam=1.4):
    """
    `total_temperature_2` computes the total temperature `t02` for Rayleigh flow.

    This form is valid for calorically perfect gases only, where `gamma`
    is constant (M1 < 5). 

    Parameters:
    m2 (float): Mach number for region 2 of Rayleigh flow.
    m1 (float): Mach number for region 1 of Rayleigh flow.
    t01 (float): total temperature of region 1. Equal to the sonic 
        reference total temperature t0* if `m1`==1.0
    gam (float): gamma, ratio of specific heats.

    Returns:
    t02: total temperature for region 2 of heat addition.
    t02/t01: total temperature ratio when `p1`==1.0.
    t0: total temperature when `m1`==1.0
    t0/t0*: reference total temperature ratio when `t01`==1.0 and `m1`==1.0
    """
    return t01 * (
        pressure_2(m2=m2,m1=m1,p1=1.0,gam=gam)**2
        * (m2*m2/m1/m1)
        * isc.total_temperature(M=m2,T=1.0,gam=gam)
        / isc.total_temperature(M=m1,T=1.0,gam=gam)
    )


def pressure_2(m2,m1=1.0,p1=1.0,gam=1.4):
    """
    `pressure_2` computes the pressure `p2` for Rayleigh flow.

    This form is valid for calorically perfect gases only, where `gamma`
    is constant (M1 < 5). 

    Parameters:
    m2 (float): Mach number for region 2 of Rayleigh flow.
    m1 (float): Mach number for region 1 of Rayleigh flow.
    p1 (float): static pressure of region 1. Equal to the sonic 
        reference pressure p* if `m1`==1.0
    gam (float): gamma, ratio of specific heats.

    Returns:
    p2: static pressure for region 2 of heat addition.
    p2/p1: pressure ratio when `p1`==1.0.
    p: temperature when `m1`==1.0
    p/p*: reference pressure ratio when `p1`==1.0 and `m1`==1.0
    """
    return p1*(1+gam*m1*m1)/(1+gam*m2*m2)


def temperature_2(m2,m1=1.0,t1=1.0,gam=1.4):
    """
    `temperature_2` computes the temperature `t2` for Rayleigh flow.

    This form is valid for calorically perfect gases only, where `gamma`
    is constant (M1 < 5). 

    Parameters:
    m2 (float): Mach number for region 2 of Rayleigh flow.
    m1 (float): Mach number for region 1 of Rayleigh flow.
    t1 (float): static temperature of region 1. Equal to the sonic 
        reference temperature T* if `m1`==1.0
    gam (float): gamma, ratio of specific heats.

    Returns:
    t2: static temperature for region 2 of heat addition.
    t2/t1: temperature ratio when `t1`==1.0.
    t: temperature when `m1`==1.0
    t/t*: reference temperature ratio when `t1`==1.0 and `m1`==1.0
    """
    return t1*( (1+gam*m1*m1)/(1+gam*m2*m2) )**2 * (m2/m1)**2


def density_2(m2,m1=1.0,rho1=1.0,gam=1.4):
    """
    `density_2` computes the density `rho2` for Rayleigh flow.

    This form is valid for calorically perfect gases only, where `gamma`
    is constant (M1 < 5). 

    Parameters:
    m2 (float): Mach number for region 2 of Rayleigh flow.
    m1 (float): Mach number for region 1 of Rayleigh flow.
    rho1 (float): density of region 1. Equal to the sonic 
        reference density rho* if `m1`==1.0
    gam (float): gamma, ratio of specific heats.

    Returns:
    rho2: density of region 2 for Rayleigh flow.
    rho2/rho1: density ratio when `rho1`==1.0.
    rho: density when `m1`==1.0
    rho/rho*: reference density ratio when `rho1`==1.0 and `m1`==1.0
    """
    return rho1*(1+gam*m2*m2)/(1+gam*m1*m1) * (m1/m2)**2
