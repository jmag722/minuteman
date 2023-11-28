"""
This module computes flow parameters of 1D, stationary, calorically perfect normal shocks.

Enthalpy is constant across these shocks. For perfect (calorically+thermally) gases, 
total temperature is also constant across the shock, and will not be output here.
"""
import numpy as np
from scipy.optimize import fsolve
from . import isentropic as isc


def lookup_table(M1=None,p21=None,r21=None,t21=None,p02_p01=None,p02_p1=None,M2=None,gam=1.4):
    """
    `lookup_table` computes the normal shock parameters for a given input.

    This form is valid for calorically perfect gases only, where gamma
    is constant (M1 < 5). Note `T02/T01=1` for perfect gases, as is total enthalpy.

    Parameters:
    p21 (float): pressure ratio p2/p1
    r21 (float): density ratio rho2/rho1
    t21 (float): temperature ratio T2/T1
    p02_p01 (float): total pressure ratio p02/p01
    p02_p1 (float): total pressure to static ratio p02/p1
    M1 (float): Mach number ahead of normal shock
    M2 (float): Mach number behind normal shock
    gam (float): ratio of specific heats gamma

    Returns:
    dict: Normal shock table parameters
    """
    if (M1 is not None and
        p21 is None and r21 is None and t21 is None 
        and p02_p01 is None and p02_p1 is None and M2 is None):
        return {"M1":M1,
                "M2":mach_2(M1,gam),
                "p21":pressure_2(M1,p1=1.0,gam=gam),
                "r21":density_2(M1,rho1=1.0,gam=gam),
                "t21":temperature_2(M1,t1=1.0,gam=gam),
                "p02_p01":total_pressure_2(M1,p01=1.0,gam=gam),
                "p02_p1":( total_pressure_2(M1,p01=1.0,gam=gam)
                         * isc.total_pressure(M=M1,p=1.0,gam=gam) )
                }
    elif (p21 is not None and
          M1 is None and r21 is None and t21 is None 
          and p02_p01 is None and p02_p1 is None and M2 is None):
        M1 = mach_1(p21=p21,gam=gam)
        return lookup_table(M1=M1,gam=gam)

    elif (r21 is not None and
          M1 is None and p21 is None and t21 is None 
          and p02_p01 is None and p02_p1 is None and M2 is None):
        M1 = mach_1(r21=r21,gam=gam)
        return lookup_table(M1=M1,gam=gam)

    elif (t21 is not None and
          M1 is None and p21 is None and r21 is None 
          and p02_p01 is None and p02_p1 is None and M2 is None):
        M1 = mach_1(t21=t21,gam=gam)
        return lookup_table(M1=M1,gam=gam)

    elif (p02_p01 is not None and
          M1 is None and p21 is None and r21 is None 
          and t21 is None and p02_p1 is None and M2 is None):
        M1 = mach_1(p02_p01=p02_p01,gam=gam)
        return lookup_table(M1=M1,gam=gam)
    
    elif (p02_p1 is not None and
          M1 is None and p21 is None and r21 is None 
          and t21 is None and p02_p01 is None and M2 is None):
        M1 = mach_1(p02_p1=p02_p1,gam=gam)
        return lookup_table(M1=M1,gam=gam)

    elif (M2 is not None and
          M1 is None and p21 is None and r21 is None 
          and t21 is None and p02_p01 is None and p02_p1 is None):
        M1 = mach_1(M2=M2,gam=gam)
        return lookup_table(M1=M1,gam=gam)
    else:
        raise Exception("Specify only p2/p1, rho2/rho1, T2/T1, p02/p01, p02/p1, M1, or M2.")



def mach_1(p21=None,r21=None,t21=None,p02_p01=None,p02_p1=None,M2=None,gam=1.4):
    """
    `mach_1` computes the Mach number `M1` ahead of a normal shock.

    This form is valid for calorically perfect gases only, where gamma
    is constant (M1 < 5). Note `T02/T01=1` for perfect gases, as is total enthalpy.

    Parameters:
    p21 (float): pressure ratio p2/p1
    r21 (float): density ratio rho2/rho1
    t21 (float): temperature ratio T2/T1
    p02_p01 (float): total pressure ratio p02/p01
    p02_p1 (float): total pressure to static ratio p02/p1
    M2 (float): Mach number behind normal shock
    gam (float): ratio of specific heats gamma

    Returns:
    M1: Mach number ahead of a normal shock
    """
    if ( p21 is not None and  # p2/p1 known
        r21 is None and t21 is None and p02_p01 is None
         and p02_p1 is None and M2 is None):
        M1 = ((p21-1)*(gam+1)/2/gam + 1)**0.5
    elif (r21 is not None and  # rho2/rho1 known
          p21 is None and t21 is None and p02_p01 is None
          and p02_p1 is None and M2 is None):
        M1 = ( 2 / ((gam+1)/r21 - gam + 1) )**0.5
    elif (t21 is not None and  # T2/T1 known
          p21 is None and r21 is None and p02_p01 is None
          and p02_p1 is None and M2 is None):
        func = lambda M1,t21,gam: t21 - temperature_2(M1,t1=1.0,gam=gam)
        M1= fsolve(func, 2.0,args=(t21,gam))[0]
    elif (M2 is not None and  # M2 known
          p21 is None and r21 is None and p02_p01 is None
          and p02_p1 is None and t21 is None):
        M1 = ( (1+M2*M2*(gam-1)/2) / (gam*M2*M2-(gam-1)/2) )**0.5
    elif (p02_p01 is not None and #p02/p01 known
          p21 is None and r21 is None and M2 is None
          and p02_p1 is None and t21 is None):
        func = lambda M1, p02_p01,gam: p02_p01 - total_pressure_2(M1,p01=1.0,gam=gam)
        M1= fsolve(func, 2.0,args=(p02_p01,gam))[0]
    elif (p02_p1 is not None and  # p02/p1 known
          p21 is None and r21 is None and M2 is None
          and p02_p01 is None and t21 is None):
        func = lambda M1,p02_p1,gam: (
            p02_p1 
            - total_pressure_2(M1,p01=1.0,gam=gam) 
            * isc.total_pressure(M1,p=1.0,gam=gam)
        )
        M1 = fsolve(func,2.0, args=(p02_p1,gam))[0]
    else:
        raise Exception("Specify only p2/p1, rho2/rho1, T2/T1, p02/p01, p02/p1, or M2.")
    return M1



def mach_2(M1,gam=1.4):
    """
    `mach_2` computes the Mach number `M2` behind a normal shock.

    This form is valid for calorically perfect gases only, where gamma
    is constant (M1 < 5).

    Parameters:
    M1 (float): Mach number ahead of normal shock
    gam (float): gamma, ratio of specific heats

    Returns:
    M2: Mach number behind a normal shock
    """
    return ( (1 + (gam-1)*M1*M1/2) / (gam*M1*M1 - (gam-1)/2) )**0.5



def density_2(M1,rho1=1.0,gam=1.4):
    """
    `density_2` computes the density `rho2` behind a normal shock.

    This form is valid for calorically perfect gases only, where `gamma`
    is constant (M1 < 5). Density ratio `rho2/rho1` returned when default
    argument `rho1`==1.0. `rho2/rho1` is equivalent to `u1/u2`.

    Parameters:
    M1 (float): Mach number ahead of normal shock
    rho1 (float): density ahead of normal shock
    gam (float): gamma, ratio of specific heats

    Returns:
    rho2: density behind normal shock
    rho2/rho1: density ratio when rho1==1.0.
    """
    return rho1 * (gam+1)*M1*M1/(2+(gam-1)*M1*M1)



def pressure_2(M1,p1=1.0,gam=1.4):
    """
    `pressure_2` computes the pressure `p2` behind a normal shock.

    This form is valid for calorically perfect gases only, where `gamma`
    is constant (M1 < 5). Pressure ratio `p2/p1` returned when default
    argument `p1`==1.0.

    Parameters:
    M1 (float): Mach number ahead of normal shock
    p1 (float): pressure ahead of normal shock
    gam (float): gamma, ratio of specific heats

    Returns:
    p2: pressure behind normal shock
    p2/p1: pressure ratio when `p1`==1.0.
    """
    return p1 * (1 + 2*gam/(gam+1)*(M1*M1-1))



def total_pressure_2(M1,p01=1.0,gam=1.4):
    """
    `total_pressure_2` computes the total pressure `p02` behind a normal shock.

    This form is valid for calorically perfect gases only, where `gamma`
    is constant (M1 < 5). Pressure ratio `p02/p01` returned when default
    argument `p01`==1.0.

    Parameters:
    M1 (float): Mach number ahead of normal shock
    p01 (float): total pressure ahead of normal shock
    gam (float): gamma, ratio of specific heats

    Returns:
    p02: pressure behind normal shock
    p02/p01: pressure ratio when `p01`==1.0.
    """
    return ( p01 
            * ( (gam+1)/(2*gam*M1*M1-gam+1) )**(1/(gam-1)) 
            * ( (gam+1)*M1*M1/((gam-1)*M1*M1+2) )**(gam/(gam-1)) )



def temperature_2(M1,t1=1.0,gam=1.4):
    """
    `temperature_2` computes the temperature `t2` behind a normal shock.

    This form is valid for calorically perfect gases only, where `gamma`
    is constant (M1 < 5). Temperature ratio `t2/t1` returned when default
    argument `t1`==1.0. `t2/t1` is equivalent to `h2/h1`.

    Parameters:
    M1 (float): Mach number ahead of normal shock
    t1 (float): temperature ahead of normal shock
    gam (float): gamma, ratio of specific heats

    Returns:
    t2: temperature behind normal shock
    t2/t1: temperature ratio when `t1`==1.0.
    """
    return t1 * (1+2*gam/(gam+1)*(M1*M1-1)) * (2+(gam-1)*M1*M1)/((gam+1)*M1*M1)



def entropy_2(R,p02_p01,s1=0.0):
    """
    `entropy_2` computes the entropy `s2` behind a normal shock.

    This form is valid for calorically perfect gases only, where `gamma`
    is constant (M1 < 5). Temperature ratio `t2/t1` returned when default
    argument `t1`==1.0. `t2/t1` is equivalent to `h2/h1`.

    Parameters:
    R (float): Specific gas constant of the gas.
    p02_p01 (float): Total pressure ratio p02/p01.
    s1 (float): Entropy before normal shock.

    Returns:
    s2: entropy behind normal shock
    ds: entropy change due to normal shock when `s1==0.0`
    """
    return s1 - R * np.log(p02_p01)



def hugoniot(p1,p2,v1,v2,e1=0.0):
    """
    `hugoniot` computes the change in specific energy about a normal shock.

    This relates only thermodynamic quantities across a shock.
    This relation is valid for perfect, chemically reacting, and real gases.

    Parameters:
    p1 (float): Static pressure ahead of normal shock.
    p2 (float): Static pressure behind normal shock.
    v1 (float): Specific volume ahead of normal shock.
    v2 (float): Specific volume behind a normal shock.
    e1 (float): Specific energy ahead of normal shock.

    Returns:
    e2: specific energy downstream of the normal shock
    de: energy change due to normal shock when `e1==0.0`
    """
    return e1 + (p1+p2)/2 * (v1-v2)