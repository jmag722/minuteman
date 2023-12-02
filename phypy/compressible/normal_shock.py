"""
This module computes flow parameters of 1D, stationary, calorically perfect normal shocks.

Enthalpy is constant across these shocks. For perfect (calorically+thermally) gases, 
total temperature is also constant across the shock, and will not be output here.
"""
import numpy as np
from scipy.optimize import fsolve
import phypy.compressible.isentropic as isen
import phypy.utils.arg_checks as ac


def lookup_table(M1:float=None, p21:float=None, r21:float=None,
                 T21:float=None, p02_p01:float=None, p02_p1:float=None,
                 M2:float=None, gam:float=1.4):
    """
    Computes the normal shock parameters for a given input.

    This form is valid for calorically perfect gases only, where gamma
    is constant (M1 < 5).
    
    Note `T02/T01`=1.0 for perfect gases, as does total enthalpy ratio.

    Parameters
    ----------
    M1 : float, optional
        Mach number upstream of the normal shock, by default None
    p21 : float, optional
        pressure ratio p2/p1, by default None
    r21 : float, optional
        density ratio rho2/rho1, by default None
    T21 : float, optional
        temperature ratio T2/T1, by default None
    p02_p01 : float, optional
        total pressure ratio p02/p01, by default None
    p02_p1 : float, optional
        total pressure to static ratio p02/p1, by default None
    M2 : float, optional
        Mach number downstream of the normal shock, by default None
    gam : float, optional
        ratio of specific heats gamma, by default 1.4

    Returns
    -------
    dict
        normal shock table parameters

    Raises
    ------
    ValueError
        Incorrect or inconsistent inputs specified. Specify one and only 
        one input parameter along with gamma.
    """
    if ac.is1known(M1, [p21, r21, T21, p02_p01, p02_p1, M2]):
        return {
            "M1": M1,
            "M2": mach2(M1=M1, gam=gam),
            "p21": pressure2(M1=M1, p1=1.0, gam=gam),
            "r21": density2(M1=M1, rho1=1.0, gam=gam),
            "T21": temperature2(M1=M1, T1=1.0, gam=gam),
            "p02_p01": total_pressure2(M1=M1, p01=1.0, gam=gam),
            "p02_p1":( total_pressure2(M1=M1, p01=1.0, gam=gam)
                       * isen.total_pressure(M=M1, p=1.0, gam=gam) )
        }
    elif ac.is1known(p21, [M1, r21, T21, p02_p01, p02_p1, M2]):
        M1 = mach_1(p21=p21, gam=gam)
        return lookup_table(M1=M1, gam=gam)

    elif ac.is1known(r21, [M1, p21, T21, p02_p01, p02_p1, M2]):
        M1 = mach_1(r21=r21, gam=gam)
        return lookup_table(M1=M1, gam=gam)

    elif ac.is1known(T21, [M1, p21, r21, p02_p01, p02_p1, M2]):
        M1 = mach_1(T21=T21, gam=gam)
        return lookup_table(M1=M1, gam=gam)

    elif ac.is1known(p02_p01, [M1, p21, r21, T21, p02_p1, M2]):
        M1 = mach_1(p02_p01=p02_p01, gam=gam)
        return lookup_table(M1=M1, gam=gam)
    
    elif ac.is1known(p02_p1, [M1, p21, r21, T21, p02_p01, M2]):
        M1 = mach_1(p02_p1=p02_p1, gam=gam)
        return lookup_table(M1=M1, gam=gam)

    elif ac.is1known(M2, [M1, p21, r21, T21, p02_p01, p02_p1]):
        M1 = mach_1(M2=M2, gam=gam)
        return lookup_table(M1=M1, gam=gam)
    else:
        raise ValueError("Specify only p2/p1, rho2/rho1, T2/T1, p02/p01, "
                         "p02/p1, M1, or M2.")

def mach_1(p21:float=None, r21:float=None, T21:float=None, p02_p01:float=None,
           p02_p1:float=None, M2:float=None, gam:float=1.4):
    """
    Computes the Mach number `M1` upstream of a normal shock.

    This form is valid for calorically perfect gases only, where gamma
    is constant (M1 < 5).
    
    Note `T02/T01`=1.0 for perfect gases, as does total enthalpy ratio.

    Parameters
    ----------
    p21 : float, optional
        pressure ratio p2/p1, by default None
    r21 : float, optional
        density ratio rho2/rho1, by default None
    T21 : float, optional
        temperature ratio T2/T1, by default None
    p02_p01 : float, optional
        total pressure ratio p02/p01, by default None
    p02_p1 : float, optional
        total pressure to static ratio p02/p1, by default None
    M2 : float, optional
        Mach number downstream of a normal shock, by default None
    gam : float, optional
        ratio of specific heats gamma, by default 1.4

    Returns
    -------
    float
        Mach number upstream of a normal shock

    Raises
    ------
    ValueError
        Incorrect or inconsistent inputs specified. Specify one and only 
        one input parameter along with gamma.
    """    
    if ac.is1known(p21, [r21, T21, p02_p01, p02_p1, M2]):
        M1 = ((p21-1)*(gam+1)/2/gam + 1)**0.5

    elif ac.is1known(r21, [p21, T21, p02_p01, p02_p1, M2]):
        M1 = ( 2 / ((gam+1)/r21 - gam + 1) )**0.5

    elif ac.is1known(T21, [p21, r21, p02_p01, p02_p1, M2]):
        func = lambda M1, T21, gam: T21 - temperature2(M1=M1, T1=1.0, gam=gam)
        M1= fsolve(func, 2.0, args=(T21, gam))[0]

    elif ac.is1known(M2, [p21, r21, p02_p01, p02_p1, T21]):
        M1 = ( (1+M2*M2*(gam-1)/2) / (gam*M2*M2-(gam-1)/2) )**0.5

    elif ac.is1known(p02_p01, [p21, r21, T21, p02_p1, M2]):
        func = lambda M1, p02_p01, gam: p02_p01 - total_pressure2(M1=M1, p01=1.0, gam=gam)
        M1= fsolve(func, 2.0, args=(p02_p01, gam))[0]

    elif ac.is1known(p02_p1, [p21, r21, T21, p02_p01, M2]):
        func = lambda M1, p02_p1, gam: (
            p02_p1 
            - total_pressure2(M1=M1, p01=1.0, gam=gam) 
            * isen.total_pressure(M1, p=1.0, gam=gam)
        )
        M1 = fsolve(func, 2.0, args=(p02_p1, gam))[0]
    else:
        raise ValueError("Specify only p2/p1, rho2/rho1, T2/T1, p02/p01, p02/p1, or M2.")
    return M1

def mach2(M1, gam:float=1.4):
    """
    Computes the Mach number `M2` behind a normal shock.

    This form is valid for calorically perfect gases only, where gamma
    is constant (M1 < 5).

    Parameters
    ----------
    M1 : Any
        Mach number upstream of the normal shock
    gam : float, optional
        ratio of specific heats gamma, by default 1.4

    Returns
    -------
    Any
        Mach number M2 downstream of the normal shock
    """
    return ( (1 + (gam-1)*M1*M1/2) / (gam*M1*M1 - (gam-1)/2) )**0.5

def density2(M1, rho1=1.0, gam:float=1.4):
    """
    Computes the density `rho2` behind a normal shock.

    This form is valid for calorically perfect gases only, where `gamma`
    is constant (M1 < 5).
    
    Density ratio `rho2/rho1` returned when `rho1`==1.0.
    
    `rho2/rho1` is equivalent to `u1/u2`.

    Parameters
    ----------
    M1 : Any
        Mach number upstream of the normal shock
    rho1 : float, optional
        Density upstream of the normal shock, by default 1.0
    gam : float, optional
        ratio of specific heats gamma, by default 1.4

    Returns
    -------
    Any
        density rho2 downstream the normal shock AND

        density ratio rho2/rho1 when `rho1`==1.0
    """
    return rho1 * (gam+1)*M1*M1/(2+(gam-1)*M1*M1)

def pressure2(M1, p1=1.0, gam:float=1.4):
    """
    Computes the pressure `p2` behind a normal shock.

    This form is valid for calorically perfect gases only, where `gamma`
    is constant (M1 < 5). 
    
    Pressure ratio `p2/p1` returned when `p1`==1.0.

    Parameters
    ----------
    M1 : Any
        Mach number upstream of the normal shock
    p1 : float, optional
        pressure upstream of the normal shock, by default 1.0
    gam : float, optional
        ratio of specific heats gamma, by default 1.4

    Returns
    -------
    Any
        pressure p2 downstream the normal shock AND

        pressure ratio p2/p1 when `p1`==1.0
    """
    return p1 * (1 + 2*gam/(gam+1)*(M1*M1-1))

def total_pressure2(M1, p01=1.0, gam:float=1.4):
    """
    Computes the total pressure `p02` behind a normal shock.

    This form is valid for calorically perfect gases only, where `gamma`
    is constant (M1 < 5).
    Pressure ratio `p02/p01` returned when `p01`==1.0.

    Parameters
    ----------
    M1 : Any
        Mach number upstream of the normal shock
    p01 : float, optional
        total pressure upstream of the normal shock, by default 1.0
    gam : float, optional
        ratio of specific heats gamma, by default 1.4

    Returns
    -------
    Any
        total pressure p02 downstream the normal shock AND

        total pressure ratio p02/p01 when `p01`==1.0
    """
    return (
        p01 
        * ( (gam+1)/(2*gam*M1*M1-gam+1) )**(1/(gam-1)) 
        * ( (gam+1)*M1*M1/((gam-1)*M1*M1+2) )**(gam/(gam-1))
    )

def temperature2(M1, T1=1.0, gam:float=1.4):
    """
    Computes the temperature `T2` behind a normal shock.

    This form is valid for calorically perfect gases only, where `gamma`
    is constant (M1 < 5).
    
    Temperature ratio `T2/T1` returned when `T1`==1.0.
    
    T2/T1 is equivalent to h2/h1.

    Parameters
    ----------
    M1 : Any
        Mach number upstream of the normal shock
    T1 : float, optional
        temperature upstream of the normal shock, by default 1.0
    gam : float, optional
        ratio of specific heats gamma, by default 1.4

    Returns
    -------
    Any
        temperature T2 downstream the normal shock AND

        temperature ratio T2/T1 when `T1`==1.0
    """
    return T1 * (1+2*gam/(gam+1)*(M1*M1-1)) * (2+(gam-1)*M1*M1) / ((gam+1)*M1*M1)

def entropy2(p02_p01, R:float, s1:float=0.0):
    """
    Computes the entropy `s2` behind a normal shock.

    This form is valid for calorically+thermally perfect gases over
    a stationary shock.
    
    Entropy change ds returned when `s1`==1.0.

    Parameters
    ----------
    p02_p01 : Any
        total pressure ratio
    R : float
        specific gas constant
    s1 : float, optional
        entropy upstream of the shock, by default 0.0

    Returns
    -------
    Any
        entropy s2 downstream of the normal shock AND

        entropy change ds due to the normal shock if `s1`==0.0
    """
    return s1 - R * np.log(p02_p01)



def hugoniot(p1:float, p2:float, v1:float, v2:float, e1:float=0.0):
    """
    Computes the change in specific energy about a normal shock.

    This relates only thermodynamic quantities across a shock.
    This relation is valid for perfect, chemically reacting, and real gases.

    Parameters
    ----------
    p1 : float
        static pressure upstream of a normal shock
    p2 : float
        static pressure downstream of a normal shock
    v1 : float
        specific volume upstream of a normal shock
    v2 : float
        specific volume downstream of a normal shock
    e1 : float, optional
        specific energy upstream of a normal shock, by default 0.0

    Returns
    -------
    float
        specific energy e2 downstream of the normal shock AND

        energy change de due to normal shock when `e1`==0.0
    """
    return e1 + (p1+p2)/2 * (v1-v2)