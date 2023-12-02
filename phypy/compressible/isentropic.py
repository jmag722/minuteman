"""
Compressible, inviscid flow relations.

These functions are basic definitions pertaining to 1-,2-, and 3-D
compressible, inviscid flow.
"""
from scipy.optimize import fsolve
import phypy.thermodynamics.caloric_perfect as calp
import phypy.utils.arg_checks as ac

def mach(u, a):
    """
    Compute Mach number

    Parameters
    ----------
    u : Any
        speed
    a : Any
        speed of sound

    Returns
    -------
    Any
        Mach Number
    """    
    return u/a

def speed_sound(T=None, R:float=None, rho=None, p=None, gam:float=1.4):
    """
    Compute the speed of sound.

    Specify only (gamma,R,T) or only (gamma,p,rho).

    Definition of speed of sound for a perfect (thermally+calorically) gas.
    Relation is invalid for chemically reacting and real gases.

    Parameters
    ----------
    T : Any, optional
        temperature, by default None.
    R : float, optional
        specific gas constant, by default None.
    rho : Any, optional
        density, by default None.
    p : Any, optional
        pressure, by default None.
    gam : float, optional
        ratio of specific heats gamma, by default 1.4

    Returns
    -------
    Any
        speed of sound

    Raises
    ------
    ValueError
        If either (gamm, R, T) is not specified OR if (gamma, p, rho)
        is not specified.
    """
    solve_with_RT = (all(v is not None for v in [T, R])
                     and all(v2 is None for v2 in [p, rho]))
    if solve_with_RT:
        return (gam * R * T)**0.5
    elif not solve_with_RT:
        return (gam * p / rho)**0.5
    else:
        raise ValueError("Specify only (gamma,R,T) or only (gamma,p,rho).")

def lookup_table(
    M:float=None, p0_ratio:float=None, r0_ratio:float=None,
    T0_ratio:float=None, a0_ratio:float=None,
    area_ratio:float=None, gam:float=1.4,
    is_supersonic:bool=True):
    """
    Provides all isentropic flow variables for a given input.

    Equivalent to a line from lookup tables for isentropic, calorically
    perfect gases (1D and quasi-1D). Function is recursive.

    Parameters
    ----------
    M : float, optional
        Mach number, by default None
    p0_ratio : float, optional
        total pressure ratio p0/p, by default None
    r0_ratio : float, optional
        total density ratio rho0/rho, by default None
    T0_ratio : float, optional
        total temperature ratio T0/T, by default None
    a0_ratio : float, optional
        total speed of sound ratio a0/a, by default None
    area_ratio : float, optional
        area ratio wrt sonic condition A/A*, by default None
    gam : float, optional
        ratio of specific heats gamma, by default 1.4
    is_supersonic : bool, optional
        For given A/A*, initialize guess of Mach number with either supersonic
        or subsonic, by default True

    Returns
    -------
    dict
        all unspecified isentropic flow parameters for a given input

    Raises
    ------
    ValueError
        Incorrect or inconsistent inputs specified.
    """
    if (ac.is1known(M, [p0_ratio, r0_ratio, T0_ratio, a0_ratio, area_ratio])):
        p0_ratio = total_pressure(M=M, p=1.0, gam=gam)
        r0_ratio = total_density(M=M, rho=1.0, gam=gam)
        T0_ratio = total_temperature(M=M, T=1.0, gam=gam)
        a0_ratio = total_speed_sound(M=M, a=1.0, gam=gam)
        area_ratio = area_mach_relation(M=M, gam=gam)
        return {
            "p0_ratio":p0_ratio,
            "r0_ratio":r0_ratio,
            "T0_ratio":T0_ratio,
            "a0_ratio":a0_ratio,
            "area_ratio":area_ratio,
            "M":M,
            "gam":gam
        }

    elif (ac.is1known(p0_ratio, [M, r0_ratio, T0_ratio, a0_ratio, area_ratio])):
        T0_ratio = calp.isentropic_process(p21=p0_ratio, gam=gam)["t21"]
        M = mach_from_temperature_ratio(T0_ratio=T0_ratio, gam=gam)
        return lookup_table(M=M, gam=gam)

    elif (ac.is1known(r0_ratio, [p0_ratio, M, T0_ratio, a0_ratio, area_ratio])):
        T0_ratio = calp.isentropic_process(r21=r0_ratio, gam=gam)["t21"]
        M = mach_from_temperature_ratio(T0_ratio=T0_ratio, gam=gam)
        return lookup_table(M=M, gam=gam)

    elif (ac.is1known(T0_ratio, [p0_ratio, r0_ratio, M, a0_ratio, area_ratio])):
        M = mach_from_temperature_ratio(T0_ratio=T0_ratio, gam=gam)
        return lookup_table(M=M, gam=gam)

    elif (ac.is1known(a0_ratio, [p0_ratio, r0_ratio, T0_ratio, M, area_ratio])):
        T0_ratio = calp.isentropic_process(a21=a0_ratio, gam=gam)["t21"]
        M = mach_from_temperature_ratio(T0_ratio=T0_ratio, gam=gam)
        return lookup_table(M=M, gam=gam)

    elif (ac.is1known(area_ratio, [p0_ratio, r0_ratio, T0_ratio, a0_ratio, M])):
        M = area_mach_relation(area_ratio=area_ratio, gam=gam,
                               is_supersonic=is_supersonic)
        return lookup_table(M=M, gam=gam)

    else: # over or under-specified
        raise ValueError("Specify Mach number, p0/p, rho0/rho, "
                         "T0/T, a0/a, or A/A*.")

def total_temperature(M=1.0, T=1.0, gam:float=1.4):
    """
    Computes the stagnation or total temperature T0.

    T0 is the temperature that would exist if the flow were adiabatically
    brought to rest. T0 remains constant where flowfield is adiabatic.

    Parameters
    ----------
    M : Any, optional
        Mach number, by default 1.0
    T : Any, optional
        static temperature, by default 1.0
    gam : float, optional
        ratio of specific heats, by default 1.4

    Returns
    -------
    Any
        T0: stagnation temperature AND

      T0/T: stagnation temperature ratio if `T`==1.0 AND

     T0/T*: stagnation temperature ratio (sonic) if `T`==1.0 and `M`==1.0
    """
    return T*(1 + (gam-1)/2 * M*M)

def mach_from_temperature_ratio(T0_ratio, gam:float=1.4):
    """
    Computes the Mach number from total temperature ratio T0/T.

    This function is the inverse of `total_temperature`.

    Parameters
    ----------
    T0_ratio : Any
        total temperature ratio
    gam : float, optional
        ratio of specific heats, by default 1.4

    Returns
    -------
    Any
        Mach number
    """    
    return ( 2/(gam-1) * (T0_ratio - 1) )**0.5

def total_pressure(M=1.0, p=1.0, gam:float=1.4):
    """
    Computes the stagnation or total pressure p0.

    p0 is the pressure that would exist if the flow were isentropically
    brought to rest. p0 remains constant where flowfield is isentropic.

    Parameters
    ----------
    M : Any, optional
        Mach number, by default 1.0
    p : Any, optional
        static pressure, by default 1.0
    gam : float, optional
        ratio of specific heats, by default 1.4

    Returns
    -------
    Any
        p0: stagnation pressure AND

        p0/p: stagnation pressure ratio if `p`==1.0 AND

       p0/p*: stagnation pressure ratio (sonic) if `p`==1.0 and `M`==1.0
    """    
    return p*total_temperature(M=M, T=1.0, gam=gam)**(gam/(gam-1))

def total_density(M=1.0, rho=1.0, gam:float=1.4):
    """
    Computes the stagnation or total density rho0.

    rho0 is the density that would exist if the flow were isentropically
    brought to rest. rho0 remains constant where flowfield is isentropic.

    Parameters
    ----------
    M : Any, optional
        Mach number, by default 1.0
    rho : Any, optional
        static density, by default 1.0
    gam : float, optional
        ratio of specific heats, by default 1.4

    Returns
    -------
    Any
        rho0: stagnation density AND

    rho0/rho: stagnation density ratio if `rho`==1.0 AND

    rho0/rho*: stagnation density ratio (sonic) if `rho`==1.0 and `M`==1.0
    """
    return rho*total_temperature(M=M, T=1.0, gam=gam)**(1/(gam-1))

def total_speed_sound(M=1.0, a=1.0, gam:float=1.4):
    """
    Computes the stagnation or total speed of sound a0.

    a0 is the speed of sound that would exist if the flow were adiabatically
    brought to rest. a0 remains constant where flowfield is adiabatic.

    Parameters
    ----------
    M : Any, optional
        Mach number, by default 1.0
    a : Any, optional
        static speed of sound, by default 1.0
    gam : float, optional
        ratio of specific heats, by default 1.4

    Returns
    -------
    Any
        a0: stagnation speed of sound AND

      a0/a: stagnation speed of sound ratio if `a`==1.0 AND

     a0/a*: stagnation speed of sound ratio (sonic) if `a`==1.0 and `M`==1.0
    """
    return a*(total_temperature(M=M, T=1.0, gam=gam))**0.5

def area_mach_relation(area_ratio:float=None, M:float=None, gam:float=1.4,
                       is_supersonic:bool=True):
    """
    Computes the area ratio A/A* or Mach number in an isentropic process.

    The equation assumes isentropic flow for a calorically perfect gas.

    Parameters
    ----------
    area_ratio : float, optional
        Area ratio A/A* of any location in duct wrt sonic throat area,
            by default None
    M : float, optional
        Mach number, by default None
    gam : float, optional
        ratio of specific heats gamma, by default 1.4
    is_supersonic : bool, optional
        if `M` is unknown, is flow supersonic or subsonic, by default True

    Returns
    -------
    float
        area ratio A/A* OR

        Mach number M

    Raises
    ------
    ValueError
        Both A/A* and M specified, or neither are specified
    """    
    guess_mach = True if M is None and area_ratio is not None else False
    func = lambda _m: (
        _m**-2 * ( 
            2/(gam+1)*total_temperature(M=_m, T=1.0, gam=gam)
        )**((gam+1)/(gam-1))
    )
    if guess_mach:
        mach_guess = 2.0 if is_supersonic else 0.2
        compute_mach = lambda _m, _Aratio: (_Aratio**2 - func(_m))
        return fsolve(compute_mach, mach_guess, area_ratio)[0]
    elif not guess_mach:
        return func(M)**0.5

    else:
        raise ValueError("Specify either A/A* or M, not both.")
    