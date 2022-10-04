"""
Compressible, inviscid flow relations.

These functions are basic definitions pertaining to 1-,2-, and 3-D
compressible, inviscid flow.
"""
from scipy.optimize import fsolve

import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(os.path.dirname(currentdir)))
sys.path.insert(0, parentdir)
import thermodynamics.caloric_perfect as calp



def mach(u,a):
    """
    `M` computes Mach number.

    Definition of Mach number. It is a ratio of the kinetic to internal
    energies of a gas.

    Parameters:
    u (float): velocity
    a (float): speed of sound

    Returns:
    M: Mach number
    """
    return u/a



def speed_sound(t=None,R=None,rho=None,p=None,gam=1.4):
    """
    `speed_sound` computes speed of sound.

    Definition of speed of sound for a perfect (thermally+calorically) gas.
    Relation is invalid for chemically reacting and real gases.

    Parameters:
    t (float): temperature
    R (float): specific gas constant
    rho (float): density
    p (float): pressure
    gam (float): ratio of specific heats gamma

    Returns:
    a: speed of sound
    """
    if t is not None and R is not None and rho is None and p is None:
        return (gam * R * t)**(0.5)
    elif rho is not None and p is not None and t is None and R is None:
        return (gam * p / rho)**(0.5)
    else:
        raise Exception("Specify (gamma,R,T) or (gamma,p,rho).")



def lookup_table(M=None,p0_ratio=None,r0_ratio=None,t0_ratio=None,
                           a0_ratio=None,area_ratio=None,gam=1.4,
                           regime="supersonic"):
    """
    `lookup_table` provides all isentropic flow variables for a given input.

    Equivalent to a line from lookup tables for isentropic, calorically perfect gases 
    (1D and quasi-1D). Function is recursive.

    Parameters:
    M (float): Mach number
    p0_ratio (float): total pressure ratio p0/p
    r0_ratio (float): total density ratio rho0/rho
    t0_ratio (float): total temperature ratio T0/T
    a0_ratio (float): total speed of sound ratio a0/a
    gam (float): ratio of specific heats gamma
    regime (string): `supersonic` or `subsonic` regime

    Returns:
    Dict of all unspecified isentropic flow parameters for a given input.
    """
    if (M is not None and  # Mach number specified
         p0_ratio is None and r0_ratio is None and
         t0_ratio is None and a0_ratio is None and
         area_ratio is None):
        p0_ratio = total_pressure(M=M,p=1.0,gam=gam)
        r0_ratio = total_density(M=M,rho=1.0,gam=gam)
        t0_ratio = total_temperature(M=M,t=1.0,gam=gam)
        a0_ratio = total_speed_sound(M=M,a=1.0,gam=gam)
        area_ratio = area_mach_relation(M=M,gam=gam)
        return {"p0_ratio":p0_ratio,
            "r0_ratio":r0_ratio,
            "t0_ratio":t0_ratio,
            "a0_ratio":a0_ratio,
            "area_ratio":area_ratio,
            "M":M,
            "gam":gam,
            "regime":regime.lower()}

    elif (p0_ratio is not None and  # p0/p specified
          M is None and r0_ratio is None and
          t0_ratio is None and a0_ratio is None and
          area_ratio is None):
        t0_ratio = calp.isentropic_process(p21=p0_ratio,gamma=gam)["t21"]
        M = mach_from_temperature_ratio(t0_ratio=t0_ratio,gam=gam)
        return lookup_table(M=M,gam=gam)

    elif (r0_ratio is not None and  # rho0/rho specified
          p0_ratio is None and M is None and
          t0_ratio is None and a0_ratio is None and
          area_ratio is None):
        t0_ratio = calp.isentropic_process(r21=r0_ratio,gamma=gam)["t21"]
        M = mach_from_temperature_ratio(t0_ratio=t0_ratio,gam=gam)
        return lookup_table(M=M,gam=gam)

    elif (t0_ratio is not None and  # T0/T specified
          p0_ratio is None and r0_ratio is None and
          M is None and a0_ratio is None and
          area_ratio is None):
        M = mach_from_temperature_ratio(t0_ratio=t0_ratio,gam=gam)
        return lookup_table(M=M,gam=gam)

    elif (a0_ratio is not None and  # a0/a specified
          p0_ratio is None and r0_ratio is None and
          t0_ratio is None and M is None and
          area_ratio is None):
        t0_ratio = calp.isentropic_process(a21=a0_ratio,gamma=gam)["t21"]
        M = mach_from_temperature_ratio(t0_ratio=t0_ratio,gam=gam)
        return lookup_table(M=M,gam=gam)

    elif (area_ratio is not None and  # A/A* specified
          p0_ratio is None and r0_ratio is None and
          t0_ratio is None and a0_ratio is None and
          M is None):
        M = area_mach_relation(area_ratio=area_ratio,gam=gam,regime=regime)
        return lookup_table(M=M,gam=gam)

    else: # over or under-specified
        raise Exception("Specify Mach number, p0/p, rho0/rho, T0/T, a0/a, or A/A*.")



def total_temperature(M=1.0,t=1.0,gam=1.4):
    """
    `total_temperature` computes the stagnation/total temperature T0.

    T0 is the temperature that would exist if the flow were adiabatically
    brought to rest. T0 remains constant where flowfield is adiabatic.

    Parameters:
    M (float): Mach number.
    t (float): static temperature.
    gam (float): gamma, ratio of specific heats.

    Returns:
    T0: stagnation temperature
    T0/T: stagnation temperature ratio when `t`==1.0
    T0/T*: stagnation temperature ratio when `t`==1.0, `M`==1.0
    """
    return t*(1 + (gam-1)/2 * M*M)



def mach_from_temperature_ratio(t0_ratio,gam=1.4):
    """
    `mach_from_temperature_ratio` computes the Mach number from total temperature ratio T0/T.

    This function is the inverse of `total_temperature`.

    Parameters:
    t0_ratio (float): total temperature ratio T0/T
    gam (float): gamma, ratio of specific heats.

    Returns:
    M: Mach number
    """
    return ( 2/(gam-1) * (t0_ratio - 1) )**0.5



def total_pressure(M=1.0,p=1.0,gam=1.4):
    """
    `total_pressure` computes the stagnation/total pressure p0.

    p0 is the pressure that would exist if the flow were isentropically
    brought to rest. p0 remains constant where flowfield is isentropic.

    Parameters:
    M (float): Mach number.
    p (float): static pressure.
    gam (float): gamma, ratio of specific heats.

    Returns:
    p0: stagnation pressure
    p0/p: stagnation pressure ratio when `p`==1.0
    p0/p*: stagnation pressure ratio when `t`==1.0, `M`==1.0
    """
    return p*total_temperature(M=M,t=1.0,gam=gam)**(gam/(gam-1))



def total_density(M=1.0,rho=1.0,gam=1.4):
    """
    `total_density` computes the stagnation/total density rho0.

    rho0 is the density that would exist if the flow were isentropically
    brought to rest. rho0 remains constant where flowfield is isentropic.

    Parameters:
    M (float): Mach number.
    rho (float): static density.
    gam (float): gamma, ratio of specific heats.

    Returns:
    rho0: stagnation density
    rho0/rho: stagnation density ratio when `rho`==1.0
    rho0/rho*: stagnation density ratio when `rho`==1.0, `M`==1.0
    """
    return rho*total_temperature(M=M,t=1.0,gam=gam)**(1/(gam-1))



def total_speed_sound(M=1.0,a=1.0,gam=1.4):
    """
    `total_speed_sound` computes the stagnation/total speed of sound a0.

    a0 is the speed of sound that would exist if the flow were isentropically
    brought to rest. a0 remains constant where flowfield is isentropic.

    Parameters:
    M (float): Mach number.
    a (float): static speed of sound.
    gam (float): gamma, ratio of specific heats.

    Returns:
    a0: stagnation speed of sound
    a0/a: stagnation speed of sound ratio when `a`==1.0
    a0/a*: stagnation speed of sound ratio when `a`==1.0, `M`==1.0
    """
    return a*total_temperature(M=M,t=1.0,gam=gam)**0.5



def area_mach_relation(area_ratio=None,M=None,gam=1.4,regime="supersonic"):
    """
    `area_mach_relation` computes the area ratio A/A* or Mach number in an isentropic process.

    The equation assumes isentropic flow for a calorically perfect gas.

    Parameters:
    area_ratio: Area ratio A/A* of any location in duct to sonic throat area.
    M (float): Mach number.
    gam (float): gamma, ratio of specific heats.
    regime (string): `supersonic` or `subsonic` regime

    Returns:
    A/A*: area ratio OR
    M: Mach number
    """
    if area_ratio is not None and M is None:

        if regime.lower() == "subsonic":
            mach_guess = 0.2
        elif regime.lower() == "supersonic":
            mach_guess = 2.0
        else:
            raise Exception("Specify 'subsonic' or 'supersonic' regime.")

        func = lambda M,area_ratio: area_ratio**2 - ( 2/(gam+1)
                *total_temperature(M=M,t=1.0,gam=gam) )**((gam+1)/(gam-1))/M/M
        return fsolve(func, mach_guess,area_ratio )[0]

    elif M is not None and area_ratio is None:
        return ( ( 2/(gam+1)
            *total_temperature(M=M,t=1.0,gam=gam) )**((gam+1)/(gam-1))/M/M )**0.5

    else:
        raise Exception("Specify either A/A* or M, not both.")
    