"""
Compressible, inviscid flow relations.

These functions are basic definitions pertaining to 1-,2-, and 3-D
compressible, inviscid flow.
"""

import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)

from thermodynamics.caloric_perfect import R_AIR_SI

def M(u,a):
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

def a(t=None,rho=None,p=None,gam=1.4,R=R_AIR_SI):
    """
    `a` computes speed of sound.

    Definition of speed of sound for a perfect (thermally+calorically) gas.
    Relation is invalid for chemically reacting and real gases.

    Parameters:
    t (float): temperature
    rho (float): density
    p (float): pressure
    gam (float): ratio of specific heats gamma
    R (float): specific gas constant

    Returns:
    a: speed of sound
    """
    if t is not None:
        return (gam * R * t)**(0.5)
    elif rho is not None and p is not None:
        return (gam * p / rho)**(0.5)
    else:
        raise Exception("Specify gamma,R,T or gamma,p,rho.")


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

def total_sound_speed(M=1.0,a=1.0,gam=1.4):
    """
    `total_sound_speed` computes the stagnation/total speed of sound a0.

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