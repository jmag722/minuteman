"""
Calorically perfect gases are those where gases are chemically unreactive
and intermolecular forces are neglected. Internal energy and enthalpy are
functions of temperature only AND the specific heats are constant.

This is the case for atmospheric air below ~1000 K. However, at higher
temperatures where O2 and N2 vibrational motion/excitation becomes
important, the gas is no longer calorically perfect.
"""

import enum
import numpy as np
import scipy.constants as scc

import minuteman.mymath.mymath as mm

NA = scc.Avogadro        # count/mol, Avogadro's number
kB_SI = scc.Boltzmann         # J/K, Boltzmann constant
kB_IMP = 5.657302466e-24  # ft-lb/R, Boltzmann constant
RU_SI = scc.R     # J/K/kg-mol, Universal gas constant (kB*NA)
RU_IMP_LBM = 1545.349     # ft-lbf/R/lbm-mol, Universal gas constant
RU_IMP_SLUG = 49720.0       # ft-lbf/R/slug-mol, Universal gas constant
R_AIR_SI = 287.058           # J/kg/K, Specific gas constant for dry air
R_AIR_IMP_LBM = 53.3533   # ft-lbf/lbm/R, Specific gas constant for dry air
R_AIR_IMP_SLUG = 1716.49  # ft-lbf/slug/R, Specific gas constant for dry air

class Units(enum.Enum):
    SI = 0
    IMP_LBM = 1
    IMP_SLUG = 2

class IdealGasLawSolver():    
    """
    Ideal gas equation solver.

    Ideal gas equation valid for perfect (thermally+calorically) 
    and reacting gases.
    """

    RHO_EQ = "p - rho*R*T"
    PARTICLE_EQ = "p - n*kB*T"
    # can't use N for # of moles, it is reserved by sympy
    VOL_EQ = "p*V - Nm*RU*T"
    def __init__(self):
        pass
    def solve(self, unknown:str, knowns:dict, units:Units=Units.SI):
        """
        Solve the ideal gas equation.

        Parameters
        ----------
        unknown : str
            State variable to compute
        knowns : dict
            Key value pairs of known state variables
        units : Units, optional
            SI, IMPERIAL (slug or lbm), by default Units.SI

        Returns
        -------
        float
            Desired unknown

        Raises
        ------
        ValueError
            Incorrect or inconsistent inputs specified
        """
        varlist = {unknown}
        varlist.update(list(knowns))
        if units==Units.SI:
            ru = RU_SI
            boltz = kB_SI
        elif units==Units.IMP_LBM:
            ru = RU_IMP_LBM
            boltz = kB_IMP
        elif units==Units.IMP_SLUG:
            ru = RU_IMP_SLUG
            boltz = kB_IMP
        if all(var in varlist for var in ["p", "rho", "R", "T"]):
            eq=self.RHO_EQ
        elif all(var in varlist for var in ["p", "n", "T"]):
            eq=self.PARTICLE_EQ
            knowns["kB"] = boltz
        elif all(var in varlist for var in ["p", "V", "Nm", "T"]):
            eq=self.VOL_EQ
            knowns["RU"] = ru
        else:
            raise ValueError("Variables {} are not supported for isentropic\
                relations".format(varlist))
            
        return mm.solve_algebraic_eqn(unknown, knowns, eq)

def R(mass=None, cp=None, cv=None, is_molar:bool=True,
      units:Units=Units.SI):    
    """
    Compute specific gas constant.

    Valid for perfect (thermally & calorically) gases.

    Parameters
    ----------
    mass : Any, optional
        gas mass (molar or per particle), by default None
    cp : Any, optional
        specific heat (constant pressure), by default None
    cv : Any, optional
        specific heat (constant volume), by default None
    is_molar : bool, optional
        mass is per mol or per particle, by default True
    units : Units, optional
        SI or IMPERIAL (slug or lbm), by default Units.SI

    Returns
    -------
    Any
        specific gas constant

    Raises
    ------
    ValueError
        User did not specify mass OR both cp and cv.
    """    
    if mass is not None:
        return _spec_gas_const_from_mass(mass, is_molar=is_molar, units=units)
    elif all(var is not None for var in [cp, cv]):
        return _spec_gas_const_from_spec_heat(cp=cp, cv=cv)
    else:
        raise ValueError("Must specify mass OR both cp and cv.")
        
def _spec_gas_const_from_mass(mass, is_molar:bool, units:Units):
    if is_molar: # mass per mol
        if units == Units.SI:
            return RU_SI/mass
        elif units == Units.IMP_LBM:
            return RU_IMP_LBM/mass
        else:
            return RU_IMP_SLUG/mass
    else: # mass per particle
        if units == Units.SI:
            return kB_SI/mass
        else:
            return kB_IMP/mass

def _spec_gas_const_from_spec_heat(cp, cv):           
    return cp - cv

def gamma(cp,cv):
    """
    Compute the ratio of specific heats.

    Valid for perfect (thermally & calorically) gases.

    Parameters
    ----------
    cp : Any
        specific heat (constant pressure)
    cv : Any
        specific heat (constant volume)

    Returns
    -------
    Any
        ratio of specific heats
    """    
    return cp/cv

def cp(gam:float, R:float=R_AIR_SI):
    """
    Computes the specific heat at constant pressure.

    Valid for perfect (thermally & calorically) gases.

    Parameters
    ----------
    gam : float
        ratio of specific heats
    R : float, optional
        specific gas constant, by default R_AIR_SI

    Returns
    -------
    float
        specific heat at constant pressure
    """    
    return gam*cv(gam, R)
    
def cv(gam:float, R:float=R_AIR_SI):
    """
    Computes the specific heat at constant volume.

    Valid for perfect (thermally & calorically) gases.

    Parameters
    ----------
    gam : float
        ratio of specific heats
    R : float, optional
        specific gas constant, by default R_AIR_SI

    Returns
    -------
    float
        specific heat at constant volume
    """
    return R/(gam-1)

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
    return np.log(p / rho**gam) * cv(gam=gam, R=R)

def entropy(t21:float=None, p21:float=None, v21:float=None,
            cp:float=None, cv:float=None, R:float=R_AIR_SI, s1:float=0.0):
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
        specific gas constant, by default R_AIR_SI
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
        error_msg="Must specify one of the following: \n\
                1) T2/T1, v2/v1, cp, R, OR\n\
                2) T2/T1, v2/v1, cv, R, OR\n\
                3) p2/p1, v2/v1, cp, cv"
        raise ValueError(error_msg)
    return s1 + expr # if s1==0, returns (s2-s1), else s2

def isentropic_process(p21:float=None, t21:float=None, r21:float=None,
                       a21:float=None, gam:float=1.4):
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
    return {"p21":p21, "t21":t21, "r21":r21, "a21":a21, "gam":gam}

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

def heat_flux(q:float=None, c:float=None, t1:float=None, t2:float=None,
              m:float=1.0):
    """
    Computes the heat addition due to the temperature change of the substance.

    Specify exactly three of the following input variables: `q`, `t2`, `t1`, or `c`. 
    The unspecified variable (assigned `None`) will be returned. 
    If `m==1.0` (default), values computed will be per unit mass.

    Valid for general calorimetry and calorically perfect gases only.

    Parameters
    ----------
    q : float, optional
        Amount of heat added. This is specific heat if `m==1.0`, by default None
    c : float, optional
        Specific heat capacity at constant pressure or volume, by default None
    t1 : float, optional
        Temperature before heat transfer, by default None
    t2 : float, optional
        Temperature after heat transfer, by default None
    mass : float, optional
        Mass of substance, by default 1.0

    Returns
    -------
    float
        q, c, m, t2, or t1

    Raises
    ------
    ValueError
        Incorrect or inconsistent inputs supplied.
    """    
    condition = lambda desired, user_specified: (
        desired is None and all(var is not None for var in user_specified)
    )
    if condition(q, [t1, t2, c]):
        return m * c * (t2 - t1)  # q
    elif condition(t2, [q, t1, c]):
        return q/c/m + t1  # T2
    elif condition(t1, [q, t2, c]):
        return t2 - q/c/m  # T1
    elif condition(c, [q, t1, t2]):
        return q/m/(t2-t1) # c
    elif condition(m, [q, t1, t2, c]):
        return q/c/(t2-t1) # m
    else:
        raise ValueError("Must specify exactly three of the following: `q`, `T1`, `T2`, `c`.")