"""
Calorically perfect gases are those where gases are chemically unreactive
and intermolecular forces are neglected. Internal energy and enthalpy are
functions of temperature only AND the specific heats are constant.

This is the case for atmospheric air below ~1000 K. However, at higher
temperatures where O2 and N2 vibrational motion/excitation becomes
important, the gas is no longer calorically perfect.
"""

import numpy as np
from multimethod import multimethod
import enum

import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)

import mymath.mymath as mm

NA = 6.02214076e23        # count/mol, Avogadro's number
kB_SI = 1.380649e-23         # J/K, Boltzmann constant
kB_IMP = 5.657302466e-24  # ft-lb/R, Boltzmann constant
RU_SI = 8.31446261815324     # J/K/kg-mol, Universal gas constant (kB*NA)
RU_IMP_LBM = 1545.349     # ft-lbf/R/lbm-mol, Universal gas constant
RU_IMP_SLUG = 49720       # ft-lbf/R/slug-mol, Universal gas constant
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
    def solve(self,x,values,units:Units=Units.SI):
        """
        `solve` solves the ideal gas equation.

        Parameters:
        x (str): State variable to compute
        values (dict): Known state variables.
        units (Units): SI, IMPERIAL (slug or lbm)

        Returns:
        x: value
        """
        varlist = {x}
        varlist.update(list(values))
        if units==Units.SI:
            ru = RU_SI
            boltz = kB_SI
        elif units==Units.IMP_LBM:
            ru = RU_IMP_LBM
            boltz = kB_IMP
        elif units==Units.IMP_SLUG:
            ru = RU_IMP_SLUG
            boltz = kB_IMP

        if "p" in varlist and "rho" in varlist\
            and "R" in varlist and "T" in varlist:
            eq=self.RHO_EQ
        elif "p" in varlist and "n" in varlist\
            and "T" in varlist:
            eq=self.PARTICLE_EQ
            values["kB"] = boltz
        elif "p" in varlist and "V" in varlist\
            and "Nm" in varlist and "T" in varlist:
            eq=self.VOL_EQ
            values["RU"] = ru
        else:
            raise Exception("Variables {} are not supported for isentropic\
                relations".format(varlist))
            
        aes = mm.AlgebraicEquationSolver()
        return aes.solve(x,values,eq)

@multimethod
def R(m:float,is_molar:bool,units:Units=Units.SI):
    """
    `R` computes specific gas constant.

    Parameters:
    m (float): mass (per particle or per mol)
    is_molar (bool): is mass per particle or per mol
    units (Units): SI or IMPERIAL (slug or lbm)

    Returns:
    R: gas constant
    """
    return R(np.asarray(m),is_molar,units)
@multimethod
def R(m:np.ndarray,is_molar:bool,units:Units=Units.SI):
    """
    `R` computes specific gas constant.

    Parameters:
    m (numpy.ndarray): mass (per particle or per mol)
    is_molar (bool): is mass per particle or per mol
    units (Units): SI or IMPERIAL (slug or lbm)

    Returns:
    R: gas constant
    """
    if is_molar: # m in mass/mol
        if units == Units.SI:
            return RU_SI/m
        elif units == Units.IMP_LBM:
            return RU_IMP_LBM/m
        else:
            return RU_IMP_SLUG/m
    else: # m in mass/particle
        if units == Units.SI:
            return kB_SI/m
        else:
            return kB_IMP/m

@multimethod
def R(cp:float, cv:float):
    """
    `R` computes specific gas constant.

    Valid for perfect (thermally & calorically) gases.

    Parameters:
    cp (float): specific heat (constant p)
    cv (float): specific heat (constant v)

    Returns:
    Gas constant R
    """
    return R(np.asarray(cp),np.asarray(cv))
@multimethod
def R(cp:np.ndarray, cv:np.ndarray):
    """
    `R` computes specific gas constant.

    Valid for perfect (thermally & calorically) gases.

    Parameters:
    cp (numpy.ndarray): specific heat (constant p)
    cv (numpy.ndarray): specific heat (constant v)

    Returns:
    Gas constant R
    """
    return cp - cv

def gamma(cp,cv):
    """
    `gamma` computes the ratio of specific heats.

    Parameters:
    cp (float): specific heat (constant p)
    cv (float): specific heat (constant v)

    Returns:
    gamma: ratio of specific heats
    """
    return cp/cv

def cp(gamma,R=R_AIR_SI):
    """
    `cp` computes the specific heat at consant pressure.

    Valid for perfect (thermally & calorically) gases.

    Parameters:
    gamma (float): ratio of specific heats
    R (float): specific gas constant

    Returns:
    cp: specific heat at constant pressure
    """
    return gamma*cv(gamma,R)
    
def cv(gamma,R=R_AIR_SI):
    """
    `cv` computes the specific heat at consant volume.

    Valid for perfect (thermally & calorically) gases.

    Parameters:
    gamma (float): ratio of specific heats
    R (float): specific gas constant

    Returns:
    cv: specific heat at constant volume
    """
    return R/(gamma-1)


def entropy(t21=None,p21=None,v21=None,
            cp=None,cv=None,R=R_AIR_SI,s1=0.0):
    """
    `entropy` computes the change in entropy for a calorically perfect gas.

    Parameters:
    t21 (float): temperature ratio T2/T1
    p21 (float): pressure ratio p2/p1
    v21 (float): specific volume ratio v2/v1
    cp (float): specific heat (constant p)
    cv (float): specific heat (constant v)
    R  (float): specific gas constant
    s1 (float): entropy at state 1.

    Returns:
    ds: change in entropy  if s1==0
    s2: entropy at state 2, s2, if s1 is given.
    """
    if (t21 and p21 and cp and R)  is not None:
        expr = cp*np.log(t21) - R * np.log(p21)
    elif (t21 and v21 and cv and R) is not None:
        expr = cv*np.log(t21) + R * np.log(v21)
    elif (p21 and v21 and cp and cv):
        expr = cv*np.log(p21) + cp*np.log(v21)
    else:
        error_msg="Must specify one of the following: \n\
                1) T2/T1, v2/v1, cp, R, OR\n\
                2) T2/T1, v2/v1, cv, R, OR\n\
                3) p2/p1, v2/v1, cp, cv"
        raise Exception(error_msg)
    return s1 + expr # if s1=0, returns (s2-s1), else s2


def isentropic_process(p21=None,t21=None,r21=None,a21=None,gamma=1.4):
    """
    `isentropic_process` returns the state of an isentropic gas.

    Isentropic relations valid for isentropic (adiabatic+reversible),
    calorically perfect gases only.

    Parameters:
    p21 (float): pressure ratio p2/p1
    t21 (float): temperature ratio T2/T1
    r21 (float): density ratio rho2/rho1
    a21 (float): speed of sound ratio a2/a1
    gamma (float): ratio of specific heats gamma

    Returns:
    dict of p2/p1, t2/t1, rho2/rho1, a2/a1, and gamma.
    """
    if p21 is not None:
        t21 = p21**((gamma-1)/gamma)
        r21 = p21**(1/gamma)
        a21 = p21**((gamma-1)/gamma/2)
    elif t21 is not None:
        p21 = t21**(gamma/(gamma-1))
        r21 = t21**(1/(gamma-1))
        a21 = t21**0.5
    elif r21 is not None:
        p21 = r21**gamma
        t21 = r21**(gamma-1)
        a21 = r21**((gamma-1)/2)
    elif a21 is not None:
        p21 = a21**(2*gamma/(gamma-1))
        t21 = a21*a21
        r21 = a21**(2/(gamma-1))
    else:
        raise Exception("Input p2/p1, T2/T1, rho2/rho1, or a2/a1.")
    return {"p21":p21,"t21":t21,"r21":r21,"a21":a21,"gamma":gamma}


def heat_flux(q=None,c=None,t1=None,t2=None,m=1.0):
    """
    `heat_flux` computes the heat addition due to the total temperature
    change of the flow.

    Specify exactly three of the following input variables: `q`, `t2`, `t1`, or `c`. 
    The unspecified variable (assigned `None`) will be returned. 
    If `m==1.0` (default), values computed will be per unit mass.

    Parameters:
    q (float): Amount of heat added. This is specific heat if `m==1.0`.
    c (float): Specific heat capacity.
    t1 (float): Temperature before heat addition.
    t2 (float): Temperature after heat addition.
    m (float): Mass of substance.
    
    Returns:
    q,c,m,t2, or t1 (float)
    """
    if q is None and t2 is not None and t1 is not None and c is not None:
        return m * c * (t2 - t1)  # q
    elif t2 is None and q is not None and t1 is not None and c is not None:
        return q/c/m + t1  # T2
    elif t1 is None and q is not None and t2 is not None and c is not None:
        return t2 - q/c/m  # T1
    elif c is None and q is not None and t2 is not None and t1 is not None:
        return q/m/(t2-t1) # c
    elif m is None and q is not None and t2 is not None and t1 is not None and c is not None:
        return q/c/(t2-t1) # m
    else:
        raise Exception("Must specify exactly three of the following: `q`, `T1`, `T2`, `c`.")