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


# for perfect (thermally+calorically) and reacting gases
class IdealGasLawSolver():
    RHO_EQ = "p - rho*R*T"
    PARTICLE_EQ = "p - n*kB*T"
    # can't use N for # of moles, seems reserved by sympy
    VOL_EQ = "p*V - Nm*RU*T"
    def __init__(self):
        pass
    def solve(self,x,values_dict,units:Units=Units.SI):
        varlist = {x}
        varlist.update(list(values_dict))
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
            values_dict["kB"] = boltz
        elif "p" in varlist and "V" in varlist\
            and "Nm" in varlist and "T" in varlist:
            eq=self.VOL_EQ
            values_dict["RU"] = ru
        else:
            raise Exception("Variables {} are not supported for isentropic\
                relations".format(varlist))
            
        aes = mm.AlgebraicEquationSolver()
        return aes.solve(x,values_dict,eq)

@multimethod
def R(m:float,is_molar:bool,units:Units=Units.SI):
    return R(np.asarray(m),is_molar,units)
@multimethod
def R(m:np.ndarray,is_molar:bool,units:Units=Units.SI):
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
    return R(np.asarray(cp),np.asarray(cv))
@multimethod
def R(cp:np.ndarray, cv:np.ndarray):
    return cp - cv

def gamma(cp,cv):
    return cp/cv

def cp(gamma,R=R_AIR_SI):
    return gamma*cv(gamma,R)

def cv(gamma,R=R_AIR_SI):
    return R/(gamma-1)


# for calorically perfect gases
def entropy(T2_T1=None,p2_p1=None,v2_v1=None,
            cp=None,cv=None,R=R_AIR_SI,s1=0.0):
    if (T2_T1 and p2_p1 and cp and R)  is not None:
        expr = cp*np.log(T2_T1) - R * np.log(p2_p1)
    elif (T2_T1 and v2_v1 and cv and R) is not None:
        expr = cv*np.log(T2_T1) + R * np.log(v2_v1)
    elif (p2_p1 and v2_v1 and cp and cv):
        expr = cv*np.log(p2_p1) + cp*np.log(v2_v1)
    else:
        error_msg="Must specify one of the following: \n\
                1) T2/T1, v2/v1, cp, R, OR\n\
                2) T2/T1, v2/v1, cv, R, OR\n\
                3) p2/p1, v2/v1, cp, cv"
        raise Exception(error_msg)
    return s1 + expr # if s1=0, returns (s2-s1), else s2

class IsentropicRelationSolver():
    # PT_EQ = "p2_p1 - T2_T1**(gam/(gam-1))"
    # PR_EQ = "p2_p1 - r2_r1**gam"
    # PA_EQ = "p2_p1 - a2_a1**(2*gam/(gam-1))"
    # RT_EQ = "r2_r1 - T2_T1**(1/(gam-1))"
    # RA_EQ = "r2_r1 - a2_a1**(2/(gam-1))"
    # TA_EQ = "T2_T1 - a2_a1*a2_a1"
    def __init__(self):
        pass
    def solve(self,x,input:tuple,gam=1.4):
        varlist = {x,input[0]}
        # values_dict = dict([input,("gam",gam)])
        ans = None
        if "p2_p1" in varlist and "T2_T1" in varlist:
            if x == "p2_p1":
                ans = input[1]**(gam/(gam-1))
            else:
                ans = input[1]**((gam-1)/gam)
            # eq=self.PT_EQ
        elif "p2_p1" in varlist and "r2_r1" in varlist:
            if x == "p2_p1":
                ans = input[1]**gam
            else:
                ans = input[1]**1/gam
            # eq = self.PR_EQ
        elif "p2_p1" in varlist and "a2_a1" in varlist:
            if x == "p2_p1":
                ans = input[1]**(2*gam/(gam-1))
            else:
                ans = input[1]**((gam-1)/2/gam)
            # eq = self.PA_EQ
        elif "T2_T1" in varlist and "r2_r1" in varlist:
            if x == "T2_T1":
                ans = input[1]**(gam-1)
            else:
                ans = input[1]**(1/(gam-1))
            # eq = self.RT_EQ
        elif "T2_T1" in varlist and "a2_a1" in varlist:
            if x == "T2_T1":
                ans = input[1]**2
            else:
                ans = input[1]**0.5
            # eq = self.TA_EQ
            # del values_dict["gam"]
        elif "r2_r1" in varlist and "a2_a1" in varlist:
            if x == "r2_r1":
                ans = input[1]**(2/(gam-1))
            else:
                ans = input[1]**((gam-1)/2)
            # eq = self.RA_EQ
        else:
            raise Exception("Variable {} or {} are not supported for isentropic\
                relations".format(x,input[0]))
        # aes = mm.AlgebraicEquationSolver()
        # return aes.solve(x,values_dict,eq=eq)
        return ans