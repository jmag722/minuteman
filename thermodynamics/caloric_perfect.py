import numpy as np
from multimethod import multimethod
import sympy as sp

import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)

import mymath.mymath as mm

kB = 1.380649e-23  # J/K, Boltzmann constant
NA = 6.02214076e23  # count/mol, Avogadro's number
RU = 8.31446261815324  # J/K/mol, Universal Gas constant (kB*NA)

class IdealGasLawSolver():
    # can't use N for # of moles, seems reserved by sympy
    variables_str = ["p","V", "rho", "n","Nm", "R", "T","kB", "RU"]
    RHO_EQ = "p - rho*R*T"
    PARTICLE_EQ = "p - n*kB*T"
    VOL_EQ = "p*V - Nm*RU*T"
    def __init__(self):
        pass

    def solve(self,x,values_dict,eq=RHO_EQ):
        if "RU" in eq:
            values_dict["RU"] = RU
        if "kB" in eq:
            values_dict["kB"] = kB
        for key,val in values_dict.items():
            if key not in self.variables_str:
                raise Exception("Variable {} not supported in ideal gas eq".format(key))
        aes = mm.AlgebraicEquationSolver()
        return aes.solve(x,values_dict,eq)

@multimethod
def R(m:float,is_molar:bool):
    return R(np.asarray(m),is_molar)
@multimethod
def R(m:np.ndarray,is_molar:bool):
    if is_molar: # m in kg/mol
        return RU/m
    else: # m in kg/particle
        return kB/m
@multimethod
def R(cp:float, cv:float):
    return R(np.asarray(cp),np.asarray(cv))
@multimethod
def R(cp:np.ndarray, cv:np.ndarray):
    return cp - cv

def gamma(cp,cv):
    return cp/cv

def cp(gamma,R):
    return gamma*cv(gamma,R)

def cv(gamma,R):
    return R/(gamma-1)

def entropy(T2_T1=None,p2_p1=None,v2_v1=None,
            cp=None,cv=None,R=None,s1=0.0):
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