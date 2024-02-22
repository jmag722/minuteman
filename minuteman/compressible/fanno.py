import numpy as np
from scipy.optimize import fsolve
import minuteman.utils.arg_checks as ac

def lookup_table(p_ratio:float=None, r_ratio:float=None, T_ratio:float=None,
                 ds_R_ratio:float=None, p0_ratio:float=None,
                 fanno_param:float=None, M:float=None, gam:float=1.4, 
                 is_supersonic:bool=True):
    """
    Computes Fanno flow parameters (reference Mach=1.0 only) for a given input.

    This form is valid for calorically perfect gases only, where gamma
    is constant (M < 5). 

    Parameters
    ----------
    p_ratio : float, optional
        pressure ratio p/p*, by default None
    r_ratio : float, optional
        density ratio rho/rho*, by default None
    T_ratio : float, optional
        temperature ratio T/T*, by default None
    ds_R_ratio : float, optional
        entropy ratio (s*-s)/R, by default None
    p0_ratio : float, optional
        total pressure ratio p0/p0*, by default None
    fanno_param : float, optional
        fano parameter 4fL*/D, by default None
    M : float, optional
        Mach number, by default None
    gam : float, optional
        ratio of specific heats, by default 1.4
    is_supersonic : bool, optional
        use supersonic solution (rather than subsonic), needed
        only if total pressure/entropy/fanno parameter is input, by default True

    Returns
    -------
    dict
        row from Fanno table

    Raises
    ------
    ValueError
        Inconsistent or incorrect inputs supplied
    """    
    Mref = 1.0
    if ac.is1known(M, [p_ratio, r_ratio, T_ratio, ds_R_ratio, p0_ratio]):
        return {
            "M": M,
            "p_ratio": pressure2(M2=M, M1=Mref, p1=1.0, gam=gam),
            "r_ratio": density2(M2=M, M1=Mref, rho1=1.0, gam=gam),
            "T_ratio": temperature2(M2=M, M1=Mref, T1=1.0, gam=gam),
            "ds_R_ratio": entropy2(M2=Mref, M1=M, R=1.0, s1=0.0, gam=gam),
            "p0_ratio": total_pressure2(M2=M, M1=Mref, p01=1.0, gam=gam),
            "fanno_param": fanno_parameter(M2=Mref, M1=M, gam=gam)
        }
    
    elif ac.is1known(p_ratio, [r_ratio, T_ratio, ds_R_ratio, p0_ratio, fanno_param, M]):
        m = mach2(M1=Mref, is_supersonic=is_supersonic, p21=p_ratio, gam=gam)

    elif ac.is1known(r_ratio, [p_ratio, T_ratio, ds_R_ratio, p0_ratio, fanno_param, M]):
        m = mach2(M1=Mref, is_supersonic=is_supersonic, r21=r_ratio, gam=gam)

    elif ac.is1known(T_ratio, [p_ratio, r_ratio, ds_R_ratio, p0_ratio, fanno_param, M]):
        m = mach2(M1=Mref, is_supersonic=is_supersonic, T21=T_ratio, gam=gam)

    elif ac.is1known(ds_R_ratio, [p_ratio, r_ratio, T_ratio, p0_ratio, fanno_param, M]):
        m = mach1(M2=Mref, is_supersonic=is_supersonic, ds21_R=ds_R_ratio, gam=gam)

    elif ac.is1known(p0_ratio, [p_ratio, r_ratio, T_ratio, ds_R_ratio, fanno_param, M]):
        m = mach2(M1=Mref, is_supersonic=is_supersonic, p02_p01=p0_ratio, gam=gam)

    elif ac.is1known(fanno_param, [p_ratio, r_ratio, T_ratio, ds_R_ratio, p0_ratio, M]):
        m = mach1(M2=Mref, is_supersonic=is_supersonic, fanno_param=fanno_param, gam=gam)

    else:
        raise ValueError("Specify exactly 1 of the following: M1, p_ratio, "
                         "T_ratio, r_ratio, ds_R_ratio, p0_ratio, or the Fanno parameter.")
    return lookup_table(M=m, gam=gam)

def mach2(M1:float=1.0, is_supersonic:bool=True, p21:float=None, r21:float=None,
          T21:float=None, ds21_R:float=None, p02_p01:float=None,
          fanno_param:float=None, gam:float=1.4):
    """
    Compute Mach number at station 2 for M1 and another input parameter.

    Parameters
    ----------
    M1 : float, optional
        Mach number at station 1 or M* if M1==1.0, by default 1.0
    is_supersonic : bool, optional
        if M1==1.0 and input param is total pressure/entropy/Fanno param,
        then the user must specify if M2 should be supersonic or not, by default True
    p21 : float, optional
        pressure ratio p2/p1 or p/p*, by default None
    r21 : float, optional
        density ratio rho2/rho1 or rho/rho*, by default None
    T21 : float, optional
        temperature ratio T2/T1 or T/T*, by default None
    ds21_R : float, optional
        entropy ratio (s2-s1)/R (cannot be s*-s/R), by default None
    p02_p01 : float, optional
        total pressure ratio p02/p01 or p0/p0*, by default None
    fanno_param : float, optional
        Fanno parameter 4f/D(x2-x1) (cannot be 4fL*/D), by default None
    gam : float, optional
        ratio of specific heats, by default 1.4

    Returns
    -------
    float
        Mach number at station 2, or M if M1==1.0

    Raises
    ------
    ValueError
        When M1=1.0 and the input parameter is Fanno parameter or entropy ratio.
    """    
    if np.abs(M1-1) < 1e-5:  
        M2_guess = 50 if is_supersonic else 0.01
    else:
        M2_guess = 0.5*(M1+1)

    if ac.is1known(p21, [T21, r21, ds21_R, p02_p01, fanno_param]):
        # expression obtained with wolfram alpha, real-only root kept
        a = 2/(gam-1)
        b = -1/(gam-1) * (M1/p21)**2 * (2 + (gam-1)*M1*M1)
        return (
            2**-0.5 * ( (a**2 - 4*b)**0.5 - a )**0.5
        )
    
    elif ac.is1known(T21, [p21, r21, ds21_R, p02_p01, fanno_param]):
        return ( ((2+(gam-1)*M1*M1) / T21 - 2) / (gam-1) )**0.5

    elif ac.is1known(r21, [p21, T21, ds21_R, p02_p01, fanno_param]):
        a = (r21/M1)**2 * (2 + (gam-1)*M1*M1)
        return (2 / (a - (gam-1)))**0.5

    elif ac.is1known(ds21_R, [p21, T21, r21, p02_p01, fanno_param]):
        func = lambda M2, M1, ds21_R, gam: (
            ds21_R - entropy2(M1=M1, M2=M2, R=1.0, s1=0.0, gam=gam)
        )
        if np.abs(M1 - 1) < 1e-5:
            raise ValueError("M2 is the sonic condition for entropy computations, use `mach1` instead.")
        return fsolve(func, M2_guess, args=(M1, ds21_R, gam))[0]
    
    elif ac.is1known(p02_p01, [p21, T21, r21, ds21_R, fanno_param]):
        func = lambda M2, M1, p02_p01, gam: (
            p02_p01 - total_pressure2(M2=M2, M1=M1, p01=1.0, gam=gam)
        )
        return fsolve(func, M2_guess, args=(M1, p02_p01, gam))[0]
    
    elif ac.is1known(fanno_param, [p21, T21, r21, ds21_R, p02_p01]):
        if np.abs(M1 - 1) < 1e-5:
            raise ValueError("M2 is the sonic condition for Fanno param computations, use `mach1` instead.")
        func = lambda M2, M1, fanno_param, gam: (
            fanno_param - fanno_parameter(M2=np.maximum(M2,0.01), M1=M1, gam=gam, throw_err=False)
        )
        return fsolve(func, M2_guess, args=(M1, fanno_param, gam))[0]
    
    else:
        raise ValueError("Specify 1 of the following: p21, "
                         "T21, r21, ds21_R, p02_p01, or the Fanno parameter.")
    
def mach1(M2=1.0, ds21_R:float=None, fanno_param:float=None, is_supersonic:bool=True,
          gam:float=1.4):
    """
    Get Mach number M or (M1 if M2 != 1.0) for a given entropy ratio or Fanno parameter.
    Use this function rather than `mach2` for these parameters because their reference
    condition Mach number is flipped (annoyingly M2 rather than M1).

    Parameters
    ----------
    M2 : float, optional
        _description_, by default 1.0
    ds21_R : float, optional
        _description_, by default None
    fanno_param : float, optional
        _description_, by default None
    is_supersonic : bool, optional
        _description_, by default True
    gam : float, optional
        _description_, by default 1.4

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    ValueError
        _description_
    """    
    M1_guess = 3 if is_supersonic else 0.2

    if ac.is1known(ds21_R, [fanno_param]):
        func = lambda M1, M2, ds21_R, gam: (
            ds21_R - entropy2(M1=M1, M2=M2, R=1.0, s1=0.0, gam=gam)
        )
        return fsolve(func, M1_guess, args=(M2, ds21_R, gam))[0]
    
    elif ac.is1known(fanno_param, [ds21_R]):
        func = lambda M1, M2, fanno_param, gam: (
            fanno_param - fanno_parameter(M2=M2, M1=M1, gam=gam, throw_err=False)
        )
        return fsolve(func, M1_guess, args=(M2, fanno_param, gam))[0]

    else:
        raise ValueError("Specify 1 of the following: ds21_R or the Fanno parameter.")

def temperature2(M2, M1=1.0, T1=1.0, gam:float=1.4):
    return T1 * (2+(gam-1)*M1*M1) / (2+(gam-1)*M2*M2)

def pressure2(M2, M1=1.0, p1=1.0, gam:float=1.4):
    return p1 * M1/M2 * temperature2(M1=M1, M2=M2, T1=1.0, gam=gam)**0.5

def density2(M2, M1=1.0, rho1=1.0, gam:float=1.4):
    return rho1 * M1/M2 * temperature2(M1=M1, M2=M2, T1=1.0, gam=gam)**-0.5

def total_pressure2(M2:float, M1=1.0, p01=1.0, gam:float=1.4):
    """
    Compute total pressure p02/p01 for Fanno flow.

    This form is valid for calorically perfect gases only, where `gamma`
    is constant (M1 < 5).

    Parameters
    ----------
    M2 : float
        Mach number at station 2
    M1 : float, optional
        Mach number at station 1, by default 1.0
    p01 : float, optional
        Stagnation pressure at station 1, by default 1.0
    gam : float, optional
        ratio of specific heats gamma, by default 1.4

    Returns
    -------
    float
        p02/p01 stagnation pressure ratio AND

        p0/p0* if `M1`==1.0 AND

        p02 if `p01`!=1.0 AND

        p0 if `M1`==1.0 and `p01`!=1.0
    """    
    return (
        p01 * M1/M2
      * temperature2(M1=M1,M2=M2, T1=1.0, gam=gam)**(-(gam+1)/(2*(gam-1)))
    )

def entropy2(M1:float, M2=1.0, R:float=1.0, s1:float=0.0, gam:float=1.4):
    """
    Compute entropy at station 2 s2 for Fanno flow.

    This form is valid for calorically perfect gases only, where `gamma`
    is constant (M1 < 5).

    Parameters
    ----------
    M1 : float
        Mach number at station 1 (upstream condition)
    M2 : float, optional
        Mach number at station 2, by default 1.0
    R : float, optional
        specific gas constant, by default 1.0
    s1 : float, optional
        Entropy at state 1, by default 0.0
    gam : float, optional
        ratio of specific heats, by default 1.4

    Returns
    -------
    float
        s2 AND

        (s2-s1) if `s1`==0 AND

        (s2-s1)/R if `s1`==0 and `R`==1.0 AND

        (s*-s)/R if `s1`==0 and `R`==1.0 and `M2`==1.0
    """    
    return (
        s1 + R * np.log(
            M2/M1 * ((1+(gam-1)/2*M1*M1)
                     / (1+(gam-1)/2*M2*M2))**(0.5*(gam+1)/(gam-1))
        )
    )

def fanno_parameter(M1, M2=1.0, gam:float=1.4, throw_err:bool=True):
    """
    Compute the Fanno parameter.

    Parameters
    ----------
    M1 : float
        Mach number at station 1, or simply M
    M2 : float, optional
        Mach number at station 2 or M* if M2==1.0, by default 1.0
    gam : float, optional
        ratio of specific heats, by default 1.4
    throw_err : bool, optional
        throw error if 2nd law violated (handy to set false when 
        root-finding), by default True

    Returns
    -------
    float
        4f/D(x2-x1) AND
        4fL*/D when M2==1.0 AND

    Raises
    ------
    ValueError
        Violation of 2nd law or revision of inlet conditions required.
    """    
    if throw_err:
        if M1>=1:
            if M2 > M1:
                raise ValueError("M2 cannot be greater than M1 when "
                                "M1 is supersonic. Violation of 2nd Law.")
            elif M2 < 1:
                raise ValueError("M2 cannot be subsonic. Flow is choked. "
                                "Normal shock to form before inlet.")
        elif M1<1:
            if M2 < M1:
                raise ValueError("M2 cannot be less than M1 when "
                                "M1 is subsonic. Violation of 2nd Law.")
            elif M2 > 1:
                raise ValueError("M2 cannot be supersonic. Flow is choked. "
                                "Revision of inlet conditions required.")
    integrand = lambda m,g: (
        -1/(g*m*m) - (g+1)/(2*g)*np.log(
            m*m/(1+(g-1)/2*m*m)
        )
    )
    return (
        integrand(M2, gam) - integrand(M1, gam)
    )

def L_star(fanno_param, D, f:float=0.005):
    """
    Compute duct length L for a given Fanno parameter 4fL*/D.

    Parameters
    ----------
    fanno_param : Any
        Fanno flow parameter 4fL*/D
    D : Any
        duct diameter
    f : float, optional
        average friction coefficient, by default 0.005 (which holds 
        for Re > 1e5 and surface roughness of 0.001D)

    Returns
    -------
    Any
        Duct length L AND L* for M=1
    """
    return 0.25*fanno_param*D/f