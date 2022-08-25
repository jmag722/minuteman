"""
Compressible, inviscid, 1D flow relations.

These functions compute flow parameters of normal shocks,
Rayleigh, and Fanno flow.
"""

def normal_shock_M2(M1,gam=1.4):
    """
    `normal_shock_M2` computes the Mach number behind a normal shock.

    This form is valid for calorically perfect gases only, where gamma
    is constant (M1 < 5).

    Parameters:
    M1(float): Mach number ahead of normal shock
    gam (float): gamma, ratio of specific heats

    Returns:
    M2: Mach number behind a normal shock
    """
    return ( (1 + (gam-1)*M1*M1/2) / (gam*M1*M1 - (gam-1)/2) )**0.5