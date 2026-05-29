"""
This module computes 1D, calorically perfect flow with heat addition (Rayleigh flow).
Online calculator for comparison at: http://www.dept.aoe.vt.edu/~devenpor/aoe3114/calc.html
Additional notes: https://kyleniemeyer.github.io/gas-dynamics-notes/compressible-flows/heat-transfer.html
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import fsolve

import minuteman.cpg.isentropic_flow as isentropic_flow
from minuteman.utils.types import (
    ArrayOrScalarFloat,
    check_equal_shape,
    Floatlike,
    FlowSpeedRegime,
    mach_guess_from_flow_regime,
    ndarray_FlowSpeedRegime,
    ndarray_f,
)


@dataclass
class RayleighFlowTable:
    """Rayleigh flow table where the initial Mach number M1=M*=1.0"""

    mach: ndarray_f
    """Mach number, M."""

    temperature_ratio: ndarray_f
    """Temperature ratio, T/T*"""

    pressure_ratio: ndarray_f
    """Static pressure ratio, p/p*"""

    density_ratio: ndarray_f
    """Density ratio, rho/rho*"""

    @property
    def velocity_ratio(self) -> ndarray_f:
        """Velocity ratio, u/u*"""
        return 1.0 / self.density_ratio

    total_pressure_ratio: ndarray_f
    """Total pressure ratio, p0/p0*"""

    total_temperature_ratio: ndarray_f
    """Total temperature ratio, T0/T0*"""

    entropy_ratio: ndarray_f
    """Entropy ratio, (s*-s)/R"""

    specific_heat_ratio: ndarray_f
    """Ratio of specific heats, gamma"""


def lookup_table_by_mach(
    mach: ArrayOrScalarFloat, specific_heat_ratio: ArrayOrScalarFloat = 1.4
) -> RayleighFlowTable:
    """Look up a Rayleigh flow table result from the Mach number, M

    Args:
        mach (ArrayOrScalarFloat): Mach number, M
        specific_heat_ratio (ArrayOrScalarFloat, optional): ratio of specific
            heats, gamma. Defaults to 1.4.

    Returns:
        RayleighFlowTable: Rayleigh flow output table
    """
    m1 = 1.0
    m2 = np.atleast_1d(mach)
    if isinstance(specific_heat_ratio, Floatlike):
        gam = np.full_like(m1, specific_heat_ratio)
    else:
        gam = np.asarray(specific_heat_ratio)
    return RayleighFlowTable(
        mach=np.atleast_1d(m2),
        temperature_ratio=temperature_ratio_by_mach(
            mach_initial=m1, mach_final=m2, specific_heat_ratio=gam
        ),
        pressure_ratio=pressure_ratio_by_mach(
            mach_initial=m1, mach_final=m2, specific_heat_ratio=gam
        ),
        density_ratio=density_ratio_by_mach(
            mach_initial=m1, mach_final=m2, specific_heat_ratio=gam
        ),
        total_pressure_ratio=total_pressure_ratio_by_mach(
            mach_initial=m1, mach_final=m2, specific_heat_ratio=gam
        ),
        total_temperature_ratio=total_temperature_ratio_by_mach(
            mach_initial=m1, mach_final=m2, specific_heat_ratio=gam
        ),
        entropy_ratio=_rev_entropy_ratio_by_mach(
            mach_initial=m1, mach_final=m2, specific_heat_ratio=gam
        ),
        specific_heat_ratio=gam,
    )


def lookup_table_by_pressure(
    pressure_ratio: ArrayOrScalarFloat,
    specific_heat_ratio: ArrayOrScalarFloat = 1.4,
) -> RayleighFlowTable:
    """Look up a Rayleigh flow table result from the pressure_ratio, p/p*

    Args:
        pressure_ratio (ArrayOrScalarFloat): static pressure ratio, p/p*
        specific_heat_ratio (ArrayOrScalarFloat, optional): ratio of specific
            heats, gamma. Defaults to 1.4.

    Returns:
        RayleighFlowTable: Rayleigh flow output table
    """
    p_ratio = np.atleast_1d(pressure_ratio)
    gam = specific_heat_ratio
    m1 = 1.0
    # invert relationship between p2/p1 and M1, M2
    m2 = (((1 + gam * m1**2) / p_ratio - 1) / gam) ** 0.5
    return lookup_table_by_mach(mach=m2, specific_heat_ratio=gam)


def _lookup_table_by_ratio(
    ratio: ArrayOrScalarFloat,
    specific_heat_ratio: ArrayOrScalarFloat,
    flow_regime: ndarray_FlowSpeedRegime | FlowSpeedRegime,
    mach_func: Callable,
) -> RayleighFlowTable:
    """Lookup the Rayleigh flow table by a generic input ratio where the
    relationship with Mach must be solved numerically

    Args:
        ratio (ArrayOrScalarFloat): ratio of interest
            (total temp., total pressure, etc)
        specific_heat_ratio (ArrayOrScalarFloat): ratio of specific heats, gamma
        flow_regime (ndarray_FlowSpeedRegime | FlowSpeedRegime): flow regime
            (supersonic, subsonic)
        mach_func (Callable): function to compute Mach from the given ratio

    Returns:
        RayleighFlowTable: Rayleigh flow table
    """
    _ratio = np.atleast_1d(ratio)
    if isinstance(specific_heat_ratio, Floatlike):
        gam = np.full_like(_ratio, specific_heat_ratio)
    else:
        gam = np.asarray(specific_heat_ratio)
    check_equal_shape(gam.shape, _ratio.shape)

    mach_guesses = mach_guess_from_flow_regime(
        flow_regime, _ratio.shape, mach_subsonic=0.5, mach_supersonic=3.0
    )

    def get_mach_by_ratio(_m, _r, _g):
        return _r - mach_func(
            mach_initial=1.0, mach_final=_m, specific_heat_ratio=_g
        )

    m2 = np.empty_like(_ratio)
    for i in range(_ratio.size):
        m2.flat[i] = fsolve(
            get_mach_by_ratio,
            mach_guesses.flat[i],
            args=(_ratio.flat[i], gam.flat[i]),
        )[0]
    return lookup_table_by_mach(mach=m2, specific_heat_ratio=gam)


def lookup_table_by_temperature(
    temperature_ratio: ArrayOrScalarFloat,
    specific_heat_ratio: ArrayOrScalarFloat = 1.4,
    flow_regime: ndarray_FlowSpeedRegime
    | FlowSpeedRegime = FlowSpeedRegime.supersonic,
) -> RayleighFlowTable:
    """Look up a Rayleigh flow table result from the temperature_ratio, T/T*

    Args:
        temperature_ratio (ArrayOrScalarFloat): static temperature ratio, T/T*
        specific_heat_ratio (ArrayOrScalarFloat, optional): ratio of specific
            heats. Defaults to 1.4.
        flow_regime (ndarray_FlowSpeedRegime | FlowSpeedRegime, optional):
            flow speed regime (either supersonic or subsonic).
            Defaults to FlowSpeedRegime.supersonic.

    Returns:
        RayleighFlowTable: Rayleigh flow output table
    """
    return _lookup_table_by_ratio(
        ratio=temperature_ratio,
        specific_heat_ratio=specific_heat_ratio,
        flow_regime=flow_regime,
        mach_func=temperature_ratio_by_mach,
    )


def lookup_table_by_density(
    density_ratio: ArrayOrScalarFloat,
    specific_heat_ratio: ArrayOrScalarFloat = 1.4,
) -> RayleighFlowTable:
    """Look up a Rayleigh flow table result from the density ratio, rho/rho*

    Args:
        density_ratio (ArrayOrScalarFloat): static pressure ratio, p/p*
        specific_heat_ratio (ArrayOrScalarFloat, optional): ratio of specific
            heats, gamma. Defaults to 1.4.

    Returns:
        RayleighFlowTable: Rayleigh flow output table
    """
    m1 = 1.0
    r_ratio = np.atleast_1d(density_ratio)
    gam = specific_heat_ratio
    m2 = ((r_ratio * (1 + gam * m1**2)) / m1**2 - gam) ** -0.5
    return lookup_table_by_mach(mach=m2, specific_heat_ratio=gam)


def lookup_table_by_total_pressure(
    total_pressure_ratio: ArrayOrScalarFloat,
    specific_heat_ratio: ArrayOrScalarFloat = 1.4,
    flow_regime: ndarray_FlowSpeedRegime
    | FlowSpeedRegime = FlowSpeedRegime.supersonic,
) -> RayleighFlowTable:
    """Look up a Rayleigh flow table result from the total pressure ratio,
    p0/p0*

    Args:
        total_pressure_ratio (ArrayOrScalarFloat): total pressure ratio, p0/p0*
        specific_heat_ratio (ArrayOrScalarFloat, optional): ratio of specific
            heats. Defaults to 1.4.
        flow_regime (ndarray_FlowSpeedRegime | FlowSpeedRegime, optional):
            flow speed regime (either supersonic or subsonic).
            Defaults to FlowSpeedRegime.supersonic.

    Returns:
        RayleighFlowTable: Rayleigh flow output table
    """
    return _lookup_table_by_ratio(
        ratio=total_pressure_ratio,
        specific_heat_ratio=specific_heat_ratio,
        flow_regime=flow_regime,
        mach_func=total_pressure_ratio_by_mach,
    )


def lookup_table_by_total_temperature(
    total_temperature_ratio: ArrayOrScalarFloat,
    specific_heat_ratio: ArrayOrScalarFloat = 1.4,
    flow_regime: ndarray_FlowSpeedRegime
    | FlowSpeedRegime = FlowSpeedRegime.supersonic,
) -> RayleighFlowTable:
    """Look up a Rayleigh flow table result from the total temperature ratio,
    T0/T0*

    Args:
        total_temperature_ratio (ArrayOrScalarFloat): total temperature ratio,
            T0/T0*
        specific_heat_ratio (ArrayOrScalarFloat, optional): ratio of specific
            heats. Defaults to 1.4.
        flow_regime (ndarray_FlowSpeedRegime | FlowSpeedRegime, optional):
            flow speed regime (either supersonic or subsonic).
            Defaults to FlowSpeedRegime.supersonic.

    Returns:
        RayleighFlowTable: Rayleigh flow output table
    """
    return _lookup_table_by_ratio(
        ratio=total_temperature_ratio,
        specific_heat_ratio=specific_heat_ratio,
        flow_regime=flow_regime,
        mach_func=total_temperature_ratio_by_mach,
    )


def lookup_table_by_entropy(
    entropy_ratio: ArrayOrScalarFloat,
    specific_heat_ratio: ArrayOrScalarFloat = 1.4,
    flow_regime: ndarray_FlowSpeedRegime
    | FlowSpeedRegime = FlowSpeedRegime.supersonic,
) -> RayleighFlowTable:
    """Look up a Rayleigh flow table result from the entropy ratio, (s*-s)/R

    Args:
        entropy_ratio (ArrayOrScalarFloat): entropy ratio, (s*-s)/R
        specific_heat_ratio (ArrayOrScalarFloat, optional): ratio of specific
            heats. Defaults to 1.4.
        flow_regime (ndarray_FlowSpeedRegime | FlowSpeedRegime, optional):
            flow speed regime (either supersonic or subsonic).
            Defaults to FlowSpeedRegime.supersonic.

    Returns:
        RayleighFlowTable: Rayleigh flow output table
    """
    return _lookup_table_by_ratio(
        ratio=entropy_ratio,
        specific_heat_ratio=specific_heat_ratio,
        flow_regime=flow_regime,
        mach_func=_rev_entropy_ratio_by_mach,
    )


def _rev_entropy_ratio_by_mach(
    mach_initial: ArrayOrScalarFloat,
    mach_final: ArrayOrScalarFloat,
    specific_heat_ratio: ArrayOrScalarFloat,
) -> ndarray_f:
    """This function computes (s1-s2)/R. This is handy because s1 becomes
    the critical/sonic point, s*, and this allows you to get the typical
    positive value back out
    """
    m1 = np.atleast_1d(mach_initial)
    m2 = mach_final
    gam = specific_heat_ratio
    return -entropy_ratio_by_mach(
        mach_final=m2, mach_initial=m1, specific_heat_ratio=gam
    )


def entropy_ratio_by_mach(
    mach_initial: ArrayOrScalarFloat,
    mach_final: ArrayOrScalarFloat,
    specific_heat_ratio: ArrayOrScalarFloat,
) -> ndarray_f:
    """Compute the entropy ratio from the Mach number for Rayleigh flow.

    Args:
        mach_initial (ArrayOrScalarFloat): Initial Mach number, M1. This is
            the reference Mach number, M*, when equal to unity.
        mach_final (ArrayOrScalarFloat): Final Mach number, M2. This is simply
            the Mach number, M, when mach_initial==1.0
        specific_heat_ratio (ArrayOrScalarFloat): ratio of specific heats, gamma

    Returns:
        ndarray_f: entropy ratio, (s2-s1)/R (or (s-s*)/R if mach_initial==1.0)
    """
    m1 = np.atleast_1d(mach_initial)
    m2 = mach_final
    gam = specific_heat_ratio

    return np.log(
        ((1 + gam * m1**2) / (1 + gam * m2**2)) ** (gam + 1)
        * (m2 / m1) ** (2 * gam)
    ) / (gam - 1)


def total_pressure_ratio_by_mach(
    mach_initial: ArrayOrScalarFloat,
    mach_final: ArrayOrScalarFloat,
    specific_heat_ratio: ArrayOrScalarFloat,
) -> ndarray_f:
    """Compute the total pressure ratio from the Mach number
    for Rayleigh flow.

    Args:
        mach_initial (ArrayOrScalarFloat): Initial Mach number, M1. This is
            the reference Mach number, M*, when equal to unity.
        mach_final (ArrayOrScalarFloat): Final Mach number, M2. This is simply
            the Mach number, M, when mach_initial==1.0
        specific_heat_ratio (ArrayOrScalarFloat): ratio of specific heats, gamma

    Returns:
        ndarray_f: total pressure ratio, p02/p01 (or p0/p0* if M2==1.0)
    """
    m1 = np.atleast_1d(mach_initial)
    m2 = mach_final
    gam = specific_heat_ratio
    return (
        pressure_ratio_by_mach(
            mach_initial=m1, mach_final=m2, specific_heat_ratio=gam
        )
        * isentropic_flow.total_pressure_ratio(
            mach=m2, specific_heat_ratio=gam
        )
        / isentropic_flow.total_pressure_ratio(
            mach=m1, specific_heat_ratio=gam
        )
    )


def total_temperature_ratio_by_mach(
    mach_initial: ArrayOrScalarFloat,
    mach_final: ArrayOrScalarFloat,
    specific_heat_ratio: ArrayOrScalarFloat,
) -> ndarray_f:
    """Compute the total temperature ratio from the Mach number
    for Rayleigh flow.

    Args:
        mach_initial (ArrayOrScalarFloat): Initial Mach number, M1. This is
            the reference Mach number, M*, when equal to unity.
        mach_final (ArrayOrScalarFloat): Final Mach number, M2. This is simply
            the Mach number, M, when mach_initial==1.0
        specific_heat_ratio (ArrayOrScalarFloat): ratio of specific heats, gamma

    Returns:
        ndarray_f: total temperature ratio, T02/T01 (or T0/T0* if M2==1.0)
    """
    m1 = np.atleast_1d(mach_initial)
    m2 = mach_final
    gam = specific_heat_ratio
    return (
        temperature_ratio_by_mach(
            mach_initial=m1, mach_final=m2, specific_heat_ratio=gam
        )
        * isentropic_flow.total_temperature_ratio(
            mach=m2, specific_heat_ratio=gam
        )
        / isentropic_flow.total_temperature_ratio(
            mach=m1, specific_heat_ratio=gam
        )
    )


def pressure_ratio_by_mach(
    mach_initial: ArrayOrScalarFloat,
    mach_final: ArrayOrScalarFloat,
    specific_heat_ratio: ArrayOrScalarFloat,
) -> ndarray_f:
    """Compute the pressure ratio from the Mach number for Rayleigh flow.

    Args:
        mach_initial (ArrayOrScalarFloat): Initial Mach number, M1. This is
            the reference Mach number, M*, when equal to unity.
        mach_final (ArrayOrScalarFloat): Final Mach number, M2. This is simply
            the Mach number, M, when mach_initial==1.0
        specific_heat_ratio (ArrayOrScalarFloat): ratio of specific heats, gamma

    Returns:
        ndarray_f: pressure ratio, p2/p1 (or p/p* if M2==1.0)
    """
    m1 = np.atleast_1d(mach_initial)
    m2 = mach_final
    gam = specific_heat_ratio
    return (1 + gam * m1**2) / (1 + gam * m2**2)


def temperature_ratio_by_mach(
    mach_initial: ArrayOrScalarFloat,
    mach_final: ArrayOrScalarFloat,
    specific_heat_ratio: ArrayOrScalarFloat,
) -> ndarray_f:
    """Compute the temperature ratio from the Mach number for Rayleigh flow.

    Args:
        mach_initial (ArrayOrScalarFloat): Initial Mach number, M1. This is
            the reference Mach number, M*, when equal to unity.
        mach_final (ArrayOrScalarFloat): Final Mach number, M2. This is simply
            the Mach number, M, when mach_initial==1.0
        specific_heat_ratio (ArrayOrScalarFloat): ratio of specific heats, gamma

    Returns:
        ndarray_f: temperature ratio, T2/T1 (or T/T* if M2==1.0)
    """
    m1 = np.atleast_1d(mach_initial)
    m2 = mach_final
    gam = specific_heat_ratio
    return ((1 + gam * m1**2) / (1 + gam * m2**2)) ** 2 * (m2 / m1) ** 2


def density_ratio_by_mach(
    mach_initial: ArrayOrScalarFloat,
    mach_final: ArrayOrScalarFloat,
    specific_heat_ratio: ArrayOrScalarFloat,
) -> ndarray_f:
    """Compute the density ratio from the Mach number for Rayleigh flow.

    Args:
        mach_initial (ArrayOrScalarFloat): Initial Mach number, M1. This is
            the reference Mach number, M*, when equal to unity.
        mach_final (ArrayOrScalarFloat): Final Mach number, M2. This is simply
            the Mach number, M, when mach_initial==1.0
        specific_heat_ratio (ArrayOrScalarFloat): ratio of specific heats, gamma

    Returns:
        ndarray_f: density ratio, rho2/rho1 (or rho/rho* if M2==1.0)
    """
    m1 = np.atleast_1d(mach_initial)
    m2 = mach_final
    gam = specific_heat_ratio
    return (1 + gam * m2**2) / (1 + gam * m1**2) * (m1 / m2) ** 2
