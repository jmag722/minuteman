r"""
Compressible, inviscid flow relations.

These functions are basic definitions pertaining to 1-,2-, and 3-D
compressible, inviscid flow.

Total or stagnation quantities (total temperature, for instance) are the
values that would exist if the flow were isentropically brought to rest.
Total quantities remain constant where flowfield is isentropic
(adiabatic and reversible).
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import fsolve

from minuteman.cpg import FlowSpeedRegime, ndarray_FlowSpeedRegime
from minuteman.cpg.base import mach_guess_from_flow_regime
import minuteman.cpg.thermo as thermo
from minuteman.utils.types import (
    ArrayOrScalarFloat,
    check_equal_shape,
    Floatlike,
    ndarray_f,
)


@dataclass
class IsentropicFlowTable:
    """Isentropic flow table, containing various isentropic parameters
    for a given Mach number and specific heat ratio
    """

    mach: ndarray_f
    r"""Mach number, $M$"""
    temperature: ndarray_f
    r"""total temperature ratio, $T_0 / T$"""
    pressure: ndarray_f
    r"""total pressure ratio, $p_0 / p$"""
    density: ndarray_f
    r"""total density ratio, $\rho_0 / \rho$"""
    speed_of_sound: ndarray_f
    r"""total speed of sound ratio, $a_0 / a$"""
    area_ratio: ndarray_f
    r"""area ratio, $A / A^*$"""
    specific_heat_ratio: ndarray_f
    r"""ratio of specific heats, $\gamma$"""


def lookup_table_by_mach(
    mach: ArrayOrScalarFloat, specific_heat_ratio: ArrayOrScalarFloat = 1.4
) -> IsentropicFlowTable:
    r"""Lookup the isentropic flow table based on Mach number, $M$

    Args:
        mach (ArrayOrScalarFloat): Mach number, $M$
        specific_heat_ratio (ArrayOrScalarFloat): ratio of
            specific heats, $\gamma$. Defaults to 1.4.

    Returns:
        IsentropicFlowTable: isentropic flow table result
    """
    m = np.atleast_1d(mach)
    gam = np.atleast_1d(specific_heat_ratio)
    check_equal_shape(m.shape, gam.shape)
    p0_ratio = total_pressure_ratio(mach=m, specific_heat_ratio=gam)
    r0_ratio = total_density_ratio(mach=m, specific_heat_ratio=gam)
    t0_ratio = total_temperature_ratio(mach=m, specific_heat_ratio=gam)
    a0_ratio = total_speed_of_sound_ratio(mach=m, specific_heat_ratio=gam)
    area_ratio = area_mach_relation(mach=m, specific_heat_ratio=gam)
    return IsentropicFlowTable(
        mach=m,
        temperature=t0_ratio,
        pressure=p0_ratio,
        density=r0_ratio,
        speed_of_sound=a0_ratio,
        area_ratio=area_ratio,
        specific_heat_ratio=gam,
    )


def lookup_table_by_temperature(
    temperature_ratio: ArrayOrScalarFloat,
    specific_heat_ratio: ArrayOrScalarFloat = 1.4,
) -> IsentropicFlowTable:
    r"""Lookup the isentropic flow table based on total temperature ratio,
    $T_0 / T$

    Args:
        temperature_ratio (ArrayOrScalarFloat): total temperature ratio,
            $T_0 / T$
        specific_heat_ratio (ArrayOrScalarFloat): ratio of
            specific heats, $\gamma$. Defaults to 1.4.

    Returns:
        IsentropicFlowTable: isentropic flow table result
    """
    gam = np.atleast_1d(specific_heat_ratio)
    mach = mach_from_temperature(
        temperature_ratio=temperature_ratio, specific_heat_ratio=gam
    )
    return lookup_table_by_mach(mach=mach, specific_heat_ratio=gam)


def lookup_table_by_pressure(
    pressure_ratio: ArrayOrScalarFloat,
    specific_heat_ratio: ArrayOrScalarFloat = 1.4,
) -> IsentropicFlowTable:
    r"""Lookup the isentropic flow table based on total pressure ratio,
    $p_0 / p$

    Args:
        pressure_ratio (ArrayOrScalarFloat): total pressure ratio, $p_0 / p$
        specific_heat_ratio (ArrayOrScalarFloat): ratio of
            specific heats, $\gamma$. Defaults to 1.4.

    Returns:
        IsentropicFlowTable: isentropic flow table result
    """
    gam = np.atleast_1d(specific_heat_ratio)
    t0_ratio = thermo.isentropic_process_from_pressure(
        pressure_ratio=pressure_ratio, specific_heat_ratio=gam
    ).temperature_ratio
    return lookup_table_by_temperature(
        temperature_ratio=t0_ratio, specific_heat_ratio=specific_heat_ratio
    )


def lookup_table_by_density(
    density_ratio: ArrayOrScalarFloat,
    specific_heat_ratio: ArrayOrScalarFloat = 1.4,
) -> IsentropicFlowTable:
    r"""Lookup the isentropic flow table based on total density ratio,
    $\rho_0 / \rho$

    Args:
        density_ratio (ArrayOrScalarFloat): total density ratio, $\rho_0 / \rho$
        specific_heat_ratio (ArrayOrScalarFloat): ratio of
            specific heats, $\gamma$. Defaults to 1.4.

    Returns:
        IsentropicFlowTable: isentropic flow table result
    """
    gam = np.atleast_1d(specific_heat_ratio)
    t0_ratio = thermo.isentropic_process_from_density(
        density_ratio=density_ratio, specific_heat_ratio=gam
    ).temperature_ratio
    return lookup_table_by_temperature(
        temperature_ratio=t0_ratio, specific_heat_ratio=specific_heat_ratio
    )


def lookup_table_by_speed_of_sound(
    speed_of_sound_ratio: ArrayOrScalarFloat,
    specific_heat_ratio: ArrayOrScalarFloat = 1.4,
) -> IsentropicFlowTable:
    r"""Lookup the isentropic flow table based on total speed of sound ratio,
    $a_0 / a$

    Args:
        speed_of_sound_ratio (ArrayOrScalarFloat): total speed of sound ratio,
            $a_0 / a$
        specific_heat_ratio (ArrayOrScalarFloat): ratio of
            specific heats, $\gamma$. Defaults to 1.4.

    Returns:
        IsentropicFlowTable: isentropic flow table result
    """
    gam = np.atleast_1d(specific_heat_ratio)
    t0_ratio = thermo.isentropic_process_from_speed_of_sound(
        speed_of_sound_ratio=speed_of_sound_ratio, specific_heat_ratio=gam
    ).temperature_ratio
    return lookup_table_by_temperature(
        temperature_ratio=t0_ratio, specific_heat_ratio=specific_heat_ratio
    )


def lookup_table_by_area_ratio(
    area_ratio: ArrayOrScalarFloat,
    specific_heat_ratio: ArrayOrScalarFloat = 1.4,
    flow_regime: ndarray_FlowSpeedRegime
    | FlowSpeedRegime = FlowSpeedRegime.supersonic,
) -> IsentropicFlowTable:
    r"""Lookup the isentropic flow table based on area ratio, $A / A^*$

    Args:
        area_ratio (ArrayOrScalarFloat): area ratio, $A / A^*$
        specific_heat_ratio (ArrayOrScalarFloat): ratio of
            specific heats, $\gamma$. Defaults to 1.4.
        flow_regime (ndarray_FlowSpeedRegime | FlowSpeedRegime): Is flowfield
            subsonic or supersonic. Default is ``FlowSpeedRegime.supersonic``

    Returns:
        IsentropicFlowTable: isentropic flow table result
    """
    aratios = np.atleast_1d(area_ratio)
    if isinstance(specific_heat_ratio, Floatlike):
        gam = np.full_like(aratios, specific_heat_ratio)
    else:
        gam = np.asarray(specific_heat_ratio)
    check_equal_shape(aratios.shape, gam.shape)

    mguess = mach_guess_from_flow_regime(
        flow_regime, aratios.shape, mach_subsonic=0.2, mach_supersonic=2.0
    )
    mach = np.empty_like(aratios)
    for i in range(aratios.size):
        mach.flat[i] = mach_from_area_ratio(
            area_ratio=aratios.flat[i],
            specific_heat_ratio=gam.flat[i],
            mach_guess=mguess.flat[i],
        )
    return lookup_table_by_mach(mach=mach, specific_heat_ratio=gam)


def total_temperature_ratio(
    mach: ArrayOrScalarFloat, specific_heat_ratio: ArrayOrScalarFloat
) -> ndarray_f:
    r"""Computes the stagnation or total temperature ratio, $T_0 / T$

    Args:
        mach (ArrayOrScalarFloat): Mach number, $M$
        specific_heat_ratio (ArrayOrScalarFloat): ratio of specific heats,
            $\gamma$

    Returns:
        ndarray_f: total temperature ratio, $T_0 / T$
    """
    gam = np.atleast_1d(specific_heat_ratio)
    return 1 + 0.5 * (gam - 1) * mach**2


def mach_from_temperature(
    temperature_ratio: ArrayOrScalarFloat,
    specific_heat_ratio: ArrayOrScalarFloat,
) -> ndarray_f:
    r"""Computes the Mach number $M$ from total temperature ratio $T_0 / T$

    Args:
        temperature_ratio (ArrayOrScalarFloat): total temperature ratio,
            $T_0 / T$
        specific_heat_ratio (ArrayOrScalarFloat): ratio of specific heats,
            $\gamma$

    Returns:
        ArrayOrScalarFloat: Mach number, $M$
    """
    gam = np.atleast_1d(specific_heat_ratio)
    return (2.0 / (gam - 1) * (temperature_ratio - 1)) ** 0.5


def total_pressure_ratio(
    mach: ArrayOrScalarFloat, specific_heat_ratio: ArrayOrScalarFloat
) -> ndarray_f:
    r"""Computes the stagnation or total pressure ratio, $p_0 / p$

    Args:
        mach (ArrayOrScalarFloat): Mach number, $M$
        specific_heat_ratio (ArrayOrScalarFloat): ratio of specific heats,
            $\gamma$

    Returns:
        ndarray_f: total pressure ratio, $p_0 / p$
    """
    gam = np.atleast_1d(specific_heat_ratio)
    return total_temperature_ratio(mach=mach, specific_heat_ratio=gam) ** (
        gam / (gam - 1)
    )


def total_density_ratio(
    mach: ArrayOrScalarFloat, specific_heat_ratio: ArrayOrScalarFloat
) -> ndarray_f:
    r"""Computes the stagnation or total density ratio, $\rho_0 / \rho$

    Args:
        mach (ArrayOrScalarFloat): Mach number, $M$
        specific_heat_ratio (ArrayOrScalarFloat): ratio of specific heats,
            $\gamma$

    Returns:
        ndarray_f: total density ratio, $\rho_0 / \rho$
    """
    gam = np.atleast_1d(specific_heat_ratio)
    return total_temperature_ratio(mach=mach, specific_heat_ratio=gam) ** (
        1.0 / (gam - 1)
    )


def total_speed_of_sound_ratio(
    mach: ArrayOrScalarFloat, specific_heat_ratio: ArrayOrScalarFloat
) -> ndarray_f:
    r"""Computes the stagnation or total speed of sound ratio, $a_0 / a$

    Args:
        mach (ArrayOrScalarFloat): Mach number, $M$
        specific_heat_ratio (ArrayOrScalarFloat): ratio of specific heats,
            $\gamma$

    Returns:
        ndarray_f: total speed of sound ratio, $a_0 / a$
    """
    gam = np.atleast_1d(specific_heat_ratio)
    return (total_temperature_ratio(mach=mach, specific_heat_ratio=gam)) ** 0.5


def _area_mach_relation_sqr(m, gam):
    return m**-2 * (
        2.0
        / (gam + 1)
        * total_temperature_ratio(mach=m, specific_heat_ratio=gam)
    ) ** ((gam + 1) / (gam - 1))


def area_mach_relation(
    mach: ArrayOrScalarFloat, specific_heat_ratio: ArrayOrScalarFloat
) -> ndarray_f:
    r"""Computes the area ratio $A / A^*$ for an isentropic nozzle.

    This is the standard area-Mach number relation

    Args:
        mach (ArrayOrScalarFloat): Mach number, $M$
        specific_heat_ratio (ArrayOrScalarFloat): ratio of specific heats,
            $\gamma$

    Returns:
        ndarray_f: area ratio, $A / A^*$
    """
    return _area_mach_relation_sqr(m=mach, gam=specific_heat_ratio) ** 0.5


def mach_from_area_ratio(
    area_ratio: Floatlike,
    specific_heat_ratio: Floatlike,
    mach_guess: Floatlike,
) -> Floatlike:
    r"""Compute the Mach number for a known area ratio, $A / A^*$

    Args:
        area_ratio (Floatlike): area ratio, $A / A^*$
        specific_heat_ratio (Floatlike): ratio of specific heats, $\gamma$
        mach_guess (Floatlike): Guess for the Mach number.
            A value above unity is suitable for supersonic Mach (2.0 is good).
            If a subsonic Mach solution is desired, set to a value below
            unity (0.2 is a good default)

    Returns:
        Floatlike: supersonic or subsonic Mach solution for a given area ratio
    """

    def compute_mach(m, aratio, gam):
        return aratio**2 - _area_mach_relation_sqr(m=m, gam=gam)

    return fsolve(compute_mach, mach_guess, (area_ratio, specific_heat_ratio))[
        0
    ]


def mach_angle(mach: ArrayOrScalarFloat) -> ndarray_f:
    r"""Compute the Mach angle, $\mu$ [radians].

    Args:
        mach (ArrayOrScalarFloat): Mach number, $M$

    Returns:
        ndarray_f: Mach angle, $\mu$ [radians]
    """
    return np.asin(1.0 / np.atleast_1d(mach))
