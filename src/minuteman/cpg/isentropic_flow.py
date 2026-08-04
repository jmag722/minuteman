# Copyright (c) 2022-2026 Jared Magnusson
# SPDX-License-Identifier: Apache-2.0

r"""```python
 import minuteman.cpg.isentropic_flow as isentropic_flow
```

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
from scipy.optimize.elementwise import find_root

from minuteman.cpg import ArraylikeFlowSpeedRegime, FlowSpeedRegime, thermo
from minuteman.cpg.base import bracket_mach_from_flow_regime
from minuteman.utils.types import (
    ArraylikeFloat,
    NDArrayFloat,
    RootFindingError,
)


@dataclass
class IsentropicFlowTable:
    """Isentropic flow table, containing various isentropic parameters
    for a given Mach number and specific heat ratio
    """

    mach: NDArrayFloat
    r"""Mach number, $M$"""
    temperature: NDArrayFloat
    r"""total temperature ratio, $T_0 / T$"""
    pressure: NDArrayFloat
    r"""total pressure ratio, $p_0 / p$"""
    density: NDArrayFloat
    r"""total density ratio, $\rho_0 / \rho$"""
    speed_of_sound: NDArrayFloat
    r"""total speed of sound ratio, $a_0 / a$"""
    area_ratio: NDArrayFloat
    r"""area ratio, $A / A^*$"""
    specific_heat_ratio: NDArrayFloat
    r"""ratio of specific heats, $\gamma$"""


def lookup_table_by_mach(
    mach: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat = 1.4,
) -> IsentropicFlowTable:
    r"""Lookup the isentropic flow table based on Mach number, $M$

    Args:
        mach (ArraylikeFloat): Mach number, $M$
        specific_heat_ratio (ArraylikeFloat): ratio of
            specific heats, $\gamma$. Defaults to 1.4.

    Returns:
        IsentropicFlowTable: isentropic flow table result

    """
    m = np.atleast_1d(mach)
    gam = np.atleast_1d(specific_heat_ratio)
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
    temperature_ratio: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat = 1.4,
) -> IsentropicFlowTable:
    r"""Lookup the isentropic flow table based on total temperature ratio,
    $T_0 / T$

    Args:
        temperature_ratio (ArraylikeFloat): total temperature ratio,
            $T_0 / T$
        specific_heat_ratio (ArraylikeFloat): ratio of
            specific heats, $\gamma$. Defaults to 1.4.

    Returns:
        IsentropicFlowTable: isentropic flow table result

    """
    gam = np.atleast_1d(specific_heat_ratio)
    mach = mach_from_temperature(
        temperature_ratio=temperature_ratio,
        specific_heat_ratio=gam,
    )
    return lookup_table_by_mach(mach=mach, specific_heat_ratio=gam)


def lookup_table_by_pressure(
    pressure_ratio: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat = 1.4,
) -> IsentropicFlowTable:
    r"""Lookup the isentropic flow table based on total pressure ratio,
    $p_0 / p$

    Args:
        pressure_ratio (ArraylikeFloat): total pressure ratio, $p_0 / p$
        specific_heat_ratio (ArraylikeFloat): ratio of
            specific heats, $\gamma$. Defaults to 1.4.

    Returns:
        IsentropicFlowTable: isentropic flow table result

    """
    gam = np.atleast_1d(specific_heat_ratio)
    t0_ratio = thermo.isentropic_process_from_pressure(
        pressure_ratio=pressure_ratio,
        specific_heat_ratio=gam,
    ).temperature_ratio
    return lookup_table_by_temperature(
        temperature_ratio=t0_ratio,
        specific_heat_ratio=specific_heat_ratio,
    )


def lookup_table_by_density(
    density_ratio: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat = 1.4,
) -> IsentropicFlowTable:
    r"""Lookup the isentropic flow table based on total density ratio,
    $\rho_0 / \rho$

    Args:
        density_ratio (ArraylikeFloat): total density ratio,
            $\rho_0 / \rho$
        specific_heat_ratio (ArraylikeFloat): ratio of
            specific heats, $\gamma$. Defaults to 1.4.

    Returns:
        IsentropicFlowTable: isentropic flow table result

    """
    gam = np.atleast_1d(specific_heat_ratio)
    t0_ratio = thermo.isentropic_process_from_density(
        density_ratio=density_ratio,
        specific_heat_ratio=gam,
    ).temperature_ratio
    return lookup_table_by_temperature(
        temperature_ratio=t0_ratio,
        specific_heat_ratio=specific_heat_ratio,
    )


def lookup_table_by_speed_of_sound(
    speed_of_sound_ratio: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat = 1.4,
) -> IsentropicFlowTable:
    r"""Lookup the isentropic flow table based on total speed of sound ratio,
    $a_0 / a$

    Args:
        speed_of_sound_ratio (ArraylikeFloat): total speed of sound ratio,
            $a_0 / a$
        specific_heat_ratio (ArraylikeFloat): ratio of
            specific heats, $\gamma$. Defaults to 1.4.

    Returns:
        IsentropicFlowTable: isentropic flow table result

    """
    gam = np.atleast_1d(specific_heat_ratio)
    t0_ratio = thermo.isentropic_process_from_speed_of_sound(
        speed_of_sound_ratio=speed_of_sound_ratio,
        specific_heat_ratio=gam,
    ).temperature_ratio
    return lookup_table_by_temperature(
        temperature_ratio=t0_ratio,
        specific_heat_ratio=specific_heat_ratio,
    )


def lookup_table_by_area_ratio(
    area_ratio: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat = 1.4,
    flow_regime: ArraylikeFlowSpeedRegime = FlowSpeedRegime.supersonic,
) -> IsentropicFlowTable:
    r"""Lookup the isentropic flow table based on area ratio, $A / A^*$

    Args:
        area_ratio (ArraylikeFloat): area ratio, $A / A^*$
        specific_heat_ratio (ArraylikeFloat): ratio of
            specific heats, $\gamma$. Defaults to 1.4.
        flow_regime (ArraylikeFlowSpeedRegime): Is flowfield
            subsonic or supersonic. Default is ``FlowSpeedRegime.supersonic``

    Returns:
        IsentropicFlowTable: isentropic flow table result

    """
    gam = np.atleast_1d(specific_heat_ratio)
    mach = mach_from_area_ratio(
        area_ratio=area_ratio,
        specific_heat_ratio=gam,
        flow_regime=flow_regime,
    )
    return lookup_table_by_mach(mach=mach, specific_heat_ratio=gam)


def total_temperature_ratio(
    mach: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat,
) -> NDArrayFloat:
    r"""Computes the stagnation or total temperature ratio, $T_0 / T$

    Args:
        mach (ArraylikeFloat): Mach number, $M$
        specific_heat_ratio (ArraylikeFloat): ratio of specific heats,
            $\gamma$

    Returns:
        NDArrayFloat: total temperature ratio, $T_0 / T$

    """
    gam = np.atleast_1d(specific_heat_ratio)
    m = np.atleast_1d(mach)
    return 1 + 0.5 * (gam - 1) * m**2


def mach_from_temperature(
    temperature_ratio: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat,
) -> NDArrayFloat:
    r"""Computes the Mach number $M$ from total temperature ratio $T_0 / T$

    Args:
        temperature_ratio (ArraylikeFloat): total temperature ratio,
            $T_0 / T$
        specific_heat_ratio (ArraylikeFloat): ratio of specific heats,
            $\gamma$

    Returns:
        ArraylikeFloat: Mach number, $M$

    """
    gam = np.atleast_1d(specific_heat_ratio)
    tratio = np.atleast_1d(temperature_ratio)
    return (2.0 / (gam - 1) * (tratio - 1)) ** 0.5


def total_pressure_ratio(
    mach: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat,
) -> NDArrayFloat:
    r"""Computes the stagnation or total pressure ratio, $p_0 / p$

    Args:
        mach (ArraylikeFloat): Mach number, $M$
        specific_heat_ratio (ArraylikeFloat): ratio of specific heats,
            $\gamma$

    Returns:
        NDArrayFloat: total pressure ratio, $p_0 / p$

    """
    gam = np.atleast_1d(specific_heat_ratio)
    return total_temperature_ratio(mach=mach, specific_heat_ratio=gam) ** (
        gam / (gam - 1)
    )


def total_density_ratio(
    mach: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat,
) -> NDArrayFloat:
    r"""Computes the stagnation or total density ratio, $\rho_0 / \rho$

    Args:
        mach (ArraylikeFloat): Mach number, $M$
        specific_heat_ratio (ArraylikeFloat): ratio of specific heats,
            $\gamma$

    Returns:
        NDArrayFloat: total density ratio, $\rho_0 / \rho$

    """
    gam = np.atleast_1d(specific_heat_ratio)
    return total_temperature_ratio(mach=mach, specific_heat_ratio=gam) ** (
        1.0 / (gam - 1)
    )


def total_speed_of_sound_ratio(
    mach: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat,
) -> NDArrayFloat:
    r"""Computes the stagnation or total speed of sound ratio, $a_0 / a$

    Args:
        mach (ArraylikeFloat): Mach number, $M$
        specific_heat_ratio (ArraylikeFloat): ratio of specific heats,
            $\gamma$

    Returns:
        NDArrayFloat: total speed of sound ratio, $a_0 / a$

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
    mach: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat,
) -> NDArrayFloat:
    r"""Computes the area ratio $A / A^*$ for an isentropic nozzle.

    This is the standard area-Mach number relation

    Args:
        mach (ArraylikeFloat): Mach number, $M$
        specific_heat_ratio (ArraylikeFloat): ratio of specific heats,
            $\gamma$

    Returns:
        NDArrayFloat: area ratio, $A / A^*$

    """
    return _area_mach_relation_sqr(m=mach, gam=specific_heat_ratio) ** 0.5


def mach_from_area_ratio(
    area_ratio: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat,
    flow_regime: ArraylikeFlowSpeedRegime,
) -> NDArrayFloat:
    r"""Compute the Mach number for a known area ratio, $A / A^*$

    Args:
        area_ratio (ArraylikeFloat): area ratio, $A / A^*$
        specific_heat_ratio (ArraylikeFloat): ratio of specific heats, $\gamma$
        flow_regime (ArraylikeFlowSpeedRegime): Is flowfield
            subsonic or supersonic.

    Returns:
        ArraylikeFloat: supersonic or subsonic Mach solution
            for a given area ratio

    """
    aratios = np.atleast_1d(area_ratio)
    gam = np.atleast_1d(specific_heat_ratio)

    mach_brackets = bracket_mach_from_flow_regime(flow_regime)

    def compute_mach(_m, _aratio, _gam):
        return _aratio**2 - _area_mach_relation_sqr(m=_m, gam=_gam)

    res = find_root(compute_mach, mach_brackets, args=(aratios, gam))
    if not np.all(res.success):
        raise RootFindingError(f"find_root did not succeed: {res.status}")
    return res.x


def mach_angle(mach: ArraylikeFloat) -> NDArrayFloat:
    r"""Compute the Mach angle, $\mu$ [radians].

    Args:
        mach (ArraylikeFloat): Mach number, $M$

    Returns:
        NDArrayFloat: Mach angle, $\mu$ [radians]

    """
    m = np.atleast_1d(mach)
    return np.asin(1.0 / np.atleast_1d(m))
