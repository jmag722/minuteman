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
from minuteman.utils.bounds_check import (
    OutOfBoundsError,
    check_positive,
    check_specific_heat_ratio,
)
from minuteman.utils.types import (
    ArraylikeFloat,
    NDArrayFloat,
    RootFindingError,
    broadcast_inputs,
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

    mach_angle: NDArrayFloat
    r"""Mach angle, $\mu$ [radians]. Values of NaN indicate there is no
        Mach angle for this regime (subsonic flow)."""

    prandtl_meyer_func: NDArrayFloat
    r"""Prandtl-Meyer function, $\nu$ [radians]. Values of NaN indictate there
        is no valid value for this regime (subsonic flow)."""

    specific_heat_ratio: NDArrayFloat
    r"""ratio of specific heats, $\gamma$"""


def lookup_table_by_mach(
    mach: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat = 1.4,
) -> IsentropicFlowTable:
    r"""Lookup the isentropic flow table based on Mach number, $M$

    Args:
        mach: Mach number, $M$. Bounds $(0, \infty)$
        specific_heat_ratio: ratio of specific heats, $\gamma$.
            Bounds: $[1, 1.67]$

    Returns:
        Isentropic flow table result

    Raises:
        OutOfBoundsError: invalid inputs
    """
    m, gam = broadcast_inputs(mach, specific_heat_ratio)
    check_positive(m)
    check_specific_heat_ratio(gam)
    p0_ratio = total_pressure_ratio_by_mach(mach=m, specific_heat_ratio=gam)
    r0_ratio = total_density_ratio_by_mach(mach=m, specific_heat_ratio=gam)
    t0_ratio = total_temperature_ratio_by_mach(mach=m, specific_heat_ratio=gam)
    a0_ratio = total_speed_of_sound_ratio_by_mach(
        mach=m, specific_heat_ratio=gam
    )
    area_ratio = area_ratio_by_mach(mach=m, specific_heat_ratio=gam)

    supersonic_mach = m >= 1.0
    mu = np.full_like(m, np.nan)
    mu[supersonic_mach] = mach_angle(m[supersonic_mach])

    pmf = np.full_like(m, np.nan)
    pmf[supersonic_mach] = prandtl_meyer_func(
        mach=m[supersonic_mach], specific_heat_ratio=gam[supersonic_mach]
    )

    return IsentropicFlowTable(
        mach=m,
        temperature=t0_ratio,
        pressure=p0_ratio,
        density=r0_ratio,
        speed_of_sound=a0_ratio,
        area_ratio=area_ratio,
        mach_angle=mu,
        prandtl_meyer_func=pmf,
        specific_heat_ratio=gam,
    )


def lookup_table_by_temperature(
    temperature_ratio: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat = 1.4,
) -> IsentropicFlowTable:
    r"""Lookup the isentropic flow table based on total temperature ratio,
    $T_0 / T$

    Args:
        temperature_ratio: total temperature ratio, $T_0 / T$.
            Bounds: $(1, \infty)$
        specific_heat_ratio: ratio of specific heats, $\gamma$.
            Bounds: $[1, 1.67]$

    Returns:
        Isentropic flow table result

    Raises:
        OutOfBoundsError: invalid inputs
    """
    tratio, gam = broadcast_inputs(temperature_ratio, specific_heat_ratio)
    if np.any(tratio <= 1.0):
        raise OutOfBoundsError("Temperature ratio T0/T must be > 1.0")
    check_specific_heat_ratio(gam)

    mach = mach_by_temperature(
        temperature_ratio=tratio,
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
        pressure_ratio: total pressure ratio, $p_0 / p$. Bounds: $(1, \infty)$
        specific_heat_ratio: ratio of specific heats, $\gamma$.
            Bounds: $[1, 1.67]$

    Returns:
        Isentropic flow table result

    Raises:
        OutOfBoundsError: invalid inputs
    """
    pratio, gam = broadcast_inputs(pressure_ratio, specific_heat_ratio)
    if np.any(pratio <= 1.0):
        raise OutOfBoundsError("Pressure ratio p0/p must be > 1.0")
    check_specific_heat_ratio(gam)

    t0_ratio = thermo.isentropic_process_by_pressure(
        pressure_ratio=pratio,
        specific_heat_ratio=gam,
    ).temperature_ratio
    return lookup_table_by_temperature(
        temperature_ratio=t0_ratio,
        specific_heat_ratio=gam,
    )


def lookup_table_by_density(
    density_ratio: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat = 1.4,
) -> IsentropicFlowTable:
    r"""Lookup the isentropic flow table based on total density ratio,
    $\rho_0 / \rho$

    Args:
        density_ratio: total density ratio, $\rho_0 / \rho$.
            Bounds: $(1, \infty)$
        specific_heat_ratio: ratio of specific heats, $\gamma$.
            Bounds: $[1, 1.67]$

    Returns:
        Isentropic flow table result

    Raises:
        OutOfBoundsError: invalid inputs
    """
    rratio, gam = broadcast_inputs(density_ratio, specific_heat_ratio)
    if np.any(rratio <= 1.0):
        raise OutOfBoundsError("Density ratio rho0/rho must be > 1.0")
    check_specific_heat_ratio(gam)

    t0_ratio = thermo.isentropic_process_by_density(
        density_ratio=rratio,
        specific_heat_ratio=gam,
    ).temperature_ratio
    return lookup_table_by_temperature(
        temperature_ratio=t0_ratio,
        specific_heat_ratio=gam,
    )


def lookup_table_by_speed_of_sound(
    speed_of_sound_ratio: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat = 1.4,
) -> IsentropicFlowTable:
    r"""Lookup the isentropic flow table based on total speed of sound ratio,
    $a_0 / a$

    Args:
        speed_of_sound_ratio: total speed of sound ratio, $a_0 / a$.
            Bounds: $(1, \infty)$
        specific_heat_ratio: ratio of specific heats, $\gamma$.
            Bounds: $[1, 1.67]$

    Returns:
        Isentropic flow table result

    Raises:
        OutOfBoundsError: invalid inputs
    """
    aratio, gam = broadcast_inputs(speed_of_sound_ratio, specific_heat_ratio)
    if np.any(aratio <= 1.0):
        raise OutOfBoundsError("Speed of sound ratio a0/a must be > 1.0")
    check_specific_heat_ratio(gam)

    t0_ratio = thermo.isentropic_process_by_speed_of_sound(
        speed_of_sound_ratio=aratio,
        specific_heat_ratio=gam,
    ).temperature_ratio
    return lookup_table_by_temperature(
        temperature_ratio=t0_ratio,
        specific_heat_ratio=gam,
    )


def lookup_table_by_area_ratio(
    area_ratio: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat = 1.4,
    flow_regime: ArraylikeFlowSpeedRegime = FlowSpeedRegime.supersonic,
) -> IsentropicFlowTable:
    r"""Lookup the isentropic flow table based on area ratio, $A / A^*$

    Args:
        area_ratio: area ratio, $A / A^*$. Bounds: $(1, \infty)$
        specific_heat_ratio: ratio of specific heats, $\gamma$.
            Bounds: $[1, 1.67]$
        flow_regime: Is flowfield subsonic or supersonic.

    Returns:
        Isentropic flow table result

    Raises:
        OutOfBoundsError: invalid inputs
    """
    aratio, gam, fr = broadcast_inputs(
        area_ratio, specific_heat_ratio, flow_regime
    )
    if np.any(aratio <= 1.0):
        raise OutOfBoundsError("Area ratio A/A* must be > 1.0")
    check_specific_heat_ratio(gam)

    mach = mach_by_area_ratio(
        area_ratio=aratio,
        specific_heat_ratio=gam,
        flow_regime=fr,
    )
    return lookup_table_by_mach(mach=mach, specific_heat_ratio=gam)


def lookup_table_by_mach_angle(
    mach_angle: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat = 1.4,
) -> IsentropicFlowTable:
    r"""Lookup the isentropic flow table based on Mach angle, $\mu$

    Args:
        mach_angle: Mach angle, $\mu$ [radians]. Bounds: $(0, 90^\circ]$
        specific_heat_ratio: ratio of specific heats, $\gamma$.
            Bounds: $[1, 1.67]$

    Returns:
        Isentropic flow table result

    Raises:
        OutOfBoundsError: invalid inputs
    """
    mu, gam = broadcast_inputs(mach_angle, specific_heat_ratio)
    if np.any((mu <= 0) | (mu > 0.5 * np.pi)):
        raise OutOfBoundsError("Mach angle mu must be within (0, 90] deg")
    check_specific_heat_ratio(gam)

    mach = 1.0 / np.sin(mu)
    return lookup_table_by_mach(mach=mach, specific_heat_ratio=gam)


def lookup_table_by_prandtl_meyer(
    prandtl_meyer: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat = 1.4,
) -> IsentropicFlowTable:
    r"""Lookup the isentropic flow table based on the Prandtl-Meyer function,
        $\nu$.

    Args:
        prandtl_meyer: Prandtl-Meyer function, $\nu$ [radians].
            Bounds: $\left[0,
            \frac{\pi}{2}
            \left(\sqrt{\frac{\gamma+1}{\gamma-1}}-1\right)\right)$
        specific_heat_ratio: ratio of specific heats, $\gamma$.
            Bounds: $[1, 1.67]$

    Returns:
        Isentropic flow table result

    Raises:
        OutOfBoundsError: invalid inputs
    """
    pm, gam = broadcast_inputs(prandtl_meyer, specific_heat_ratio)
    check_specific_heat_ratio(gam)
    pm_max = 0.5 * np.pi * (np.sqrt((gam + 1) / (gam - 1)) - 1)
    if np.any((pm < 0) | (pm >= pm_max)):
        pm_deg = np.degrees(pm_max)
        raise OutOfBoundsError(
            f"Prandtl-Meyer function nu must be within [0, {pm_deg}) deg"
        )
    # invert the PM-mach relationship

    def pmfunc(mguess, _pm, _g):
        return _pm - prandtl_meyer_func(
            mach=mguess,
            specific_heat_ratio=_g,
        )

    m_bracket = (1.0, 1e10)
    res = find_root(pmfunc, m_bracket, args=(pm, gam))
    if not np.all(res.success):
        raise RootFindingError(f"find_root did not succeed: {res.status}")

    m = res.x
    return lookup_table_by_mach(mach=m, specific_heat_ratio=gam)


def total_temperature_ratio_by_mach(
    mach: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat,
) -> NDArrayFloat:
    r"""Computes the stagnation or total temperature ratio, $T_0 / T$

    Args:
        mach: Mach number, $M$
        specific_heat_ratio: ratio of specific heats, $\gamma$

    Returns:
        Total temperature ratio, $T_0 / T$

    """
    gam = np.atleast_1d(specific_heat_ratio)
    m = np.atleast_1d(mach)
    return 1 + 0.5 * (gam - 1) * m**2


def mach_by_temperature(
    temperature_ratio: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat,
) -> NDArrayFloat:
    r"""Computes the Mach number $M$ from total temperature ratio $T_0 / T$

    Args:
        temperature_ratio: total temperature ratio, $T_0 / T$
        specific_heat_ratio: ratio of specific heats, $\gamma$

    Returns:
        Mach number, $M$

    """
    gam = np.atleast_1d(specific_heat_ratio)
    tratio = np.atleast_1d(temperature_ratio)
    return (2.0 / (gam - 1) * (tratio - 1)) ** 0.5


def total_pressure_ratio_by_mach(
    mach: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat,
) -> NDArrayFloat:
    r"""Computes the stagnation or total pressure ratio, $p_0 / p$

    Args:
        mach: Mach number, $M$
        specific_heat_ratio: ratio of specific heats, $\gamma$

    Returns:
        Total pressure ratio, $p_0 / p$

    """
    gam = np.atleast_1d(specific_heat_ratio)
    return total_temperature_ratio_by_mach(
        mach=mach, specific_heat_ratio=gam
    ) ** (gam / (gam - 1))


def total_density_ratio_by_mach(
    mach: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat,
) -> NDArrayFloat:
    r"""Computes the stagnation or total density ratio, $\rho_0 / \rho$

    Args:
        mach: Mach number, $M$
        specific_heat_ratio: ratio of specific heats, $\gamma$

    Returns:
        Total density ratio, $\rho_0 / \rho$

    """
    gam = np.atleast_1d(specific_heat_ratio)
    return total_temperature_ratio_by_mach(
        mach=mach, specific_heat_ratio=gam
    ) ** (1.0 / (gam - 1))


def total_speed_of_sound_ratio_by_mach(
    mach: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat,
) -> NDArrayFloat:
    r"""Computes the stagnation or total speed of sound ratio, $a_0 / a$

    Args:
        mach: Mach number, $M$
        specific_heat_ratio: ratio of specific heats, $\gamma$

    Returns:
        Total speed of sound ratio, $a_0 / a$

    """
    gam = np.atleast_1d(specific_heat_ratio)
    return (
        total_temperature_ratio_by_mach(mach=mach, specific_heat_ratio=gam)
        ** 0.5
    )


def _area_ratio_by_mach_sqr(m, gam):
    return m**-2 * (
        2.0
        / (gam + 1)
        * total_temperature_ratio_by_mach(mach=m, specific_heat_ratio=gam)
    ) ** ((gam + 1) / (gam - 1))


def area_ratio_by_mach(
    mach: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat,
) -> NDArrayFloat:
    r"""Computes the area ratio $A / A^*$ for an isentropic nozzle.

    This is the standard area-Mach number relation

    Args:
        mach: Mach number, $M$
        specific_heat_ratio: ratio of specific heats, $\gamma$

    Returns:
        Area ratio, $A / A^*$

    """
    return _area_ratio_by_mach_sqr(m=mach, gam=specific_heat_ratio) ** 0.5


def mach_by_area_ratio(
    area_ratio: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat,
    flow_regime: ArraylikeFlowSpeedRegime,
) -> NDArrayFloat:
    r"""Compute the Mach number for a known area ratio, $A / A^*$

    Args:
        area_ratio: area ratio, $A / A^*$
        specific_heat_ratio: ratio of specific heats, $\gamma$
        flow_regime: Is flowfield subsonic or supersonic.

    Returns:
        Supersonic or subsonic Mach solution for a given area ratio

    """
    aratio, gam, fr = broadcast_inputs(
        area_ratio, specific_heat_ratio, flow_regime
    )

    mach_brackets = bracket_mach_from_flow_regime(fr)

    def compute_mach(_m, _aratio, _gam):
        return _aratio**2 - _area_ratio_by_mach_sqr(m=_m, gam=_gam)

    res = find_root(compute_mach, mach_brackets, args=(aratio, gam))
    if not np.all(res.success):
        raise RootFindingError(f"find_root did not succeed: {res.status}")
    return res.x


def mach_angle(mach: ArraylikeFloat) -> NDArrayFloat:
    r"""Compute the Mach angle, $\mu$ [radians].

    Args:
        mach: Mach number, $M$

    Returns:
        Mach angle, $\mu$ [radians]

    """
    m = np.atleast_1d(mach)
    return np.asin(1.0 / np.atleast_1d(m))


def prandtl_meyer_func(
    mach: ArraylikeFloat, specific_heat_ratio: ArraylikeFloat
) -> NDArrayFloat:
    r"""Compute the Prandtl-Meyer function, $\nu$

    Args:
        mach: Mach number, $M$
        specific_heat_ratio: ratio of specific heats, $\gamma$

    Returns:
        Prandtl-Meyer function, $\nu$ [radians]

    """
    m = np.atleast_1d(mach)
    gam = np.atleast_1d(specific_heat_ratio)
    return np.sqrt((gam + 1) / (gam - 1)) * np.atan(
        np.sqrt((gam - 1) / (gam + 1) * (m**2 - 1))
    ) - np.atan(np.sqrt(m**2 - 1))
