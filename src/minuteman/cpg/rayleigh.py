# Copyright (c) 2022-2026 Jared Magnusson
# SPDX-License-Identifier: Apache-2.0

"""```python
 import minuteman.cpg.rayleigh as rayleigh
```

This module computes 1D, calorically perfect flow with heat addition
(Rayleigh flow).
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
from scipy.optimize.elementwise import find_root

from minuteman.cpg import (
    ArraylikeFlowSpeedRegime,
    FlowSpeedRegime,
    InvalidFlowRegimeError,
    isentropic_flow,
)
from minuteman.cpg.base import bracket_mach_from_flow_regime
from minuteman.utils.bounds_check import (
    OutOfBoundsError,
    check_nonnegative,
    check_positive,
    check_specific_heat_ratio,
)
from minuteman.utils.types import (
    ArraylikeFloat,
    InvalidArrayShapeError,
    NDArrayFloat,
    RootFindingError,
)


@dataclass
class RayleighFlowTable:
    r"""Rayleigh flow table where the initial Mach number $M_1=M^*=1.0$"""

    mach: NDArrayFloat
    r"""Mach number, $M$."""

    temperature_ratio: NDArrayFloat
    r"""Static temperature ratio, $T / T^*$"""

    pressure_ratio: NDArrayFloat
    r"""Static pressure ratio, $p / p^*$"""

    density_ratio: NDArrayFloat
    r"""Density ratio, $\rho / \rho^*$"""

    @property
    def velocity_ratio(self) -> NDArrayFloat:
        r"""Velocity ratio, $u / u^*$"""
        return 1.0 / self.density_ratio

    total_pressure_ratio: NDArrayFloat
    r"""Total pressure ratio, $p_0 / p_0^*$"""

    total_temperature_ratio: NDArrayFloat
    r"""Total temperature ratio, $T_0 / T_0^*$"""

    entropy_ratio: NDArrayFloat
    r"""Specific entropy ratio, $(s^* - s) / R$"""

    specific_heat_ratio: NDArrayFloat
    r"""Ratio of specific heats, $\gamma$"""


def lookup_table_by_mach(
    mach: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat = 1.4,
) -> RayleighFlowTable:
    r"""Look up a Rayleigh flow table result from the Mach number, $M$

    Args:
        mach (ArraylikeFloat): Mach number, $M$. Bounds: $(0, \infty)$
        specific_heat_ratio (ArraylikeFloat, optional): ratio of specific
            heats, $\gamma$. Bounds $[1, 1.67]$

    Returns:
        RayleighFlowTable: Rayleigh flow output table

    Raises:
        OutOfBoundsError: invalid inputs
    """
    m1 = 1.0
    m2 = np.atleast_1d(mach)
    gam = np.atleast_1d(specific_heat_ratio)
    check_positive(m2)
    check_specific_heat_ratio(gam)

    return RayleighFlowTable(
        mach=m2,
        temperature_ratio=temperature_ratio_by_mach(
            mach_initial=m1,
            mach_final=m2,
            specific_heat_ratio=gam,
        ),
        pressure_ratio=pressure_ratio_by_mach(
            mach_initial=m1,
            mach_final=m2,
            specific_heat_ratio=gam,
        ),
        density_ratio=density_ratio_by_mach(
            mach_initial=m1,
            mach_final=m2,
            specific_heat_ratio=gam,
        ),
        total_pressure_ratio=total_pressure_ratio_by_mach(
            mach_initial=m1,
            mach_final=m2,
            specific_heat_ratio=gam,
        ),
        total_temperature_ratio=total_temperature_ratio_by_mach(
            mach_initial=m1,
            mach_final=m2,
            specific_heat_ratio=gam,
        ),
        entropy_ratio=_rev_entropy_ratio_by_mach(
            mach_initial=m1,
            mach_final=m2,
            specific_heat_ratio=gam,
        ),
        specific_heat_ratio=gam,
    )


def lookup_table_by_pressure(
    pressure_ratio: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat = 1.4,
) -> RayleighFlowTable:
    r"""Look up a Rayleigh flow table result from the static pressure_ratio,
    $p / p^*$

    Args:
        pressure_ratio (ArraylikeFloat): static pressure ratio, $p / p^*$.
            Bounds: $(0, \gamma+1)$
        specific_heat_ratio (ArraylikeFloat, optional): ratio of specific
            heats, $\gamma$. Bounds $[1, 1.67]$

    Returns:
        RayleighFlowTable: Rayleigh flow output table

    Raises:
        OutOfBoundsError: invalid inputs
    """
    m1 = 1.0
    p_ratio = np.atleast_1d(pressure_ratio)
    gam = np.atleast_1d(specific_heat_ratio)
    pratio_max = 1.0 + gam
    if np.any((p_ratio <= 0.0) | (p_ratio >= pratio_max)):
        raise OutOfBoundsError(
            f"Pressure ratio p/p* must be within (0.0, {pratio_max})"
        )
    check_specific_heat_ratio(gam)

    # invert relationship between $p_2 / p_1$ and $M_1$, $M_2$
    m2 = (((1 + gam * m1**2) / p_ratio - 1) / gam) ** 0.5
    return lookup_table_by_mach(mach=m2, specific_heat_ratio=gam)


def _lookup_table_by_ratio(
    ratio: NDArrayFloat,
    specific_heat_ratio: NDArrayFloat,
    flow_regime: ArraylikeFlowSpeedRegime,
    mach_func: Callable,
) -> RayleighFlowTable:
    r"""Lookup the Rayleigh flow table by a generic input ratio where the
    relationship with Mach must be solved numerically

    Args:
        ratio (ArraylikeFloat): ratio of interest
            (total temp., total pressure, etc)
        specific_heat_ratio (ArraylikeFloat): ratio of specific heats,
            $\gamma$
        flow_regime (ArraylikeFlowSpeedRegime): flow regime
            (supersonic, subsonic)
        mach_func (Callable): function to compute Mach from the given ratio

    Returns:
        RayleighFlowTable: Rayleigh flow table

    """
    gam = specific_heat_ratio
    mach_brackets = bracket_mach_from_flow_regime(flow_regime)

    def get_mach_by_ratio(_m, _r, _g):
        return _r - mach_func(
            mach_initial=1.0,
            mach_final=_m,
            specific_heat_ratio=_g,
        )

    res = find_root(get_mach_by_ratio, mach_brackets, args=(ratio, gam))
    if not np.all(res.success):
        raise RootFindingError(f"find_root did not succeed: {res.status}")
    m2 = res.x
    return lookup_table_by_mach(mach=m2, specific_heat_ratio=gam)


class RayleighTemperatureRegime(Enum):
    r"""Is the condition above or below $M=1/\gamma$ where max temperature
    occurs"""

    lowspeed = auto()
    r"""$M < 1/\gamma$, subsonic, to the left of $T_{max}$ on Rayleigh curve"""
    highspeed = auto()
    r"""$M > 1/\gamma$, subsonic or supersonic, to the right of $T_{max}$
    on the Rayleigh curve"""


ArraylikeRayleighTemperatureRegime: TypeAlias = (
    npt.NDArray[np.object_]
    | list[RayleighTemperatureRegime]
    | RayleighTemperatureRegime
)
"""Arraylike of RayleighTemperatureRegime objects"""


def _bracket_mach_from_rayleigh_temperature_regime(
    flow_regime: ArraylikeRayleighTemperatureRegime,
    mach_at_tmax: NDArrayFloat,
) -> tuple[NDArrayFloat, NDArrayFloat]:
    # these are mach number bounds
    left_tmax_min = 1e-50
    left_tmax_max = mach_at_tmax
    right_tmax_min = mach_at_tmax
    right_tmax_max = 1e10
    if flow_regime is RayleighTemperatureRegime.highspeed:
        return (
            np.atleast_1d(right_tmax_min),
            np.atleast_1d(right_tmax_max),
        )
    if flow_regime is RayleighTemperatureRegime.lowspeed:
        return (
            np.atleast_1d(left_tmax_min),
            np.atleast_1d(left_tmax_max),
        )
    if isinstance(flow_regime, np.ndarray | list):
        return (
            np.where(
                flow_regime == RayleighTemperatureRegime.lowspeed,
                left_tmax_min,
                right_tmax_min,
            ),
            np.where(
                flow_regime == RayleighTemperatureRegime.highspeed,
                right_tmax_max,
                left_tmax_max,
            ),
        )
    raise InvalidFlowRegimeError(
        "Use ArraylikeRayleighTemperatureRegime to set flow_regime",
    )


def lookup_table_by_temperature(
    temperature_ratio: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat = 1.4,
    flow_regime: ArraylikeRayleighTemperatureRegime = (
        RayleighTemperatureRegime.highspeed
    ),
) -> RayleighFlowTable:
    r"""Look up a Rayleigh flow table result from the static temperature ratio,
    $T / T^*$

    Args:
        temperature_ratio (ArraylikeFloat): static temperature ratio,
            $T / T^*$. Bounds: $\left(0, \frac{1}{4} \left(\gamma +
                \frac{1}{\gamma}\right) + \frac{1}{2} \right]$
        specific_heat_ratio (ArraylikeFloat, optional): ratio of specific
            heats, $\gamma$. Bounds $[1, 1.67]$
        flow_regime (ArraylikeRayleighTemperatureRegime, optional):
            Rayleigh flow speed regime based upon $T_{max}$.

    Returns:
        RayleighFlowTable: Rayleigh flow output table

    Raises:
        OutOfBoundsError: invalid inputs
    """
    tratio = np.atleast_1d(temperature_ratio)
    gam = np.atleast_1d(specific_heat_ratio)
    tratio_max = temperature_ratio_by_mach(
        mach_initial=1.0, mach_final=1.0 / gam**0.5, specific_heat_ratio=gam
    )
    if np.any((tratio <= 0.0) | (tratio > tratio_max)):
        raise OutOfBoundsError(
            f"Temperature ratio T/T* must be within (0.0, {tratio_max}]"
        )
    check_specific_heat_ratio(gam)

    mach_func = temperature_ratio_by_mach
    mach_brackets = _bracket_mach_from_rayleigh_temperature_regime(
        flow_regime, tratio_max
    )

    def get_mach_by_ratio(_m, _r, _g):
        return _r - mach_func(
            mach_initial=1.0,
            mach_final=_m,
            specific_heat_ratio=_g,
        )

    res = find_root(get_mach_by_ratio, mach_brackets, args=(tratio, gam))
    if not np.all(res.success):
        raise RootFindingError(f"find_root did not succeed: {res.status}")
    m2 = res.x
    return lookup_table_by_mach(mach=m2, specific_heat_ratio=gam)


def lookup_table_by_density(
    density_ratio: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat = 1.4,
) -> RayleighFlowTable:
    r"""Look up a Rayleigh flow table result from the static density ratio,
    $\rho / \rho^*$

    Args:
        density_ratio (ArraylikeFloat): static density ratio,
            $\rho / \rho^*$. Bounds: $\left(\frac{\gamma}{1+\gamma},
            \infty \right)$
        specific_heat_ratio (ArraylikeFloat, optional): ratio of specific
            heats, $\gamma$. Bounds $[1, 1.67]$

    Returns:
        RayleighFlowTable: Rayleigh flow output table

    Raises:
        OutOfBoundsError: invalid inputs
    """
    m1 = 1.0
    r_ratio = np.atleast_1d(density_ratio)
    gam = np.atleast_1d(specific_heat_ratio)
    r_ratio_max = gam / (1.0 + gam)
    if np.any(r_ratio <= r_ratio_max):
        raise OutOfBoundsError(
            f"Density ratio rho/rho* must be > {r_ratio_max}"
        )
    check_specific_heat_ratio(gam)
    m2 = ((r_ratio * (1 + gam * m1**2)) / m1**2 - gam) ** -0.5
    return lookup_table_by_mach(mach=m2, specific_heat_ratio=gam)


def lookup_table_by_total_pressure(
    total_pressure_ratio: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat = 1.4,
    flow_regime: ArraylikeFlowSpeedRegime = FlowSpeedRegime.supersonic,
) -> RayleighFlowTable:
    r"""Look up a Rayleigh flow table result from the total pressure ratio,
    $p_0 / p_0^*$

    Args:
        total_pressure_ratio (ArraylikeFloat): total pressure ratio,
            $p_0 / p_0^*$.
            Subsonic Bounds: $\left[1, \left(1+\gamma\right)
                \left(\frac{2}{\gamma+1}\right)^\frac{\gamma}{\gamma-1}\right)$
            , Supersonic Bounds: $[1, \infty)$
        specific_heat_ratio (ArraylikeFloat, optional): ratio of specific
            heats, $\gamma$. Bounds $[1, 1.67]$
        flow_regime (ArraylikeFlowSpeedRegime, optional):
            flow speed regime (either supersonic or subsonic).

    Returns:
        RayleighFlowTable: Rayleigh flow output table

    Raises:
        OutOfBoundsError: invalid inputs
    """
    p0ratio = np.atleast_1d(total_pressure_ratio)
    gam = np.atleast_1d(specific_heat_ratio)
    if isinstance(flow_regime, FlowSpeedRegime):
        fr = np.full(p0ratio.shape, flow_regime)
    else:
        fr = np.atleast_1d(np.asarray(flow_regime))
    if p0ratio.shape != fr.shape:
        raise InvalidArrayShapeError(
            "total_pressure_ratio and flow_regime shapes must match,"
            "or flow_regime must be a scalar"
        )

    check_specific_heat_ratio(gam)

    # p0/p0* limit when M=0
    p0_max_sub = (1 + gam) * (2 / (gam + 1)) ** (gam / (gam - 1))
    if np.any(p0ratio[fr == FlowSpeedRegime.supersonic] < 1.0):
        raise OutOfBoundsError(
            "Total pressure ratio p0/p0* must be >= 1.0 for supersonic flow"
        )
    if np.any(
        (p0ratio[fr == FlowSpeedRegime.subsonic] < 1.0)
        | (p0ratio[fr == FlowSpeedRegime.subsonic] >= p0_max_sub)
    ):
        raise OutOfBoundsError(
            f"Total pressure ratio p0/p0* must be within [1.0, "
            f"{p0_max_sub}) for subsonic flow"
        )

    return _lookup_table_by_ratio(
        ratio=p0ratio,
        specific_heat_ratio=gam,
        flow_regime=flow_regime,
        mach_func=total_pressure_ratio_by_mach,
    )


def lookup_table_by_total_temperature(
    total_temperature_ratio: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat = 1.4,
    flow_regime: ArraylikeFlowSpeedRegime = FlowSpeedRegime.supersonic,
) -> RayleighFlowTable:
    r"""Look up a Rayleigh flow table result from the total temperature ratio,
    $T_0 / T_0^*$

    Args:
        total_temperature_ratio (ArraylikeFloat): total temperature ratio,
            $T_0 / T_0^*$.
            Subsonic Bounds: $(0, 1]$,
            Supersonic Bounds: $\left[\frac{(\gamma+1)(\gamma-1)}{\gamma^2},
                                      1\right]$
        specific_heat_ratio (ArraylikeFloat, optional): ratio of specific
            heats, $\gamma$. Bounds $[1, 1.67]$
        flow_regime (ArraylikeFlowSpeedRegime, optional):
            flow speed regime (either supersonic or subsonic).

    Returns:
        RayleighFlowTable: Rayleigh flow output table

    Raises:
        OutOfBoundsError: invalid inputs
    """
    t0ratio = np.atleast_1d(total_temperature_ratio)
    gam = np.atleast_1d(specific_heat_ratio)
    if isinstance(flow_regime, FlowSpeedRegime):
        fr = np.full(t0ratio.shape, flow_regime)
    else:
        fr = np.atleast_1d(np.asarray(flow_regime))
    if t0ratio.shape != fr.shape:
        raise InvalidArrayShapeError(
            "total_temperature_ratio and flow_regime shapes must match,"
            "or flow_regime must be a scalar"
        )

    check_specific_heat_ratio(gam)

    t0_min_sup = (gam + 1) * (gam - 1) / gam**2

    if np.any(
        (t0ratio[fr == FlowSpeedRegime.supersonic] > 1.0)
        | (t0ratio[fr == FlowSpeedRegime.supersonic] < t0_min_sup)
    ):
        raise OutOfBoundsError(
            f"Total temperature ratio T0/T0* must be "
            f"within [{t0_min_sup}, 1.0] for supersonic flow"
        )
    if np.any(
        (t0ratio[fr == FlowSpeedRegime.subsonic] > 1.0)
        | (t0ratio[fr == FlowSpeedRegime.subsonic] <= 0.0)
    ):
        raise OutOfBoundsError(
            "Total temperature ratio T0/T0* must be "
            "within (0.0, 1.0] for subsonic flow"
        )

    return _lookup_table_by_ratio(
        ratio=t0ratio,
        specific_heat_ratio=gam,
        flow_regime=flow_regime,
        mach_func=total_temperature_ratio_by_mach,
    )


def lookup_table_by_entropy(
    entropy_ratio: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat = 1.4,
    flow_regime: ArraylikeFlowSpeedRegime = FlowSpeedRegime.supersonic,
) -> RayleighFlowTable:
    r"""Look up a Rayleigh flow table result from the specific entropy ratio,
    $(s^* - s) / R$

    Args:
        entropy_ratio (ArraylikeFloat): specific entropy ratio,
            $(s^* - s) / R$. Bounds: $[0, \infty)$
        specific_heat_ratio (ArraylikeFloat, optional): ratio of specific
            heats, $\gamma$. Bounds $[1, 1.67]$
        flow_regime (ArraylikeFlowSpeedRegime, optional):
            flow speed regime (either supersonic or subsonic).

    Returns:
        RayleighFlowTable: Rayleigh flow output table

    Raises:
        OutOfBoundsError: invalid inputs
    """
    sratio = np.atleast_1d(entropy_ratio)
    gam = np.atleast_1d(specific_heat_ratio)
    check_specific_heat_ratio(gam)
    check_nonnegative(sratio)

    return _lookup_table_by_ratio(
        ratio=sratio,
        specific_heat_ratio=gam,
        flow_regime=flow_regime,
        mach_func=_rev_entropy_ratio_by_mach,
    )


def _rev_entropy_ratio_by_mach(
    mach_initial: ArraylikeFloat,
    mach_final: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat,
) -> NDArrayFloat:
    """This function computes (s1-s2)/R. This is handy because s1 becomes
    the critical/sonic point, s*, and this allows you to get the typical
    positive value back out
    """
    m1 = np.atleast_1d(mach_initial)
    m2 = np.atleast_1d(mach_final)
    gam = np.atleast_1d(specific_heat_ratio)
    return -entropy_ratio_by_mach(
        mach_final=m2,
        mach_initial=m1,
        specific_heat_ratio=gam,
    )


def entropy_ratio_by_mach(
    mach_initial: ArraylikeFloat,
    mach_final: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat,
) -> NDArrayFloat:
    r"""Compute the specific entropy ratio $(s_2-s_1) / R$ from the
    Mach number $M$ for Rayleigh flow.

    Args:
        mach_initial (ArraylikeFloat): Initial Mach number, $M_1$. This is
            the reference Mach number, $M^*$, when equal to unity.
        mach_final (ArraylikeFloat): Final Mach number, $M_2$. This is
            simply the Mach number, $M$, when ``mach_initial==1.0``
        specific_heat_ratio (ArraylikeFloat): ratio of specific heats,
            $\gamma$

    Returns:
        NDArrayFloat: entropy ratio, $(s_2-s_1) / R$
            ($(s - s^*) / R$ if $M_1=1.0$)

    """
    m1 = np.atleast_1d(mach_initial)
    m2 = np.atleast_1d(mach_final)
    gam = np.atleast_1d(specific_heat_ratio)
    return np.log(
        ((1 + gam * m1**2) / (1 + gam * m2**2)) ** (gam + 1)
        * (m2 / m1) ** (2 * gam),
    ) / (gam - 1)


def total_pressure_ratio_by_mach(
    mach_initial: ArraylikeFloat,
    mach_final: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat,
) -> NDArrayFloat:
    r"""Compute the total pressure ratio $p_{02} / p_{01}$ from the Mach number
    for Rayleigh flow.

    Args:
        mach_initial (ArraylikeFloat): Initial Mach number, $M_1$. This is
            the reference Mach number, $M^*$, when equal to unity.
        mach_final (ArraylikeFloat): Final Mach number, $M_2$. This is
            simply the Mach number, $M$, when ``mach_initial==1.0``
        specific_heat_ratio (ArraylikeFloat): ratio of specific heats,
            $\gamma$

    Returns:
        NDArrayFloat: total pressure ratio, $p_{02} / p_{01}$
            ($p_0 / p_0^*$ if $M_1=1.0$)

    """
    m1 = np.atleast_1d(mach_initial)
    m2 = np.atleast_1d(mach_final)
    gam = np.atleast_1d(specific_heat_ratio)
    return (
        pressure_ratio_by_mach(
            mach_initial=m1,
            mach_final=m2,
            specific_heat_ratio=gam,
        )
        * isentropic_flow.total_pressure_ratio_by_mach(
            mach=m2,
            specific_heat_ratio=gam,
        )
        / isentropic_flow.total_pressure_ratio_by_mach(
            mach=m1,
            specific_heat_ratio=gam,
        )
    )


def total_temperature_ratio_by_mach(
    mach_initial: ArraylikeFloat,
    mach_final: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat,
) -> NDArrayFloat:
    r"""Compute the total temperature ratio $T_{02} / T_{01}$ from the Mach
    number $M$ for Rayleigh flow.

    Args:
        mach_initial (ArraylikeFloat): Initial Mach number, $M_1$. This is
            the reference Mach number, $M^*$, when equal to unity.
        mach_final (ArraylikeFloat): Final Mach number, $M_2$. This is
            simply the Mach number, $M$, when ``mach_initial==1.0``
        specific_heat_ratio (ArraylikeFloat): ratio of specific heats,
            $\gamma$

    Returns:
        NDArrayFloat: total temperature ratio, $T_{02} / T_{01}$
            ($T_0 / T_0^*$ if $M_1=1.0$)

    """
    m1 = np.atleast_1d(mach_initial)
    m2 = np.atleast_1d(mach_final)
    gam = np.atleast_1d(specific_heat_ratio)
    return (
        temperature_ratio_by_mach(
            mach_initial=m1,
            mach_final=m2,
            specific_heat_ratio=gam,
        )
        * isentropic_flow.total_temperature_ratio_by_mach(
            mach=m2,
            specific_heat_ratio=gam,
        )
        / isentropic_flow.total_temperature_ratio_by_mach(
            mach=m1,
            specific_heat_ratio=gam,
        )
    )


def pressure_ratio_by_mach(
    mach_initial: ArraylikeFloat,
    mach_final: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat,
) -> NDArrayFloat:
    r"""Compute the static pressure ratio $p_2 / p_1$ from the Mach number $M$
    for Rayleigh flow.

    Args:
        mach_initial (ArraylikeFloat): Initial Mach number, $M_1$. This is
            the reference Mach number, $M^*$, when equal to unity.
        mach_final (ArraylikeFloat): Final Mach number, $M_2$. This is
            simply the Mach number, $M$, when ``mach_initial==1.0``
        specific_heat_ratio (ArraylikeFloat): ratio of specific heats,
            $\gamma$

    Returns:
        NDArrayFloat: static pressure ratio, $p_2 / p_1$
            ($p / p^*$ if $M_1=1.0$)

    """
    m1 = np.atleast_1d(mach_initial)
    m2 = np.atleast_1d(mach_final)
    gam = np.atleast_1d(specific_heat_ratio)
    return (1 + gam * m1**2) / (1 + gam * m2**2)


def temperature_ratio_by_mach(
    mach_initial: ArraylikeFloat,
    mach_final: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat,
) -> NDArrayFloat:
    r"""Compute the static temperature ratio $T_2 / T_1$ from the Mach number
    for Rayleigh flow.

    Args:
        mach_initial (ArraylikeFloat): Initial Mach number, $M_1$. This is
            the reference Mach number, $M^*$, when equal to unity.
        mach_final (ArraylikeFloat): Final Mach number, $M_2$. This is
            simply the Mach number, $M$, when ``mach_initial==1.0``
        specific_heat_ratio (ArraylikeFloat): ratio of specific heats,
            $\gamma$

    Returns:
        NDArrayFloat: static temperature ratio, $T_2 / T_1$
            ($T / T^*$ if $M_1=1.0$)

    """
    m1 = np.atleast_1d(mach_initial)
    m2 = np.atleast_1d(mach_final)
    gam = np.atleast_1d(specific_heat_ratio)
    return ((1 + gam * m1**2) / (1 + gam * m2**2)) ** 2 * (m2 / m1) ** 2


def density_ratio_by_mach(
    mach_initial: ArraylikeFloat,
    mach_final: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat,
) -> NDArrayFloat:
    r"""Compute the density ratio $\rho_2 / \rho_1$ from the Mach number
    for Rayleigh flow.

    Args:
        mach_initial (ArraylikeFloat): Initial Mach number, $M_1$. This is
            the reference Mach number, $M^*$, when equal to unity.
        mach_final (ArraylikeFloat): Final Mach number, $M_2$. This is
            simply the Mach number, $M$, when ``mach_initial==1.0``
        specific_heat_ratio (ArraylikeFloat): ratio of specific heats,
            $\gamma$

    Returns:
        NDArrayFloat: density ratio, $\rho_2 / \rho_1$
            ($\rho / \rho^*$ if $M_1=1.0$)

    """
    m1 = np.atleast_1d(mach_initial)
    m2 = np.atleast_1d(mach_final)
    gam = np.atleast_1d(specific_heat_ratio)
    return (1 + gam * m2**2) / (1 + gam * m1**2) * (m1 / m2) ** 2
