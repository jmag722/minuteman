# Copyright (c) 2022-2026 Jared Magnusson
# SPDX-License-Identifier: Apache-2.0

r"""```python
 import minuteman.cpg.fanno as fanno
```

This module computes 1D, calorically perfect gas with friction
(Fanno flow).
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize.elementwise import find_root

from minuteman.cpg import ArraylikeFlowSpeedRegime, FlowSpeedRegime
from minuteman.cpg.base import bracket_mach_from_flow_regime
from minuteman.utils.types import (
    ArraylikeFloat,
    NDArrayFloat,
    RootFindingError,
)


@dataclass
class FannoFlowTable:
    r"""Fanno flow table where the initial Mach number $M_1=M^*=1.0$"""

    mach: NDArrayFloat
    r"""Mach number, $M$"""

    temperature_ratio: NDArrayFloat
    r"""Temperature ratio, $T / T^*$"""

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

    fanno_parameter: NDArrayFloat
    r"""Fanno parameter, $4 f L^* / D$"""

    entropy_ratio: NDArrayFloat
    r"""Specific entropy ratio, $(s^* - s) / R$"""

    specific_heat_ratio: NDArrayFloat
    r"""Ratio of specific heats, $\gamma$"""


def lookup_table_by_mach(
    mach: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat = 1.4,
) -> FannoFlowTable:
    r"""Look up a Fanno flow table result from the Mach number, $M$

    Args:
        mach (ArraylikeFloat): Mach number, $M$
        specific_heat_ratio (ArraylikeFloat, optional): ratio of specific
            heats, $\gamma$. Defaults to 1.4.

    Returns:
        FannoFlowTable: Fanno flow output table

    """
    m1 = 1.0
    m2 = np.atleast_1d(mach)
    gam = np.atleast_1d(specific_heat_ratio)
    return FannoFlowTable(
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
        fanno_parameter=_rev_fanno_parameter_by_mach(
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
) -> FannoFlowTable:
    r"""Look up a Fanno flow table result from the static pressure ratio,
    $p / p^*$

    Args:
        pressure_ratio (ArraylikeFloat): static pressure ratio, $p / p^*$
        specific_heat_ratio (ArraylikeFloat, optional): ratio of specific
            heats, $\gamma$. Defaults to 1.4.

    Returns:
        FannoFlowTable: Fanno flow output table

    """
    m1 = 1.0
    pratio = np.atleast_1d(pressure_ratio)
    gam = np.atleast_1d(specific_heat_ratio)
    # expression obtained with wolfram alpha, real-only root kept
    a = 2.0 / (gam - 1)
    b = -1.0 / (gam - 1) * (m1 / pratio) ** 2 * (2 + (gam - 1) * m1**2)
    m2 = 2**-0.5 * ((a**2 - 4 * b) ** 0.5 - a) ** 0.5
    return lookup_table_by_mach(mach=m2, specific_heat_ratio=gam)


def lookup_table_by_temperature(
    temperature_ratio: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat = 1.4,
) -> FannoFlowTable:
    r"""Look up a Fanno flow table result from the static temperature ratio,
    $T / T^*$

    Args:
        temperature_ratio (ArraylikeFloat): static temperature ratio,
            $T / T^*$
        specific_heat_ratio (ArraylikeFloat, optional): ratio of specific
            heats, $\gamma$. Defaults to 1.4.

    Returns:
        FannoFlowTable: Fanno flow output table

    """
    m1 = 1.0
    tratio = np.atleast_1d(temperature_ratio)
    gam = np.atleast_1d(specific_heat_ratio)
    m2 = (((2 + (gam - 1) * m1**2) / tratio - 2) / (gam - 1)) ** 0.5
    return lookup_table_by_mach(mach=m2, specific_heat_ratio=gam)


def lookup_table_by_density(
    density_ratio: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat = 1.4,
) -> FannoFlowTable:
    r"""Look up a Fanno flow table result from the density ratio,
    $\rho / \rho^*$

    Args:
        density_ratio (ArraylikeFloat): density ratio, $\rho / \rho^*$
        specific_heat_ratio (ArraylikeFloat, optional): ratio of specific
            heats, $\gamma$. Defaults to 1.4.

    Returns:
        FannoFlowTable: Fanno flow output table

    """
    m1 = 1.0
    rratio = np.atleast_1d(density_ratio)
    gam = np.atleast_1d(specific_heat_ratio)
    a = (rratio / m1) ** 2 * (2 + (gam - 1) * m1**2)
    m2 = (2.0 / (a - (gam - 1))) ** 0.5
    return lookup_table_by_mach(mach=m2, specific_heat_ratio=gam)


def _lookup_table_by_ratio(
    ratio: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat,
    flow_regime: ArraylikeFlowSpeedRegime,
    mach_func: Callable,
) -> FannoFlowTable:
    r"""Lookup the Fanno flow table by a generic input ratio where the
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
        FannoFlowTable: Fanno flow table

    """
    _ratio = np.atleast_1d(ratio)
    gam = np.atleast_1d(specific_heat_ratio)

    mach_brackets = bracket_mach_from_flow_regime(flow_regime)

    def get_mach_by_ratio(_m, _r, _g):
        return _r - mach_func(
            mach_initial=1.0,
            mach_final=_m,
            specific_heat_ratio=_g,
        )

    res = find_root(get_mach_by_ratio, mach_brackets, args=(_ratio, gam))
    if not np.all(res.success):
        raise RootFindingError(f"find_root did not succeed: {res.status}")
    m2 = res.x
    return lookup_table_by_mach(mach=m2, specific_heat_ratio=gam)


def lookup_table_by_total_pressure(
    total_pressure_ratio: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat = 1.4,
    flow_regime: ArraylikeFlowSpeedRegime = FlowSpeedRegime.supersonic,
) -> FannoFlowTable:
    r"""Look up a Fanno flow table result from the total pressure ratio,
    $p_0 / p_0^*$

    Args:
        total_pressure_ratio (ArraylikeFloat): total pressure ratio,
            $p_0 / p_0^*$
        specific_heat_ratio (ArraylikeFloat, optional): ratio of specific
            heats, $\gamma$. Defaults to 1.4.
        flow_regime (ArraylikeFlowSpeedRegime, optional):
            flow speed regime (either supersonic or subsonic).
            Defaults to ``FlowSpeedRegime.supersonic``.

    Returns:
        FannoFlowTable: Fanno flow output table

    """
    return _lookup_table_by_ratio(
        ratio=total_pressure_ratio,
        specific_heat_ratio=specific_heat_ratio,
        flow_regime=flow_regime,
        mach_func=total_pressure_ratio_by_mach,
    )


def lookup_table_by_entropy(
    entropy_ratio: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat = 1.4,
    flow_regime: ArraylikeFlowSpeedRegime = FlowSpeedRegime.supersonic,
) -> FannoFlowTable:
    r"""Look up a Fanno flow table result from the entropy ratio,
    $(s* - s) / R$

    Args:
        entropy_ratio (ArraylikeFloat): entropy ratio, $(s* - s) / R$
        specific_heat_ratio (ArraylikeFloat, optional): ratio of specific
            heats, $\gamma$. Defaults to 1.4.
        flow_regime (ArraylikeFlowSpeedRegime, optional):
            flow speed regime (either supersonic or subsonic).
            Defaults to ``FlowSpeedRegime.supersonic``.

    Returns:
        FannoFlowTable: Fanno flow output table

    """
    return _lookup_table_by_ratio(
        ratio=entropy_ratio,
        specific_heat_ratio=specific_heat_ratio,
        flow_regime=flow_regime,
        mach_func=_rev_entropy_ratio_by_mach,
    )


def lookup_table_by_fanno_parameter(
    fanno_parameter: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat = 1.4,
    flow_regime: ArraylikeFlowSpeedRegime = FlowSpeedRegime.supersonic,
) -> FannoFlowTable:
    r"""Look up a Fanno flow table result from the Fanno parameter,
    $4 f L^* / D$

    Args:
        fanno_parameter (ArraylikeFloat): Fanno parameter,
            $4 f L^* / D$
        specific_heat_ratio (ArraylikeFloat, optional): ratio of specific
            heats, $\gamma$. Defaults to 1.4.
        flow_regime (ArraylikeFlowSpeedRegime, optional):
            flow speed regime (either supersonic or subsonic).
            Defaults to ``FlowSpeedRegime.supersonic``.

    Returns:
        FannoFlowTable: Fanno flow output table

    """
    return _lookup_table_by_ratio(
        ratio=fanno_parameter,
        specific_heat_ratio=specific_heat_ratio,
        flow_regime=flow_regime,
        mach_func=_rev_fanno_parameter_by_mach,
    )


def temperature_ratio_by_mach(
    mach_initial: ArraylikeFloat,
    mach_final: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat,
) -> NDArrayFloat:
    r"""Compute the static temperature ratio $T_2 / T_1$ from the Mach number
    for Fanno flow.

    Args:
        mach_initial (ArraylikeFloat): Initial Mach number, $M_1$. This is
            the reference Mach number, $M^*$, when equal to unity.
        mach_final (ArraylikeFloat): Final Mach number, $M_2$. This is
            simply the Mach number, $M$, when ``mach_initial==1.0``
        specific_heat_ratio (ArraylikeFloat): ratio of specific heats,
            $\gamma$

    Returns:
        NDArrayFloat: static temperature ratio, $T_2 / T_1$
            ($T / T^*$ if $M_2=1.0$)

    """
    m1 = np.atleast_1d(mach_initial)
    m2 = np.atleast_1d(mach_final)
    gam = np.atleast_1d(specific_heat_ratio)
    return (2 + (gam - 1) * m1**2) / (2 + (gam - 1) * m2**2)


def pressure_ratio_by_mach(
    mach_initial: ArraylikeFloat,
    mach_final: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat,
) -> NDArrayFloat:
    r"""Compute the static pressure ratio $p_2 / p_1$ from the Mach number $M$
    for Fanno flow.

    Args:
        mach_initial (ArraylikeFloat): Initial Mach number, $M_1$. This is
            the reference Mach number, $M^*$, when equal to unity.
        mach_final (ArraylikeFloat): Final Mach number, $M_2$. This is
            simply the Mach number, $M$, when ``mach_initial==1.0``
        specific_heat_ratio (ArraylikeFloat): ratio of specific heats,
            $\gamma$

    Returns:
        NDArrayFloat: static pressure ratio, $p_2 / p_1$
            ($p / p^*$ if $M_2=1.0$)

    """
    m1 = np.atleast_1d(mach_initial)
    m2 = np.atleast_1d(mach_final)
    gam = np.atleast_1d(specific_heat_ratio)
    return (
        m1
        / m2
        * temperature_ratio_by_mach(
            mach_initial=m1,
            mach_final=m2,
            specific_heat_ratio=gam,
        )
        ** 0.5
    )


def density_ratio_by_mach(
    mach_initial: ArraylikeFloat,
    mach_final: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat,
) -> NDArrayFloat:
    r"""Compute the density ratio $\rho_2 / \rho_1$ from the Mach number
    for Fanno flow.

    This is equivalent to velocity ratio $u_1 / u_2$

    Args:
        mach_initial (ArraylikeFloat): Initial Mach number, $M_1$. This is
            the reference Mach number, $M^*$, when equal to unity.
        mach_final (ArraylikeFloat): Final Mach number, $M_2$. This is
            simply the Mach number, $M$, when ``mach_initial==1.0``
        specific_heat_ratio (ArraylikeFloat): ratio of specific heats,
            $\gamma$

    Returns:
        NDArrayFloat: density ratio, $\rho_2 / \rho_1$
            ($\rho / \rho^*$ if $M_2=1.0$)

    """
    m1 = np.atleast_1d(mach_initial)
    m2 = np.atleast_1d(mach_final)
    gam = np.atleast_1d(specific_heat_ratio)
    return (
        m1
        / m2
        * temperature_ratio_by_mach(
            mach_initial=m1,
            mach_final=m2,
            specific_heat_ratio=gam,
        )
        ** -0.5
    )


def total_pressure_ratio_by_mach(
    mach_initial: ArraylikeFloat,
    mach_final: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat,
) -> NDArrayFloat:
    r"""Compute total pressure ratio $p_{02} / p_{01}$ for Fanno flow.

    Args:
        mach_initial (ArraylikeFloat): Initial Mach number, $M_1$. This is
            the reference Mach number, $M^*$, when equal to unity.
        mach_final (ArraylikeFloat): Final Mach number, $M_2$. This is
            simply the Mach number, $M$, when ``mach_initial==1.0``
        specific_heat_ratio (ArraylikeFloat): ratio of specific heats,
            $\gamma$

    Returns:
        NDArrayFloat: total pressure ratio, $p_{02} / p_{01}$
            ($p_0 / p_0^*$ if $M_2=1.0$)

    """
    m1 = np.atleast_1d(mach_initial)
    m2 = np.atleast_1d(mach_final)
    gam = np.atleast_1d(specific_heat_ratio)
    return (
        m1
        / m2
        * temperature_ratio_by_mach(
            mach_initial=m1,
            mach_final=m2,
            specific_heat_ratio=gam,
        )
        ** (-(gam + 1) / (2 * (gam - 1)))
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
    return -1.0 * entropy_ratio_by_mach(
        mach_final=m2,
        mach_initial=m1,
        specific_heat_ratio=gam,
    )


def entropy_ratio_by_mach(
    mach_initial: ArraylikeFloat,
    mach_final: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat,
) -> NDArrayFloat:
    r"""Compute specific entropy ratio $(s_2 - s_1) / R$ for Fanno flow.

    Args:
        mach_initial (ArraylikeFloat): Mach number at station 1, $M_1$
            (initial or upstream Mach)
        mach_final (ArraylikeFloat): Mach number at station 2, $M_2$
            (final or downstream Mach)
        specific_heat_ratio (ArraylikeFloat): ratio of specific heats,
            $\gamma$

    Returns:
        NDArrayFloat: specific entropy ratio, $(s_2 - s_1) / R$
            ($(s - s^*) / R$ if ``mach_initial==1.0``)

    """
    m1 = np.atleast_1d(mach_initial)
    m2 = np.atleast_1d(mach_final)
    gam = np.atleast_1d(specific_heat_ratio)
    return np.log(
        m2
        / m1
        * ((2 + (gam - 1) * m1**2) / (2 + (gam - 1) * m2**2))
        ** (0.5 * (gam + 1) / (gam - 1)),
    )


# class InvalidMachError(Exception):
#     pass


# def _check_valid_mach(mach_initial: NDArrayFloat,
#                       mach_final: NDArrayFloat):
#     valid_mach = (mach_initial > 0.0) & (mach_final > 0.0)
#     if not valid_mach.all():
#         raise InvalidMachError("Mach must be positive")

#     invalid_2nd_law = (((mach_initial < 1) & (mach_final < mach_initial)) |
#                        (mach_initial >= 1) & (mach_final > mach_initial))
#     if invalid_2nd_law.any():
#         raise InvalidMachError(
#             "Choice of Mach numbers violates 2nd law of thermodynamics"
#         )
#     choked_flow = (
#         ((mach_initial < 1) & (mach_final > 1)) | (
#             (mach_initial >= 1) & (mach_final < 1))
#     )
#     if choked_flow.any():
#         raise InvalidMachError("Flow is choked, a normal shock will develop")


def _rev_fanno_parameter_by_mach(
    mach_initial: ArraylikeFloat,
    mach_final: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat,
):
    return -1.0 * fanno_parameter_by_mach(
        mach_initial=mach_initial,
        mach_final=mach_final,
        specific_heat_ratio=specific_heat_ratio,
    )


def fanno_parameter_by_mach(
    mach_initial: ArraylikeFloat,
    mach_final: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat,
) -> NDArrayFloat:
    r"""Compute Fanno parameter, $4L f / D$ where $L=x_2 - x_1$

    Args:
        mach_initial (ArraylikeFloat): Initial Mach number, $M_1$. This is
            simply the Mach number, $M$, when ``mach_final==1.0``
        mach_final (ArraylikeFloat): Final Mach number, $M_2$. This is
            the reference Mach number, $M^*$, when equal to unity.
        specific_heat_ratio (ArraylikeFloat): ratio of specific heats,
            $\gamma$

    Returns:
        NDArrayFloat: Fanno parameter, $4L f / D$
            ($4 L^* f / D$ if $M_2=1.0$)

    """
    m1 = np.atleast_1d(mach_initial)
    m2 = np.atleast_1d(mach_final)
    gam = np.atleast_1d(specific_heat_ratio)

    def fanno_integrand(m, g):
        return -1.0 / (g * m**2) - (g + 1) / (2 * g) * np.log(
            m**2 / (1 + (g - 1) / 2 * m**2),
        )

    return fanno_integrand(m2, gam) - fanno_integrand(m1, gam)


def duct_length(
    fanno_parameter: ArraylikeFloat,
    diameter: ArraylikeFloat,
    friction_coeff: ArraylikeFloat = 0.005,
) -> NDArrayFloat:
    r"""Compute duct length $L$ for a given Fanno parameter $4 f L / D$.

    Args:
        fanno_parameter (ArraylikeFloat): Fanno flow parameter, $4 f L / D$
        diameter (ArraylikeFloat): duct diameter, $D$
        friction_coeff (ArraylikeFloat, optional): average friction
            coefficient $f$. Defaults to 0.005.
            The default holds for $Re > 1e5$ and surface roughness of $0.001 D$

    Returns:
        NDArrayFloat: duct length $L$ or $L^*$

    """
    fanparam = np.atleast_1d(fanno_parameter)
    d = np.atleast_1d(diameter)
    f = np.atleast_1d(friction_coeff)
    return 0.25 * fanparam * d / f
