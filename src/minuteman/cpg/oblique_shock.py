# Copyright (c) 2022-2026 Jared Magnusson
# SPDX-License-Identifier: Apache-2.0

r"""```python
 import minuteman.cpg.oblique_shock as oblique_shock
```

This module computes flow parameters of a 2D, stationary, calorically perfect
oblique shocks.
"""

from dataclasses import dataclass
from enum import IntEnum, auto
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from minuteman.cpg import isentropic_flow, normal_shock
from minuteman.utils.bounds_check import (
    OutOfBoundsError,
    check_mach_supersonic,
    check_specific_heat_ratio,
)
from minuteman.utils.types import (
    ArraylikeFloat,
    NDArrayFloat,
    broadcast_inputs,
)


@dataclass
class ObliqueShockTable:
    """Oblique shock table for a calorically perfect gas"""

    mach_upstream: NDArrayFloat
    r"""Upstream mach number, $M_1$"""

    mach_downstream: NDArrayFloat
    r"""Downstream mach number, $M_2$"""

    mach_upstream_normal: NDArrayFloat
    r"""Normal component of upstream Mach number, $M_{n1}$"""

    mach_downstream_normal: NDArrayFloat
    r"""Normal component of downstream Mach number, $M_{n2}$"""

    deflection_angle: NDArrayFloat
    r"""Deflection angle, $\theta$"""

    shock_angle: NDArrayFloat
    r"""Shock angle, $\beta$"""

    temperature_ratio: NDArrayFloat
    r"""Temperature ratio, $T_2 / T_1$"""

    pressure_ratio: NDArrayFloat
    r"""Static pressure ratio, $p_2 / p_1$"""

    density_ratio: NDArrayFloat
    r"""Density ratio, $\rho_2 / \rho_1$"""

    total_pressure_ratio: NDArrayFloat
    r"""Total pressure ratio, $p_{02} / p_{01}$"""

    specific_heat_ratio: NDArrayFloat
    r"""Ratio of specific heats, $\gamma$"""


class ObliqueShockType(IntEnum):
    """Oblique shock type (weak or strong shock)"""

    strong = 0
    """Strong shock"""
    weak = auto()
    """Weak shock (most common in nature)"""


ArraylikeObliqueShockType: TypeAlias = (
    npt.NDArray[np.object_] | list[ObliqueShockType] | ObliqueShockType
)
"""Arraylike of ObliqueShockType objects"""


def lookup_table_by_deflection_angle(
    deflection_angle: ArraylikeFloat,
    mach_upstream: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat = 1.4,
    shock_type: ArraylikeObliqueShockType = ObliqueShockType.weak,
) -> ObliqueShockTable:
    r"""Look up the oblique shock properties from a known flow deflection
    angle $\theta$

    Args:
        deflection_angle: deflection angle, $\theta$ [radians].
            Bounds: $(0, \theta_{max}]$
        mach_upstream: upstream Mach number $M_1$. Bounds: $(1, \infty)$
        specific_heat_ratio: ratio of specific heats, $\gamma$.
            Bounds: $(1, 1.67]$
        shock_type: shock type

    Returns:
        Oblique shock table

    Raises:
        OutOfBoundsError: invalid inputs
    """
    theta, m1, gam, st = broadcast_inputs(
        deflection_angle, mach_upstream, specific_heat_ratio, shock_type
    )

    check_specific_heat_ratio(gam)
    check_mach_supersonic(m1)
    check_deflection_angle(
        deflection_angle=theta, mach_upstream=m1, specific_heat_ratio=gam
    )

    beta = shock_angle_by_deflection_mach(
        deflection_angle=theta,
        mach_upstream=m1,
        specific_heat_ratio=gam,
        shock_type=st,
    )
    mn1 = mach_upstream_normal_component(mach_upstream=m1, shock_angle=beta)
    mn2 = mach_downstream_normal_component(
        mach_upstream_normal=mn1,
        specific_heat_ratio=gam,
    )
    m2 = mach_downstream_by_postshock(
        mach_downstream_normal=mn2,
        shock_angle=beta,
        deflection_angle=theta,
    )
    return ObliqueShockTable(
        mach_upstream=m1,
        mach_downstream=m2,
        mach_upstream_normal=mn1,
        mach_downstream_normal=mn2,
        deflection_angle=theta,
        shock_angle=beta,
        temperature_ratio=normal_shock.temperature_ratio_by_mach(
            mach_upstream=mn1,
            specific_heat_ratio=gam,
        ),
        pressure_ratio=normal_shock.pressure_ratio_by_mach(
            mach_upstream=mn1,
            specific_heat_ratio=gam,
        ),
        density_ratio=normal_shock.density_ratio_by_mach(
            mach_upstream=mn1,
            specific_heat_ratio=gam,
        ),
        total_pressure_ratio=normal_shock.total_pressure_ratio_by_mach(
            mach_upstream=mn1,
            specific_heat_ratio=gam,
        ),
        specific_heat_ratio=gam,
    )


def lookup_table_by_shock_angle(
    shock_angle: ArraylikeFloat,
    mach_upstream: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat = 1.4,
) -> ObliqueShockTable:
    r"""Look up the oblique shock properties from a known shock angle, $\beta$

    Args:
        shock_angle: shock angle, $\beta$ [radians].
            Bounds: $[\arcsin\left(\frac{1}{M1}\right), 90^\circ]$
        mach_upstream: upstream Mach number, $M_1$. Bounds: $(1, \infty)$
        specific_heat_ratio: ratio of specific heats, $\gamma$.
            Bounds: $(1, 1.67]$

    Returns:
        Oblique shock table

    Raises:
        OutOfBoundsError: invalid inputs
    """
    beta, m1, gam = broadcast_inputs(
        shock_angle, mach_upstream, specific_heat_ratio
    )

    check_specific_heat_ratio(gam)
    check_mach_supersonic(m1)
    check_shock_angle(beta, m1)

    theta = deflection_angle_by_shock_mach(
        shock_angle=beta,
        mach_upstream=m1,
        specific_heat_ratio=gam,
    )
    mn1 = mach_upstream_normal_component(mach_upstream=m1, shock_angle=beta)
    mn2 = mach_downstream_normal_component(
        mach_upstream_normal=mn1,
        specific_heat_ratio=gam,
    )
    m2 = mach_downstream_by_postshock(
        mach_downstream_normal=mn2,
        shock_angle=beta,
        deflection_angle=theta,
    )
    return ObliqueShockTable(
        mach_upstream=m1,
        mach_downstream=m2,
        mach_upstream_normal=mn1,
        mach_downstream_normal=mn2,
        deflection_angle=theta,
        shock_angle=beta,
        temperature_ratio=normal_shock.temperature_ratio_by_mach(
            mach_upstream=mn1,
            specific_heat_ratio=gam,
        ),
        pressure_ratio=normal_shock.pressure_ratio_by_mach(
            mach_upstream=mn1,
            specific_heat_ratio=gam,
        ),
        density_ratio=normal_shock.density_ratio_by_mach(
            mach_upstream=mn1,
            specific_heat_ratio=gam,
        ),
        total_pressure_ratio=normal_shock.total_pressure_ratio_by_mach(
            mach_upstream=mn1,
            specific_heat_ratio=gam,
        ),
        specific_heat_ratio=gam,
    )


def lookup_table_by_mach_upstream_normal(
    mach_upstream_normal: ArraylikeFloat,
    mach_upstream: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat = 1.4,
) -> ObliqueShockTable:
    r"""Look up oblique shock table by the normal component of the upstream
    Mach number, $M_{n1}$

    Args:
        mach_upstream_normal: normal component of the upstream Mach number,
            $M_{n1}$. Bounds: $(1, M_1)$
        mach_upstream: upstream Mach number, $M_1$. Bounds: $(1, \infty)$
        specific_heat_ratio: ratio of specific heats, $\gamma$.
            Bounds: $(1, 1.67]$

    Returns:
        Oblique shock table

    Raises:
        OutOfBoundsError: invalid inputs
    """
    mn1, m1, gam = broadcast_inputs(
        mach_upstream_normal, mach_upstream, specific_heat_ratio
    )

    check_specific_heat_ratio(gam)
    check_mach_supersonic(m1)
    if np.any((mn1 <= 1.0) | (mn1 >= m1)):
        raise OutOfBoundsError(
            "Normal component of Mach Mn1 must be > 1.0 "
            "and < upstream Mach, M1"
        )

    mn2 = mach_downstream_normal_component(
        mach_upstream_normal=mn1,
        specific_heat_ratio=gam,
    )
    beta = np.asin(mn1 / m1)
    theta = deflection_angle_by_shock_mach(
        shock_angle=beta,
        mach_upstream=m1,
        specific_heat_ratio=gam,
    )
    m2 = mach_downstream_by_postshock(
        mach_downstream_normal=mn2,
        shock_angle=beta,
        deflection_angle=theta,
    )

    return ObliqueShockTable(
        mach_upstream=m1,
        mach_downstream=m2,
        mach_upstream_normal=mn1,
        mach_downstream_normal=mn2,
        deflection_angle=theta,
        shock_angle=beta,
        temperature_ratio=normal_shock.temperature_ratio_by_mach(
            mach_upstream=mn1,
            specific_heat_ratio=gam,
        ),
        pressure_ratio=normal_shock.pressure_ratio_by_mach(
            mach_upstream=mn1,
            specific_heat_ratio=gam,
        ),
        density_ratio=normal_shock.density_ratio_by_mach(
            mach_upstream=mn1,
            specific_heat_ratio=gam,
        ),
        total_pressure_ratio=normal_shock.total_pressure_ratio_by_mach(
            mach_upstream=mn1,
            specific_heat_ratio=gam,
        ),
        specific_heat_ratio=gam,
    )


def mach_upstream_normal_component(
    mach_upstream: ArraylikeFloat,
    shock_angle: ArraylikeFloat,
) -> NDArrayFloat:
    r"""Compute the normal component of the upstream Mach number, $M_{n1}$

    Args:
        mach_upstream: upstream Mach number, $M_1$
        shock_angle: shock angle, $\beta$ [radians]

    Returns:
        Normal component of the upstream Mach number, $M_{n1}$

    """
    m1 = np.atleast_1d(mach_upstream)
    beta = np.atleast_1d(shock_angle)
    check_shock_angle(shock_angle=beta, mach=m1)
    return m1 * np.sin(beta)


def mach_downstream_normal_component(
    mach_upstream_normal: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat,
) -> NDArrayFloat:
    r"""Compute the component of downstream Mach number normal to the shock,
    $M_{n2}$

    Args:
        mach_upstream_normal: component of upstream Mach number normal
            to the shock, $M_{n1}$
        specific_heat_ratio: ratio of specific heats, $\gamma$

    Returns:
        Component of downstream Mach number normal to the shock, $M_{n2}$

    """
    mn1 = np.atleast_1d(mach_upstream_normal)
    gam = np.atleast_1d(specific_heat_ratio)
    return np.sqrt(
        (mn1**2 + (2 / (gam - 1))) / (2 * gam / (gam - 1) * mn1**2 - 1),
    )


def mach_downstream_by_postshock(
    mach_downstream_normal: ArraylikeFloat,
    shock_angle: ArraylikeFloat,
    deflection_angle: ArraylikeFloat,
) -> NDArrayFloat:
    r"""Compute the downstream Mach number, $M_2$

    Args:
        mach_downstream_normal: component of downstream Mach number
            normal to the shock, $M_{n2}$
        shock_angle: shock angle, $\beta$ [radians]
        deflection_angle: flow deflection angle, $\theta$ [radians]

    Returns:
        Downstream Mach number, $M_2$

    Raises:
        OutOfBoundsError: Deflection angle must be smaller than shock angle

    """
    mn2 = np.atleast_1d(mach_downstream_normal)
    beta = np.atleast_1d(shock_angle)
    theta = np.atleast_1d(deflection_angle)
    if np.any(theta >= beta):
        raise OutOfBoundsError(
            "Deflection angle theta must be smaller than shock angle beta.",
        )
    return mn2 / np.sin(beta - theta)


def check_shock_angle(
    shock_angle: ArraylikeFloat,
    mach: ArraylikeFloat,
) -> None:
    r"""Ensure shock angle $\beta$ is within bounds or throw an error.

    Args:
        shock_angle: shock angle $\beta$ [radians]
        mach: Mach number $M$

    Raises:
        OutOfBoundsError: shock angle is out of bounds for given
            Mach number

    """
    beta = np.atleast_1d(shock_angle)
    mu = isentropic_flow.mach_angle(mach)
    if np.any((beta < mu) | (beta > 0.5 * np.pi)):
        raise OutOfBoundsError(
            f"Shock angle beta must be within [{np.degrees(mu)}, 90] deg",
        )


def check_deflection_angle(
    deflection_angle: ArraylikeFloat,
    mach_upstream: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat,
) -> None:
    r"""Ensure deflection angle $\theta$ is within bounds or throw an error.

    Args:
        deflection_angle: deflection angle $\theta$ [radians]
        mach_upstream: upstream Mach number, $M_1$
        specific_heat_ratio: ratio of specific heats, $\gamma$

    Raises:
        OutOfBoundsError: deflection angle is invalid

    """
    theta = np.atleast_1d(deflection_angle)
    m1 = np.atleast_1d(mach_upstream)
    gam = np.atleast_1d(specific_heat_ratio)
    theta_max = deflection_angle_max(mach_upstream=m1, specific_heat_ratio=gam)

    if np.any((theta <= 0.0) | (theta > theta_max)):
        raise OutOfBoundsError(
            "Deflection angle theta must be within "
            f"(0, {np.degrees(theta_max)}] deg",
        )


def deflection_angle_by_shock_mach(
    shock_angle: ArraylikeFloat,
    mach_upstream: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat,
) -> NDArrayFloat:
    r"""Compute the deflection angle $\theta$ for a given shock angle $\beta$
    and upstream Mach number $M_1$. This is the $\theta$-$\beta$-$M$ relation
    (Eq. 4.17 in [1](oblique_shock.md#references)).

    Args:
        shock_angle: shock angle $\beta$ [radians]
        mach_upstream: upstream Mach number $M_1$
        specific_heat_ratio: ratio of specific heats, $\gamma$

    Returns:
        Flow deflection angle, $\theta$ [radians]

    """
    beta = np.atleast_1d(shock_angle)
    m1 = np.atleast_1d(mach_upstream)
    gam = np.atleast_1d(specific_heat_ratio)
    return np.atan(
        2.0
        / np.tan(beta)
        * (
            ((m1 * np.sin(beta)) ** 2 - 1)
            / (m1**2 * (gam + np.cos(2 * beta)) + 2)
        ),
    )


def shock_angle_by_deflection_mach(
    deflection_angle: ArraylikeFloat,
    mach_upstream: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat,
    shock_type: ArraylikeObliqueShockType = ObliqueShockType.weak,
) -> NDArrayFloat:
    r"""Compute the oblique shock angle $\beta$ [radians] for a given
    deflection angle $\theta$ and upstream Mach number $M_1$.

    This is the lesser-known $\beta$-$\theta$-$M$ relation (Eq. 4.19-4.21 of
    [1](oblique_shock.md#references))

    Args:
        deflection_angle: flow deflection angle, $\theta$ [radians]
        mach_upstream: upstream Mach number, $M_1$
        specific_heat_ratio: ratio of specific heats, $\gamma$
        shock_type: Oblique shock type (weak or strong).

    Returns:
        Shock angle $\beta$ [radians]

    """
    theta = np.atleast_1d(deflection_angle)
    m1 = np.atleast_1d(mach_upstream)
    gam = np.atleast_1d(specific_heat_ratio)
    delta = np.atleast_1d(shock_type)

    # For valid solution for a given upstream Mach:
    # - `lam` must be real value
    # - `xi` must be within [-1, 1]
    # This means, at a minimum, `theta` > 0 and m1 > 1
    sqrt_crit = (m1**2 - 1) ** 2 - 3 * (1 + (gam - 1) / 2 * m1**2) * (
        1 + (gam + 1) / 2 * m1**2
    ) * (np.tan(theta)) ** 2
    lam = sqrt_crit**0.5

    xi = (
        (m1**2 - 1) ** 3
        - 9
        * (1 + (gam - 1) / 2 * m1**2)
        * (1 + (gam - 1) / 2 * m1**2 + (gam + 1) / 4 * m1**4)
        * (np.tan(theta)) ** 2
    ) / lam**3

    # for max deflection angle cases, round back to 1.0
    epsilon = 1e-9
    xi = np.where(np.abs(xi - 1.0) < epsilon, np.sign(xi) * 1.0, xi)

    return np.atan(
        (m1**2 - 1 + 2 * lam * np.cos((4 * np.pi * delta + np.acos(xi)) / 3))
        / (3 * np.tan(theta) * (1 + (gam - 1) / 2 * m1**2)),
    )


def shock_angle_max(
    mach_upstream: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat,
) -> NDArrayFloat:
    r"""Compute the shock angle $\beta_{max}$ [radians] that yields the max
    deflection angle $\theta_{max}$ for a given Mach number $M$ (slide 11 of
    [2](oblique_shock.md#references)).

    Args:
        mach_upstream: upstream Mach number, $M_1$
        specific_heat_ratio: ratio of specific heats, $\gamma$.

    Returns:
        Maximum shock angle $\beta_{max}$ [radians] that is still attached

    """
    # Computed by taking derivative of deflection angle wrt shock angle
    # (dtheta/dbeta) and setting it equal to zero, and taking the (-) root.
    m1 = np.atleast_1d(mach_upstream)
    gam = np.atleast_1d(specific_heat_ratio)
    return 0.5 * np.arccos(
        (
            m1**2 * (gam - 1)
            + 4
            - np.sqrt(
                16 * (gam + 1)
                + 8 * m1**2 * (gam**2 - 1)
                + m1**4 * (gam + 1) ** 2,
            )
        )
        / (2 * gam * m1**2),
    )


def deflection_angle_max(
    mach_upstream: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat,
) -> NDArrayFloat:
    r"""Compute the maximum flow deflection angle $\theta_{max}$ [radians]
    for a given upstream Mach number $M_1$.

    Args:
        mach_upstream: upstream Mach number, $M_1$
        specific_heat_ratio: ratio of specific heats, $\gamma$

    Returns:
        Max flow deflection angle, $\theta_{max}$ [radians]

    """
    m1 = np.atleast_1d(mach_upstream)
    gam = np.atleast_1d(specific_heat_ratio)
    beta = shock_angle_max(mach_upstream=m1, specific_heat_ratio=gam)
    return deflection_angle_by_shock_mach(
        shock_angle=beta,
        mach_upstream=m1,
        specific_heat_ratio=gam,
    )


def shock_angle_sonic(
    mach_upstream: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat,
) -> NDArrayFloat:
    r"""Compute the oblique shock angle $\beta$ [radians] which will yield a
    sonic downstream Mach number, $M_2=1$ (slide 29 of
    [2](oblique_shock.md#references)).

    Args:
        mach_upstream: upstream Mach number, $M_1$
        specific_heat_ratio: ratio of specific heats, $\gamma$

    Returns:
        Shock angle $\beta$ [radians] yielding sonic flow

    """
    m1 = np.atleast_1d(mach_upstream)
    gam = np.atleast_1d(specific_heat_ratio)
    return np.arcsin(
        np.sqrt(
            (
                (gam - 3) * m1**2
                + (gam + 1) * m1**4
                + np.sqrt(
                    16 * gam * m1**4
                    + ((gam - 3) * m1**2 + (gam + 1) * m1**4) ** 2,
                )
            )
            / (4 * gam * m1**4),
        ),
    )


def deflection_angle_sonic(
    mach_upstream: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat,
) -> NDArrayFloat:
    r"""Compute the flow deflection angle $\theta$ [radians] such that the
    downstream Mach number is sonic ($M_2=1$).

    Args:
        mach_upstream: upstream Mach number, $M_1$
        specific_heat_ratio: ratio of specific heats, $\gamma$

    Returns:
        Deflection angle $\theta$ [radians] yielding sonic flow

    """
    m1 = np.atleast_1d(mach_upstream)
    gam = np.atleast_1d(specific_heat_ratio)
    beta = shock_angle_sonic(mach_upstream=m1, specific_heat_ratio=gam)
    return deflection_angle_by_shock_mach(
        shock_angle=beta,
        mach_upstream=m1,
        specific_heat_ratio=gam,
    )
