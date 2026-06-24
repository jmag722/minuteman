r"""
This module computes flow parameters of a 2D, stationary, calorically perfect 
oblique shocks.
"""

from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Annotated, TypeAlias

import numpy as np
import numpy.typing as npt

import minuteman.cpg.isentropic_flow as isentropic_flow
import minuteman.cpg.normal_shock as normal_shock
from minuteman.utils.types import (
    ArrayOrScalarFloat,
    ndarray_f,
)


@dataclass
class ObliqueShockTable:
    """Oblique shock table for a calorically perfect gas"""

    mach_upstream: ndarray_f
    r"""Upstream mach number, $M_1$"""

    mach_downstream: ndarray_f
    r"""Downstream mach number, $M_2$"""

    mach_upstream_normal: ndarray_f
    r"""Normal component of upstream Mach number, $M_{n1}$"""

    mach_downstream_normal: ndarray_f
    r"""Normal component of downstream Mach number, $M_{n2}$"""

    deflection_angle: ndarray_f
    r"""Deflection angle, $\theta$"""

    shock_angle: ndarray_f
    r"""Shock angle, $\beta$"""

    temperature_ratio: ndarray_f
    r"""Temperature ratio, $T_2 / T_1$"""

    pressure_ratio: ndarray_f
    r"""Static pressure ratio, $p_2 / p_1$"""

    density_ratio: ndarray_f
    r"""Density ratio, $\rho_2 / \rho_1$"""

    total_pressure_ratio: ndarray_f
    r"""Total pressure ratio, $p_{02} / p_{01}$"""

    specific_heat_ratio: ndarray_f
    r"""Ratio of specific heats, $\gamma$"""


class ObliqueShockType(IntEnum):
    """Oblique shock type (weak or strong shock)"""

    strong = 0
    """Strong shock"""
    weak = auto()
    """Weak shock (most common in nature)"""


ndarray_ObliqueShockType: TypeAlias = Annotated[
    npt.NDArray[np.object_], ObliqueShockType
]
"""Array of ObliqueShockType objects"""


def lookup_table_by_deflection_angle(
    deflection_angle: ArrayOrScalarFloat,
    mach_upstream: ArrayOrScalarFloat,
    specific_heat_ratio: ArrayOrScalarFloat = 1.4,
    shock_type: ObliqueShockType
    | ndarray_ObliqueShockType = ObliqueShockType.weak,
) -> ObliqueShockTable:
    r"""Look up the oblique shock properties from a known flow deflection
    angle $\theta$

    Args:
        deflection_angle (ArrayOrScalarFloat): deflection angle, $\theta$
            [radians]
        mach_upstream (ArrayOrScalarFloat): upstream Mach number $M_1$
        specific_heat_ratio (ArrayOrScalarFloat): ratio of specific heats,
            $\gamma$. Defaults to 1.4.
        shock_type (ObliqueShockType | ndarray_ObliqueShockType, optional):
            shock type. Defaults to ``ObliqueShockType.weak``.

    Returns:
        ObliqueShockTable: oblique shock table
    """
    theta = np.atleast_1d(deflection_angle)
    m1 = np.atleast_1d(mach_upstream)
    gam = np.atleast_1d(specific_heat_ratio)
    beta = shock_angle_by_deflection_mach(
        deflection_angle=theta,
        mach_upstream=m1,
        specific_heat_ratio=gam,
        shock_type=shock_type,
    )
    mn1 = mach_upstream_normal_component(mach_upstream=m1, shock_angle=beta)
    mn2 = mach_downstream_normal_component(
        mach_upstream_normal=mn1, specific_heat_ratio=gam
    )
    m2 = mach_downstream_by_postshock(
        mach_downstream_normal=mn2, shock_angle=beta, deflection_angle=theta
    )
    return ObliqueShockTable(
        mach_upstream=m1,
        mach_downstream=m2,
        mach_upstream_normal=mn1,
        mach_downstream_normal=mn2,
        deflection_angle=theta,
        shock_angle=beta,
        temperature_ratio=normal_shock.temperature_ratio_by_upstream_mach(
            mach_upstream=mn1, specific_heat_ratio=gam
        ),
        pressure_ratio=normal_shock.pressure_ratio(
            mach_upstream=mn1, specific_heat_ratio=gam
        ),
        density_ratio=normal_shock.density_ratio(
            mach_upstream=mn1, specific_heat_ratio=gam
        ),
        total_pressure_ratio=normal_shock.total_pressure_ratio_by_mach(
            mach_upstream=mn1, specific_heat_ratio=gam
        ),
        specific_heat_ratio=gam,
    )


def lookup_table_by_shock_angle(
    shock_angle: ArrayOrScalarFloat,
    mach_upstream: ArrayOrScalarFloat,
    specific_heat_ratio: ArrayOrScalarFloat = 1.4,
) -> ObliqueShockTable:
    r"""Look up the oblique shock properties from a known shock angle, $\beta$

    Args:
        shock_angle (ArrayOrScalarFloat): shock angle, $\beta$ [radians]
        mach_upstream (ArrayOrScalarFloat): upstream Mach number, $M_1$
        specific_heat_ratio (ArrayOrScalarFloat, optional): ratio of specific
            heats, $\gamma$. Defaults to 1.4.

    Returns:
        ObliqueShockTable: oblique shock table
    """
    beta = np.atleast_1d(shock_angle)
    m1 = np.atleast_1d(mach_upstream)
    gam = np.atleast_1d(specific_heat_ratio)
    theta = deflection_angle_by_shock_mach(
        shock_angle=beta, mach_upstream=m1, specific_heat_ratio=gam
    )
    mn1 = mach_upstream_normal_component(mach_upstream=m1, shock_angle=beta)
    mn2 = mach_downstream_normal_component(
        mach_upstream_normal=mn1, specific_heat_ratio=gam
    )
    m2 = mach_downstream_by_postshock(
        mach_downstream_normal=mn2, shock_angle=beta, deflection_angle=theta
    )
    return ObliqueShockTable(
        mach_upstream=m1,
        mach_downstream=m2,
        mach_upstream_normal=mn1,
        mach_downstream_normal=mn2,
        deflection_angle=theta,
        shock_angle=beta,
        temperature_ratio=normal_shock.temperature_ratio_by_upstream_mach(
            mach_upstream=mn1, specific_heat_ratio=gam
        ),
        pressure_ratio=normal_shock.pressure_ratio(
            mach_upstream=mn1, specific_heat_ratio=gam
        ),
        density_ratio=normal_shock.density_ratio(
            mach_upstream=mn1, specific_heat_ratio=gam
        ),
        total_pressure_ratio=normal_shock.total_pressure_ratio_by_mach(
            mach_upstream=mn1, specific_heat_ratio=gam
        ),
        specific_heat_ratio=gam,
    )


def lookup_table_by_mach_upstream_normal(
    mach_upstream_normal: ArrayOrScalarFloat,
    mach_upstream: ArrayOrScalarFloat,
    specific_heat_ratio: ArrayOrScalarFloat = 1.4,
) -> ObliqueShockTable:
    r"""Look up oblique shock table by the normal component of the upstream
    Mach number, $M_{n1}$

    Args:
        mach_upstream_normal (ArrayOrScalarFloat): normal component of the
            upstream Mach number, $M_{n1}$
        mach_upstream (ArrayOrScalarFloat): upstream Mach number, $M_1$
        specific_heat_ratio (ArrayOrScalarFloat, optional): ratio of specific
            heats, $\gamma$. Defaults to 1.4.

    Returns:
        ObliqueShockTable: oblique shock table
    """
    mn1 = np.atleast_1d(mach_upstream_normal)
    m1 = np.atleast_1d(mach_upstream)
    gam = np.atleast_1d(specific_heat_ratio)
    mn2 = mach_downstream_normal_component(
        mach_upstream_normal=mn1, specific_heat_ratio=gam
    )
    beta = np.asin(mn1 / m1)
    theta = deflection_angle_by_shock_mach(
        shock_angle=beta, mach_upstream=m1, specific_heat_ratio=gam
    )
    m2 = mach_downstream_by_postshock(
        mach_downstream_normal=mn2, shock_angle=beta, deflection_angle=theta
    )

    return ObliqueShockTable(
        mach_upstream=m1,
        mach_downstream=m2,
        mach_upstream_normal=mn1,
        mach_downstream_normal=mn2,
        deflection_angle=theta,
        shock_angle=beta,
        temperature_ratio=normal_shock.temperature_ratio_by_upstream_mach(
            mach_upstream=mn1, specific_heat_ratio=gam
        ),
        pressure_ratio=normal_shock.pressure_ratio(
            mach_upstream=mn1, specific_heat_ratio=gam
        ),
        density_ratio=normal_shock.density_ratio(
            mach_upstream=mn1, specific_heat_ratio=gam
        ),
        total_pressure_ratio=normal_shock.total_pressure_ratio_by_mach(
            mach_upstream=mn1, specific_heat_ratio=gam
        ),
        specific_heat_ratio=gam,
    )


def mach_upstream_normal_component(
    mach_upstream: ArrayOrScalarFloat, shock_angle: ArrayOrScalarFloat
) -> ndarray_f:
    r"""Compute the normal component of the upstream Mach number, $M_{n1}$

    Args:
        mach_upstream (ArrayOrScalarFloat): upstream Mach number, $M_1$
        shock_angle (ArrayOrScalarFloat): shock angle, $\beta$ [radians]

    Returns:
        ndarray_f: normal component of the upstream Mach number, $M_{n1}$
    """
    m1 = np.atleast_1d(mach_upstream)
    beta = np.atleast_1d(shock_angle)
    check_shock_angle(shock_angle=beta, mach=m1)
    return m1 * np.sin(beta)


def mach_downstream_normal_component(
    mach_upstream_normal: ArrayOrScalarFloat,
    specific_heat_ratio: ArrayOrScalarFloat,
) -> ndarray_f:
    r"""Compute the component of downstream Mach number normal to the shock,
    $M_{n2}$

    Args:
        mach_upstream_normal (ArrayOrScalarFloat): component of upstream Mach
            number normal to the shock, $M_{n1}$
        specific_heat_ratio (ArrayOrScalarFloat): ratio of specific heats,
            $\gamma$

    Returns:
        ndarray_f: component of downstream Mach number normal to the shock,
            $M_{n2}$
    """
    mn1 = np.atleast_1d(mach_upstream_normal)
    gam = np.atleast_1d(specific_heat_ratio)
    return np.sqrt(
        (mn1**2 + (2 / (gam - 1))) / (2 * gam / (gam - 1) * mn1**2 - 1)
    )


def mach_downstream_by_postshock(
    mach_downstream_normal: ArrayOrScalarFloat,
    shock_angle: ArrayOrScalarFloat,
    deflection_angle: ArrayOrScalarFloat,
) -> ndarray_f:
    r"""Compute the downstream Mach number, $M_2$

    Args:
        mach_downstream_normal (ArrayOrScalarFloat): component of downstream
            Mach number normal to the shock, $M_{n2}$
        shock_angle (ArrayOrScalarFloat): shock angle, $\beta$ [radians]
        deflection_angle (ArrayOrScalarFloat): flow deflection angle, $\theta$
            [radians]

    Raises:
        InvalidDeflectionAngle: Deflection angle must be smaller than shock
            angle

    Returns:
        ndarray_f: downstream Mach number, $M_{n2}$
    """
    mn2 = np.atleast_1d(mach_downstream_normal)
    beta = np.atleast_1d(shock_angle)
    theta = np.atleast_1d(deflection_angle)
    if np.any(theta >= beta):
        raise InvalidDeflectionAngle(
            "Deflection angle must be smaller than shock angle."
        )
    return mn2 / np.sin(beta - theta)


class InvalidShockAngle(Exception):
    r"""Shock angle $\beta$ is invalid"""

    pass


def check_shock_angle(
    shock_angle: ArrayOrScalarFloat, mach: ArrayOrScalarFloat
) -> None:
    r"""Ensure shock angle $\beta$ is within bounds or throw an error.

    Args:
        shock_angle (ArrayOrScalarFloat): shock angle $\beta$ [radians]
        mach (ArrayOrScalarFloat): Mach number $M$

    Raises:
        InvalidShockAngle: shock angle is out of bounds for given Mach number
    """
    mu = isentropic_flow.mach_angle(mach)
    valid_beta = np.all((shock_angle >= mu) & (shock_angle <= 0.5 * np.pi))
    if not valid_beta:
        raise InvalidShockAngle(f"Must be within [{np.degrees(mu)}, 90] deg")


class InvalidDeflectionAngle(Exception):
    r"""Deflection angle $\theta$ is invalid"""

    pass


def check_deflection_angle(
    deflection_angle: ArrayOrScalarFloat,
    mach_upstream: ArrayOrScalarFloat,
    specific_heat_ratio: ArrayOrScalarFloat,
) -> None:
    r"""Ensure deflection angle $\theta$ is within bounds or throw an error.

    Args:
        deflection_angle (ArrayOrScalarFloat): deflection angle $\theta$
            [radians]
        mach_upstream (ArrayOrScalarFloat): upstream Mach number, $M_1$
        specific_heat_ratio (ArrayOrScalarFloat, optional): ratio of specific
            heats, $\gamma$

    Raises:
        InvalidDeflectionAngle: deflection angle is invalid
    """
    theta = np.atleast_1d(deflection_angle)
    m1 = np.atleast_1d(mach_upstream)
    gam = np.atleast_1d(specific_heat_ratio)
    theta_max = deflection_angle_max(mach_upstream=m1, specific_heat_ratio=gam)
    valid_theta = np.all((theta >= 0.0) & (theta <= theta_max))
    if not valid_theta:
        raise InvalidDeflectionAngle(
            f"Must be within [0, {np.degrees(theta_max)}] deg"
        )


def deflection_angle_by_shock_mach(
    shock_angle: ArrayOrScalarFloat,
    mach_upstream: ArrayOrScalarFloat,
    specific_heat_ratio: ArrayOrScalarFloat,
) -> ndarray_f:
    r"""Compute the deflection angle $\theta$ for a given shock angle $\beta$
    and upstream Mach number $M_1$. This is the $\theta$-$\beta$-$M$ relation
    (Eq. 4.17 in [1](oblique_shock.md#references)).

    Args:
        shock_angle (ArrayOrScalarFloat): shock angle $\beta$ [radians]
        mach_upstream (ArrayOrScalarFloat): upstream Mach number $M_1$
        specific_heat_ratio (ArrayOrScalarFloat, optional): ratio of specific
            heats, $\gamma$

    Returns:
        ndarray_f: flow deflection angle, $\theta$ [radians]
    """
    beta = np.atleast_1d(shock_angle)
    m1 = np.atleast_1d(mach_upstream)
    gam = np.atleast_1d(specific_heat_ratio)
    check_shock_angle(shock_angle=beta, mach=m1)
    return np.atan(
        2.0
        / np.tan(beta)
        * (
            ((m1 * np.sin(beta)) ** 2 - 1)
            / (m1**2 * (gam + np.cos(2 * beta)) + 2)
        )
    )


def shock_angle_by_deflection_mach(
    deflection_angle: ArrayOrScalarFloat,
    mach_upstream: ArrayOrScalarFloat,
    specific_heat_ratio: ArrayOrScalarFloat,
    shock_type: ndarray_ObliqueShockType
    | ObliqueShockType = ObliqueShockType.weak,
) -> ndarray_f:
    r"""Compute the oblique shock angle $\beta$ [radians] for a given
    deflection angle $\theta$ and upstream Mach number $M_1$.

    This is the lesser-known $\beta$-$\theta$-$M$ relation (Eq. 4.19-4.21 of
    [1](oblique_shock.md#references))

    Args:
        deflection_angle (ArrayOrScalarFloat): flow deflection angle, $\theta$
            [radians]
        mach_upstream (ArrayOrScalarFloat): upstream Mach number, $M_1$
        specific_heat_ratio (ArrayOrScalarFloat): ratio of specific heats,
            $\gamma$
        shock_type (ndarray_ObliqueShockType | ObliqueShockType): Oblique shock
            type (weak or strong). Defaults to ``ObliqueShockType.weak``.

    Raises:
        InvalidDeflectionAngle: Deflection angle is invalid for the given
            upstream Mach.

    Returns:
        ndarray_f: shock angle $\beta$ [radians]
    """
    theta = np.atleast_1d(deflection_angle)
    m1 = np.atleast_1d(mach_upstream)
    gam = np.atleast_1d(specific_heat_ratio)
    check_deflection_angle(
        deflection_angle=theta, mach_upstream=m1, specific_heat_ratio=gam
    )
    delta = np.atleast_1d(shock_type)

    # for valid solution for a given upstream Mach, lam must be real value, and
    # xi must be within (-1, 1)
    sqrt_crit = (m1**2 - 1) ** 2 - 3 * (1 + (gam - 1) / 2 * m1**2) * (
        1 + (gam + 1) / 2 * m1**2
    ) * (np.tan(theta)) ** 2
    if sqrt_crit <= 0.0:
        raise InvalidDeflectionAngle()
    lam = sqrt_crit**0.5

    xi = (
        (m1**2 - 1) ** 3
        - 9
        * (1 + (gam - 1) / 2 * m1**2)
        * (1 + (gam - 1) / 2 * m1**2 + (gam + 1) / 4 * m1**4)
        * (np.tan(theta)) ** 2
    ) / lam**3
    if np.abs(xi) > 1.0:
        if np.abs(xi) - 1.0 < 1e-9:
            xi = np.sign(xi) * 1.0
        else:
            raise InvalidDeflectionAngle()

    return np.atan(
        (m1**2 - 1 + 2 * lam * np.cos((4 * np.pi * delta + np.acos(xi)) / 3))
        / (3 * np.tan(theta) * (1 + (gam - 1) / 2 * m1**2))
    )


def shock_angle_max(
    mach_upstream: ArrayOrScalarFloat, specific_heat_ratio: ArrayOrScalarFloat
) -> ndarray_f:
    r"""Compute the max shock angle $\beta_{max}$ [radians] that is still
    attached for a given Mach number $M$. This is achieved at the maximum flow
    deflection angle, $\theta_{max}$ (slide 11 of
    [2](oblique_shock.md#references)).

    Args:
        mach_upstream (ArrayOrScalarFloat): upstream Mach number, $M_1$
        specific_heat_ratio (ArrayOrScalarFloat): ratio of specific heats,
            $\gamma$.

    Returns:
        ndarray_f: maximum shock angle $\beta_{max}$ [radians] that is still
            attached
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
                + m1**4 * (gam + 1) ** 2
            )
        )
        / (2 * gam * m1**2)
    )


def deflection_angle_max(
    mach_upstream: ArrayOrScalarFloat, specific_heat_ratio: ArrayOrScalarFloat
) -> ndarray_f:
    r"""Compute the maximum flow deflection angle $\theta_{max}$ [radians]
    for a given upstream Mach number $M_1$.

    Args:
        mach_upstream (ArrayOrScalarFloat): upstream Mach number, $M_1$
        specific_heat_ratio (ArrayOrScalarFloat): ratio of specific heats,
            $\gamma$

    Returns:
        ndarray_f: max flow deflection angle, $\theta_{max}$ [radians]
    """
    m1 = np.atleast_1d(mach_upstream)
    gam = np.atleast_1d(specific_heat_ratio)
    beta = shock_angle_max(mach_upstream=m1, specific_heat_ratio=gam)
    return deflection_angle_by_shock_mach(
        shock_angle=beta, mach_upstream=m1, specific_heat_ratio=gam
    )


def shock_angle_sonic(
    mach_upstream: ArrayOrScalarFloat, specific_heat_ratio: ArrayOrScalarFloat
) -> ndarray_f:
    r"""Compute the oblique shock angle $\beta$ [radians] which will yield a
    sonic downstream Mach number, $M_2=1$ (slide 29 of
    [2](oblique_shock.md#references)).

    Args:
        mach_upstream (ArrayOrScalarFloat): upstream Mach number, $M_1$
        specific_heat_ratio (ArrayOrScalarFloat, optional): ratio of specific
            heats, $\gamma$

    Returns:
        ndarray_f: shock angle $\beta$ [radians] yielding sonic flow
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
                    + ((gam - 3) * m1**2 + (gam + 1) * m1**4) ** 2
                )
            )
            / (4 * gam * m1**4)
        )
    )


def deflection_angle_sonic(
    mach_upstream: ArrayOrScalarFloat, specific_heat_ratio: ArrayOrScalarFloat
) -> ndarray_f:
    r"""Compute the flow deflection angle $\theta$ [radians] such that the
    downstream Mach number is sonic ($M_2=1$).

    Args:
        mach_upstream (ArrayOrScalarFloat): upstream Mach number, $M_1$
        specific_heat_ratio (ArrayOrScalarFloat, optional): ratio of specific
            heats, $\gamma$

    Returns:
        ndarray_f: deflection angle $\theta$ [radians] yielding sonic flow
    """
    m1 = np.atleast_1d(mach_upstream)
    gam = np.atleast_1d(specific_heat_ratio)
    beta = shock_angle_sonic(mach_upstream=m1, specific_heat_ratio=gam)
    return deflection_angle_by_shock_mach(
        shock_angle=beta, mach_upstream=m1, specific_heat_ratio=gam
    )
