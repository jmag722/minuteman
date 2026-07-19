r"""
This module computes flow parameters of 1D, stationary,
calorically perfect normal shocks ($\gamma$ is constant).

For a stationary, nonreacting normal shock, total enthalpy is constant
(adiabatic flow).
For perfect (calorically+thermally) gases, total temperature is thus also
constant across the shock.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import fsolve

import minuteman.cpg.isentropic_flow as isentropic_flow
from minuteman.utils.types import (
    ArrayOrScalarFloat,
    Floatlike,
    check_equal_shape,
    ndarray_f,
)


@dataclass
class NormalShockTable:
    """Normal shock table, for a stationary, calorically perfect gas"""

    mach_upstream: ndarray_f
    r"""Upstream mach number, $M_1$"""

    mach_downstream: ndarray_f
    r"""Downstream mach number, $M_2$"""

    temperature_ratio: ndarray_f
    r"""Temperature ratio, $T_2 / T_1$"""

    pressure_ratio: ndarray_f
    r"""Static pressure ratio, $p_2 / p_1$"""

    density_ratio: ndarray_f
    r"""Density ratio, $\rho_2 / \rho_1$"""

    total_pressure_ratio: ndarray_f
    r"""Total pressure ratio, $p_{02} / p_{01}$"""

    pitot_pressure_ratio: ndarray_f
    r"""Rayleigh Pitot tube pressure ratio, $p_{02} / p_1$"""

    specific_heat_ratio: ndarray_f
    r"""Ratio of specific heats, $\gamma$"""


def lookup_table_by_upstream_mach(
    mach_upstream: ArrayOrScalarFloat,
    specific_heat_ratio: ArrayOrScalarFloat = 1.4,
) -> NormalShockTable:
    r"""Look up a normal shock table result from the upstream Mach number,
    $M_1$

    Args:
        mach_upstream (ArrayOrScalarFloat): upstream Mach number, $M_1$
        specific_heat_ratio (ArrayOrScalarFloat, optional): Ratio of specific
            heats, $\gamma$. Defaults to 1.4.

    Returns:
        NormalShockTable: normal shock table result
    """
    m1 = np.atleast_1d(mach_upstream)
    gam = np.atleast_1d(specific_heat_ratio)
    p02_p01 = total_pressure_ratio_by_mach(
        mach_upstream=m1, specific_heat_ratio=gam
    )
    p01_p1 = isentropic_flow.total_pressure_ratio(
        mach=m1, specific_heat_ratio=gam
    )
    return NormalShockTable(
        mach_upstream=m1,
        mach_downstream=mach_downstream(
            mach_upstream=m1, specific_heat_ratio=gam
        ),
        temperature_ratio=temperature_ratio_by_upstream_mach(
            mach_upstream=m1, specific_heat_ratio=gam
        ),
        pressure_ratio=pressure_ratio(
            mach_upstream=m1, specific_heat_ratio=gam
        ),
        density_ratio=density_ratio(mach_upstream=m1, specific_heat_ratio=gam),
        total_pressure_ratio=p02_p01,
        pitot_pressure_ratio=p02_p01 * p01_p1,
        specific_heat_ratio=gam,
    )


def lookup_table_by_temperature(
    temperature_ratio: ArrayOrScalarFloat,
    specific_heat_ratio: ArrayOrScalarFloat = 1.4,
) -> NormalShockTable:
    r"""Look up a normal shock table result from the temperature ratio,
    $T_2 / T_1$

    Args:
        temperature_ratio (ArrayOrScalarFloat): temperature ratio, $T_2 / T_1$
        specific_heat_ratio (ArrayOrScalarFloat, optional): Ratio of specific
            heats, $\gamma$. Defaults to 1.4.

    Returns:
        NormalShockTable: normal shock table result
    """
    t21 = np.atleast_1d(temperature_ratio)
    if isinstance(specific_heat_ratio, Floatlike):
        gam = np.full_like(t21, specific_heat_ratio)
    else:
        gam = specific_heat_ratio
    check_equal_shape(t21.shape, gam.shape)

    # invert the temperature-mach relationship
    def tfunc(mguess: float, _t: float, _g: float):
        return _t - temperature_ratio_by_upstream_mach(
            mach_upstream=mguess, specific_heat_ratio=_g
        )

    m1 = np.array(
        [
            fsolve(tfunc, 2.0, args=(t, g))[0]
            for t, g in zip(t21.flat, gam.flat, strict=True)
        ]
    )
    return lookup_table_by_upstream_mach(
        mach_upstream=m1, specific_heat_ratio=gam
    )


def lookup_table_by_pressure(
    pressure_ratio: ArrayOrScalarFloat,
    specific_heat_ratio: ArrayOrScalarFloat = 1.4,
) -> NormalShockTable:
    r"""Look up a normal shock table result from the static pressure ratio,
    $p_2 / p_1$

    Args:
        pressure_ratio (ArrayOrScalarFloat): static pressure ratio, $p_2 / p_1$
        specific_heat_ratio (ArrayOrScalarFloat, optional): Ratio of specific
            heats, $\gamma$. Defaults to 1.4.

    Returns:
        NormalShockTable: normal shock table result
    """
    p21 = np.atleast_1d(pressure_ratio)
    gam = specific_heat_ratio
    # invert the pressure-mach relationship
    m1 = ((p21 - 1) * (gam + 1) / (2 * gam) + 1) ** 0.5
    return lookup_table_by_upstream_mach(
        mach_upstream=m1, specific_heat_ratio=gam
    )


def lookup_table_by_density(
    density_ratio: ArrayOrScalarFloat,
    specific_heat_ratio: ArrayOrScalarFloat = 1.4,
) -> NormalShockTable:
    r"""Look up a normal shock table result from the density ratio,
    $\rho_2 / \rho_1$

    Args:
        density_ratio (ArrayOrScalarFloat): density ratio, $\rho_2 / \rho_1$
        specific_heat_ratio (ArrayOrScalarFloat, optional): Ratio of specific
            heats, $\gamma$. Defaults to 1.4.

    Returns:
        NormalShockTable: normal shock table result
    """
    r21 = np.atleast_1d(density_ratio)
    gam = specific_heat_ratio
    # invert the density-mach relationship
    m1 = (2.0 / ((gam + 1) / r21 - gam + 1)) ** 0.5
    return lookup_table_by_upstream_mach(
        mach_upstream=m1, specific_heat_ratio=gam
    )


def lookup_table_by_total_pressure(
    total_pressure_ratio: ArrayOrScalarFloat,
    specific_heat_ratio: ArrayOrScalarFloat = 1.4,
) -> NormalShockTable:
    r"""Look up a normal shock table result from the total pressure ratio,
    $p_{02} / p_{01}$

    Args:
        total_pressure_ratio (ArrayOrScalarFloat): total pressure ratio,
            $p_{02} / p_{01}$
        specific_heat_ratio (ArrayOrScalarFloat, optional): Ratio of specific
            heats, $\gamma$. Defaults to 1.4.

    Returns:
        NormalShockTable: normal shock table result
    """
    p02_p01 = np.atleast_1d(total_pressure_ratio)
    if isinstance(specific_heat_ratio, Floatlike):
        gam = np.full_like(p02_p01, specific_heat_ratio)
    else:
        gam = specific_heat_ratio
    check_equal_shape(p02_p01.shape, gam.shape)

    # invert the total pressure-mach relationship
    def pfunc(mguess: float, _p021: float, _g: float):
        return _p021 - total_pressure_ratio_by_mach(
            mach_upstream=mguess, specific_heat_ratio=_g
        )

    m1 = np.array(
        [
            fsolve(pfunc, 2.0, args=(p, g))[0]
            for p, g in zip(p02_p01.flat, gam.flat, strict=True)
        ]
    )
    return lookup_table_by_upstream_mach(
        mach_upstream=m1, specific_heat_ratio=gam
    )


def lookup_table_by_pitot_pressure(
    pitot_pressure_ratio: ArrayOrScalarFloat,
    specific_heat_ratio: ArrayOrScalarFloat = 1.4,
) -> NormalShockTable:
    r"""Look up a normal shock table result from the Rayleigh Pitot tube
    pressure ratio, $p_{02} / p_1$

    Args:
        pitot_pressure_ratio (ArrayOrScalarFloat): Rayleigh Pitot tube
            pressure ratio, $p_{02} / p_1$
        specific_heat_ratio (ArrayOrScalarFloat, optional): Ratio of specific
            heats, $\gamma$. Defaults to 1.4.

    Returns:
        NormalShockTable: normal shock table result
    """
    p02_p1 = np.atleast_1d(pitot_pressure_ratio)
    if isinstance(specific_heat_ratio, Floatlike):
        gam = np.full_like(p02_p1, specific_heat_ratio)
    else:
        gam = specific_heat_ratio
    check_equal_shape(p02_p1.shape, gam.shape)

    # invert the rayleigh pitot pressure-mach relationship
    def pfunc(mguess: float, _p021: float, _g: float):
        return _p021 - total_pressure_ratio_by_mach(
            mach_upstream=mguess, specific_heat_ratio=_g
        ) * isentropic_flow.total_pressure_ratio(
            mach=mguess, specific_heat_ratio=_g
        )

    m1 = np.array(
        [
            fsolve(pfunc, 2.0, args=(p, g))[0]
            for p, g in zip(p02_p1.flat, gam.flat, strict=True)
        ]
    )
    return lookup_table_by_upstream_mach(
        mach_upstream=m1, specific_heat_ratio=gam
    )


def lookup_table_by_downstream_mach(
    mach_downstream: ArrayOrScalarFloat,
    specific_heat_ratio: ArrayOrScalarFloat = 1.4,
) -> NormalShockTable:
    r"""Look up a normal shock table result from the downstream Mach number,
    $M_2$

    Args:
        mach_downstream (ArrayOrScalarFloat): downstream Mach number, $M_2$
        specific_heat_ratio (ArrayOrScalarFloat, optional): Ratio of specific
            heats, $\gamma$. Defaults to 1.4.

    Returns:
        NormalShockTable: normal shock table result
    """
    m2 = np.atleast_1d(mach_downstream)
    gam = specific_heat_ratio

    # invert the mach relationship
    m1 = (
        (1 + m2**2 * 0.5 * (gam - 1)) / (gam * m2**2 - 0.5 * (gam - 1))
    ) ** 0.5
    return lookup_table_by_upstream_mach(
        mach_upstream=m1, specific_heat_ratio=gam
    )


def mach_downstream(
    mach_upstream: ArrayOrScalarFloat, specific_heat_ratio: ArrayOrScalarFloat
) -> ndarray_f:
    r"""Computes the Mach number downstream of a normal shock, $M_2$.

    Args:
        mach_upstream (ArrayOrScalarFloat): upstream Mach number, $M_1$
        specific_heat_ratio (ArrayOrScalarFloat): ratio of specific heats,
            $\gamma$

    Returns:
        ndarray_f: downstream Mach number, $M_2$
    """
    m1 = np.atleast_1d(mach_upstream)
    gam = specific_heat_ratio
    return (
        (1 + 0.5 * (gam - 1) * m1**2) / (gam * m1**2 - 0.5 * (gam - 1))
    ) ** 0.5


def density_ratio(
    mach_upstream: ArrayOrScalarFloat, specific_heat_ratio: ArrayOrScalarFloat
) -> ndarray_f:
    r"""Computes the density ratio across a normal shock, $\rho_2 / \rho_1$.

    The density ratio $\rho_2 / \rho_1$ is equivalent to the inverse velocity
    ratio $u_1 / u_2$

    Args:
        mach_upstream (ArrayOrScalarFloat): upstream Mach number, $M_1$
        specific_heat_ratio (ArrayOrScalarFloat): ratio of specific heats,
            $\gamma$

    Returns:
        ndarray_f: density ratio, $\rho_2 / \rho_1$
    """
    m1 = np.atleast_1d(mach_upstream)
    gam = specific_heat_ratio
    return (gam + 1) * m1**2 / (2 + (gam - 1) * m1**2)


def pressure_ratio(
    mach_upstream: ArrayOrScalarFloat, specific_heat_ratio: ArrayOrScalarFloat
) -> ndarray_f:
    r"""Computes the static pressure ratio across a normal shock, $p_2 / p_1$.

    Args:
        mach_upstream (ArrayOrScalarFloat): upstream Mach number, $M_1$
        specific_heat_ratio (ArrayOrScalarFloat): ratio of specific heats,
            $\gamma$

    Returns:
        ndarray_f: static pressure ratio, $p_2 / p_1$
    """
    m1 = np.atleast_1d(mach_upstream)
    gam = specific_heat_ratio
    return 1 + 2 * gam / (gam + 1) * (m1**2 - 1)


def total_pressure_ratio_by_mach(
    mach_upstream: ArrayOrScalarFloat, specific_heat_ratio: ArrayOrScalarFloat
) -> ndarray_f:
    r"""Computes the total pressure ratio across a normal shock,
    $p_{02} / p_{01}$.

    Args:
        mach_upstream (ArrayOrScalarFloat): upstream Mach number, $M_1$
        specific_heat_ratio (ArrayOrScalarFloat): ratio of specific heats,
            $\gamma$

    Returns:
        ndarray_f: total pressure ratio across shock, $p_{02} / p_{01}$
    """
    m1 = np.atleast_1d(mach_upstream)
    gam = specific_heat_ratio
    return ((gam + 1) / (2 * gam * m1**2 - gam + 1)) ** (1.0 / (gam - 1)) * (
        (gam + 1) * m1**2 / ((gam - 1) * m1**2 + 2)
    ) ** (gam / (gam - 1))


def temperature_ratio_by_upstream_mach(
    mach_upstream: ArrayOrScalarFloat, specific_heat_ratio: ArrayOrScalarFloat
) -> ndarray_f:
    r"""Computes the temperature ratio across a normal shock, $T_2 / T_1$.

    This is also equivalent to the enthalpy ratio across the normal shock,
    $h_2 / h_1$.

    Args:
        mach_upstream (ArrayOrScalarFloat): upstream Mach number, $M_1$
        specific_heat_ratio (ArrayOrScalarFloat): ratio of specific heats,
            $\gamma$

    Returns:
        ndarray_f: temperature ratio across shock, $T_2 / T_1$
    """
    m1 = np.atleast_1d(mach_upstream)
    gam = specific_heat_ratio
    return (
        (1 + 2 * gam / (gam + 1) * (m1**2 - 1))
        * (2 + (gam - 1) * m1**2)
        / ((gam + 1) * m1**2)
    )


def entropy_change(
    total_pressure_ratio: ArrayOrScalarFloat, gas_constant: ArrayOrScalarFloat
) -> ndarray_f:
    r"""Compute the change in specific entropy across a normal shock,
    $s_2 - s_1$

    Args:
        total_pressure_ratio (ArrayOrScalarFloat): total pressure ratio,
            $p_{02} / p_{01}$
        gas_constant (ArrayOrScalarFloat): specific gas constant, $R$

    Returns:
        ndarray_f: change in specific entropy, $s_2 - s_1$
    """
    p02_p01 = np.atleast_1d(total_pressure_ratio)
    return -gas_constant * np.log(p02_p01)


def internal_energy_change(
    pressure_upstream: ArrayOrScalarFloat,
    pressure_downstream: ArrayOrScalarFloat,
    density_upstream: ArrayOrScalarFloat,
    density_downstream: ArrayOrScalarFloat,
) -> ndarray_f:
    r"""Computes the change in specific internal energy across a normal shock,
    $e$. This is the Hugoniot relation.

    This relates only thermodynamic quantities across a normal shock.
    This relation is valid for perfect, chemically reacting, and real gases.

    Args:
        pressure_upstream (ArrayOrScalarFloat): upstream static pressure, $p_1$
        pressure_downstream (ArrayOrScalarFloat): downstream static pressure,
            $p_2$
        density_upstream (ArrayOrScalarFloat): upstream static density,
            $\rho_1$
        density_downstream (ArrayOrScalarFloat): downstream static density,
            $\rho_2$

    Returns:
        ndarray_f: change in specific internal energy, $e$
    """
    p1 = np.atleast_1d(pressure_upstream)
    # compute specific volumes
    v1 = 1.0 / density_upstream
    v2 = 1.0 / density_downstream
    return 0.5 * (p1 + pressure_downstream) * (v1 - v2)
