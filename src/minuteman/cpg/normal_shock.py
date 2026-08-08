# Copyright (c) 2022-2026 Jared Magnusson
# SPDX-License-Identifier: Apache-2.0

r"""```python
 import minuteman.cpg.normal_shock as normal_shock
```

This module computes flow parameters of 1D, stationary,
calorically perfect normal shocks ($\gamma$ is constant).

For a stationary, nonreacting normal shock, total enthalpy is constant
(adiabatic flow).
For perfect (calorically+thermally) gases, total temperature is thus also
constant across the shock.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize.elementwise import find_root

from minuteman.cpg import isentropic_flow
from minuteman.utils.bounds_check import (
    OutOfBoundsError,
    check_specific_heat_ratio,
)
from minuteman.utils.types import (
    ArraylikeFloat,
    NDArrayFloat,
    RootFindingError,
)


@dataclass
class NormalShockTable:
    """Normal shock table, for a stationary, calorically perfect gas"""

    mach_upstream: NDArrayFloat
    r"""Upstream mach number, $M_1$"""

    mach_downstream: NDArrayFloat
    r"""Downstream mach number, $M_2$"""

    temperature_ratio: NDArrayFloat
    r"""Temperature ratio, $T_2 / T_1$"""

    pressure_ratio: NDArrayFloat
    r"""Static pressure ratio, $p_2 / p_1$"""

    density_ratio: NDArrayFloat
    r"""Density ratio, $\rho_2 / \rho_1$"""

    total_pressure_ratio: NDArrayFloat
    r"""Total pressure ratio, $p_{02} / p_{01}$"""

    pitot_pressure_ratio: NDArrayFloat
    r"""Rayleigh Pitot tube pressure ratio, $p_{02} / p_1$"""

    specific_heat_ratio: NDArrayFloat
    r"""Ratio of specific heats, $\gamma$"""


def lookup_table_by_upstream_mach(
    mach_upstream: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat = 1.4,
) -> NormalShockTable:
    r"""Look up a normal shock table result from the upstream Mach number,
    $M_1$

    Args:
        mach_upstream (ArraylikeFloat): upstream Mach number, $M_1$
        specific_heat_ratio (ArraylikeFloat, optional): Ratio of specific
            heats, $\gamma$. Defaults to 1.4.

    Returns:
        NormalShockTable: normal shock table result

    """
    m1 = np.atleast_1d(mach_upstream)
    gam = np.atleast_1d(specific_heat_ratio)
    if np.any(m1 < 1.0):
        raise OutOfBoundsError("Upstream Mach number M1 must >= 1.0")
    check_specific_heat_ratio(gam)

    p02_p01 = total_pressure_ratio_by_mach(
        mach_upstream=m1,
        specific_heat_ratio=gam,
    )
    p01_p1 = isentropic_flow.total_pressure_ratio_by_mach(
        mach=m1,
        specific_heat_ratio=gam,
    )
    return NormalShockTable(
        mach_upstream=m1,
        mach_downstream=mach_downstream_by_mach(
            mach_upstream=m1,
            specific_heat_ratio=gam,
        ),
        temperature_ratio=temperature_ratio_by_mach(
            mach_upstream=m1,
            specific_heat_ratio=gam,
        ),
        pressure_ratio=pressure_ratio_by_mach(
            mach_upstream=m1,
            specific_heat_ratio=gam,
        ),
        density_ratio=density_ratio_by_mach(
            mach_upstream=m1, specific_heat_ratio=gam
        ),
        total_pressure_ratio=p02_p01,
        pitot_pressure_ratio=p02_p01 * p01_p1,
        specific_heat_ratio=gam,
    )


def lookup_table_by_temperature(
    temperature_ratio: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat = 1.4,
) -> NormalShockTable:
    r"""Look up a normal shock table result from the temperature ratio,
    $T_2 / T_1$

    Args:
        temperature_ratio (ArraylikeFloat): temperature ratio, $T_2 / T_1$
        specific_heat_ratio (ArraylikeFloat, optional): Ratio of specific
            heats, $\gamma$. Defaults to 1.4.

    Returns:
        NormalShockTable: normal shock table result

    """
    t21 = np.atleast_1d(temperature_ratio)
    gam = np.atleast_1d(specific_heat_ratio)
    if np.any(t21 < 1.0):
        raise OutOfBoundsError("Temperature ratio T2/T1 must be >= 1.0")
    check_specific_heat_ratio(gam)

    m1_bracket = (1.0, 1e10)

    # invert the temperature-mach relationship
    def tfunc(mguess, _t, _g):
        return _t - temperature_ratio_by_mach(
            mach_upstream=mguess,
            specific_heat_ratio=_g,
        )

    res = find_root(tfunc, m1_bracket, args=(t21, gam))
    if not np.all(res.success):
        raise RootFindingError(f"find_root did not succeed: {res.status}")
    m1 = res.x

    return lookup_table_by_upstream_mach(
        mach_upstream=m1,
        specific_heat_ratio=gam,
    )


def lookup_table_by_pressure(
    pressure_ratio: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat = 1.4,
) -> NormalShockTable:
    r"""Look up a normal shock table result from the static pressure ratio,
    $p_2 / p_1$

    Args:
        pressure_ratio (ArraylikeFloat): static pressure ratio, $p_2 / p_1$
        specific_heat_ratio (ArraylikeFloat, optional): Ratio of specific
            heats, $\gamma$. Defaults to 1.4.

    Returns:
        NormalShockTable: normal shock table result

    """
    p21 = np.atleast_1d(pressure_ratio)
    gam = np.atleast_1d(specific_heat_ratio)
    if np.any(p21 < 1.0):
        raise OutOfBoundsError("Pressure ratio p2/p1 must be >= 1.0")
    check_specific_heat_ratio(gam)

    # invert the pressure-mach relationship
    m1 = ((p21 - 1) * (gam + 1) / (2 * gam) + 1) ** 0.5
    return lookup_table_by_upstream_mach(
        mach_upstream=m1,
        specific_heat_ratio=gam,
    )


def lookup_table_by_density(
    density_ratio: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat = 1.4,
) -> NormalShockTable:
    r"""Look up a normal shock table result from the density ratio,
    $\rho_2 / \rho_1$

    Args:
        density_ratio (ArraylikeFloat): density ratio, $\rho_2 / \rho_1$
        specific_heat_ratio (ArraylikeFloat, optional): Ratio of specific
            heats, $\gamma$. Defaults to 1.4.

    Returns:
        NormalShockTable: normal shock table result

    """
    r21 = np.atleast_1d(density_ratio)
    gam = np.atleast_1d(specific_heat_ratio)
    rho_minf = (gam + 1.0) / (gam - 1.0)
    if np.any((r21 < 1.0) | (r21 > rho_minf)):
        raise OutOfBoundsError(
            f"Density ratio rho2/rho1 must be within [{1.0}, {rho_minf}]"
        )
    check_specific_heat_ratio(gam)

    # invert the density-mach relationship
    m1 = (2.0 / ((gam + 1) / r21 - gam + 1)) ** 0.5
    return lookup_table_by_upstream_mach(
        mach_upstream=m1,
        specific_heat_ratio=gam,
    )


def lookup_table_by_total_pressure(
    total_pressure_ratio: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat = 1.4,
) -> NormalShockTable:
    r"""Look up a normal shock table result from the total pressure ratio,
    $p_{02} / p_{01}$

    Args:
        total_pressure_ratio (ArraylikeFloat): total pressure ratio,
            $p_{02} / p_{01}$
        specific_heat_ratio (ArraylikeFloat, optional): Ratio of specific
            heats, $\gamma$. Defaults to 1.4.

    Returns:
        NormalShockTable: normal shock table result

    """
    p02_p01 = np.atleast_1d(total_pressure_ratio)
    gam = np.atleast_1d(specific_heat_ratio)
    if np.any((p02_p01 <= 0.0) | (p02_p01 > 1.0)):
        raise OutOfBoundsError(
            "Total pressure ratio p02/p01 must be within (0.0, 1.0]"
        )
    check_specific_heat_ratio(gam)

    m1_bracket = (1.0, 1e10)

    # invert the total pressure-mach relationship
    def pfunc(mguess, _p021, _g):
        return _p021 - total_pressure_ratio_by_mach(
            mach_upstream=mguess,
            specific_heat_ratio=_g,
        )

    res = find_root(pfunc, m1_bracket, args=(p02_p01, gam))
    if not np.all(res.success):
        raise RootFindingError(f"find_root did not succeed: {res.status}")
    m1 = res.x

    return lookup_table_by_upstream_mach(
        mach_upstream=m1,
        specific_heat_ratio=gam,
    )


def lookup_table_by_pitot_pressure(
    pitot_pressure_ratio: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat = 1.4,
) -> NormalShockTable:
    r"""Look up a normal shock table result from the Rayleigh Pitot tube
    pressure ratio, $p_{02} / p_1$

    Args:
        pitot_pressure_ratio (ArraylikeFloat): Rayleigh Pitot tube
            pressure ratio, $p_{02} / p_1$
        specific_heat_ratio (ArraylikeFloat, optional): Ratio of specific
            heats, $\gamma$. Defaults to 1.4.

    Returns:
        NormalShockTable: normal shock table result

    """
    p02_p1 = np.atleast_1d(pitot_pressure_ratio)
    gam = np.atleast_1d(specific_heat_ratio)
    p02_p1_min = pitot_pressure_by_mach(
        mach_upstream=1.0, specific_heat_ratio=gam
    )
    if np.any(p02_p1 < p02_p1_min):
        raise OutOfBoundsError(
            f"Rayleigh Pitot pressure ratio p02/p1 must be >= {p02_p1_min}"
        )
    check_specific_heat_ratio(gam)

    m1_bracket = (1.0, 1e10)

    # invert the rayleigh pitot pressure-mach relationship
    def pfunc(mguess, _p021, _g):
        return _p021 - pitot_pressure_by_mach(
            mach_upstream=mguess, specific_heat_ratio=_g
        )

    res = find_root(pfunc, m1_bracket, args=(p02_p1, gam))
    if not np.all(res.success):
        raise RootFindingError(f"find_root did not succeed: {res.status}")
    m1 = res.x

    return lookup_table_by_upstream_mach(
        mach_upstream=m1,
        specific_heat_ratio=gam,
    )


def lookup_table_by_downstream_mach(
    mach_downstream: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat = 1.4,
) -> NormalShockTable:
    r"""Look up a normal shock table result from the downstream Mach number,
    $M_2$

    Args:
        mach_downstream (ArraylikeFloat): downstream Mach number, $M_2$
        specific_heat_ratio (ArraylikeFloat, optional): Ratio of specific
            heats, $\gamma$. Defaults to 1.4.

    Returns:
        NormalShockTable: normal shock table result

    """
    m2 = np.atleast_1d(mach_downstream)
    gam = np.atleast_1d(specific_heat_ratio)
    m2_min = np.sqrt(0.5 * (gam - 1) / gam)
    if np.any((m2 > 1) | (m2 < m2_min)):
        raise OutOfBoundsError(
            f"Downstream Mach number M2 must be between [{m2_min}, 1.0]"
        )
    check_specific_heat_ratio(gam)

    # invert the mach relationship
    m1 = (
        (1 + m2**2 * 0.5 * (gam - 1)) / (gam * m2**2 - 0.5 * (gam - 1))
    ) ** 0.5
    return lookup_table_by_upstream_mach(
        mach_upstream=m1,
        specific_heat_ratio=gam,
    )


def mach_downstream_by_mach(
    mach_upstream: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat,
) -> NDArrayFloat:
    r"""Computes the Mach number downstream of a normal shock, $M_2$.

    Args:
        mach_upstream (ArraylikeFloat): upstream Mach number, $M_1$
        specific_heat_ratio (ArraylikeFloat): ratio of specific heats,
            $\gamma$

    Returns:
        NDArrayFloat: downstream Mach number, $M_2$

    """
    m1 = np.atleast_1d(mach_upstream)
    gam = np.atleast_1d(specific_heat_ratio)
    return (
        (1 + 0.5 * (gam - 1) * m1**2) / (gam * m1**2 - 0.5 * (gam - 1))
    ) ** 0.5


def density_ratio_by_mach(
    mach_upstream: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat,
) -> NDArrayFloat:
    r"""Computes the density ratio across a normal shock, $\rho_2 / \rho_1$.

    The density ratio $\rho_2 / \rho_1$ is equivalent to the inverse velocity
    ratio $u_1 / u_2$

    Args:
        mach_upstream (ArraylikeFloat): upstream Mach number, $M_1$
        specific_heat_ratio (ArraylikeFloat): ratio of specific heats,
            $\gamma$

    Returns:
        NDArrayFloat: density ratio, $\rho_2 / \rho_1$

    """
    m1 = np.atleast_1d(mach_upstream)
    gam = np.atleast_1d(specific_heat_ratio)
    return (gam + 1) * m1**2 / (2 + (gam - 1) * m1**2)


def pressure_ratio_by_mach(
    mach_upstream: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat,
) -> NDArrayFloat:
    r"""Computes the static pressure ratio across a normal shock, $p_2 / p_1$.

    Args:
        mach_upstream (ArraylikeFloat): upstream Mach number, $M_1$
        specific_heat_ratio (ArraylikeFloat): ratio of specific heats,
            $\gamma$

    Returns:
        NDArrayFloat: static pressure ratio, $p_2 / p_1$

    """
    m1 = np.atleast_1d(mach_upstream)
    gam = np.atleast_1d(specific_heat_ratio)
    return 1 + 2 * gam / (gam + 1) * (m1**2 - 1)


def total_pressure_ratio_by_mach(
    mach_upstream: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat,
) -> NDArrayFloat:
    r"""Computes the total pressure ratio across a normal shock,
    $p_{02} / p_{01}$.

    Args:
        mach_upstream (ArraylikeFloat): upstream Mach number, $M_1$
        specific_heat_ratio (ArraylikeFloat): ratio of specific heats,
            $\gamma$

    Returns:
        NDArrayFloat: total pressure ratio across shock, $p_{02} / p_{01}$

    """
    m1 = np.atleast_1d(mach_upstream)
    gam = np.atleast_1d(specific_heat_ratio)
    return ((gam + 1) / (2 * gam * m1**2 - gam + 1)) ** (1.0 / (gam - 1)) * (
        (gam + 1) * m1**2 / ((gam - 1) * m1**2 + 2)
    ) ** (gam / (gam - 1))


def pitot_pressure_by_mach(
    mach_upstream: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat,
) -> NDArrayFloat:
    r"""Computes the Rayleigh Pitot tube pressure ratio across a normal shock,
    $p_{02} / p_1$.

    Args:
        mach_upstream (ArraylikeFloat): upstream Mach number, $M_1$
        specific_heat_ratio (ArraylikeFloat): ratio of specific heats,
            $\gamma$

    Returns:
        NDArrayFloat: Rayleigh Pitot tube pressure ratio across shock,
            $p_{02} / p_1$

    """
    m1 = np.atleast_1d(mach_upstream)
    gam = np.atleast_1d(specific_heat_ratio)
    return total_pressure_ratio_by_mach(
        mach_upstream=m1, specific_heat_ratio=gam
    ) * isentropic_flow.total_pressure_ratio_by_mach(
        mach=m1, specific_heat_ratio=gam
    )


def temperature_ratio_by_mach(
    mach_upstream: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat,
) -> NDArrayFloat:
    r"""Computes the temperature ratio across a normal shock, $T_2 / T_1$.

    This is also equivalent to the enthalpy ratio across the normal shock,
    $h_2 / h_1$.

    Args:
        mach_upstream (ArraylikeFloat): upstream Mach number, $M_1$
        specific_heat_ratio (ArraylikeFloat): ratio of specific heats,
            $\gamma$

    Returns:
        NDArrayFloat: temperature ratio across shock, $T_2 / T_1$

    """
    m1 = np.atleast_1d(mach_upstream)
    gam = np.atleast_1d(specific_heat_ratio)
    return (
        (1 + 2 * gam / (gam + 1) * (m1**2 - 1))
        * (2 + (gam - 1) * m1**2)
        / ((gam + 1) * m1**2)
    )


def entropy_change(
    total_pressure_ratio: ArraylikeFloat,
    gas_constant: ArraylikeFloat,
) -> NDArrayFloat:
    r"""Compute the change in specific entropy across a normal shock,
    $s_2 - s_1$

    Args:
        total_pressure_ratio (ArraylikeFloat): total pressure ratio,
            $p_{02} / p_{01}$
        gas_constant (ArraylikeFloat): specific gas constant, $R$

    Returns:
        NDArrayFloat: change in specific entropy, $s_2 - s_1$

    """
    p02_p01 = np.atleast_1d(total_pressure_ratio)
    gc = np.atleast_1d(gas_constant)
    return -gc * np.log(p02_p01)


def internal_energy_change(
    pressure_upstream: ArraylikeFloat,
    pressure_downstream: ArraylikeFloat,
    density_upstream: ArraylikeFloat,
    density_downstream: ArraylikeFloat,
) -> NDArrayFloat:
    r"""Computes the change in specific internal energy across a normal shock,
    $e$. This is the Hugoniot relation.

    This relates only thermodynamic quantities across a normal shock.
    This relation is valid for perfect, chemically reacting, and real gases.

    Args:
        pressure_upstream (ArraylikeFloat): upstream static pressure, $p_1$
        pressure_downstream (ArraylikeFloat): downstream static pressure,
            $p_2$
        density_upstream (ArraylikeFloat): upstream static density,
            $\rho_1$
        density_downstream (ArraylikeFloat): downstream static density,
            $\rho_2$

    Returns:
        NDArrayFloat: change in specific internal energy, $e$

    """
    p1 = np.atleast_1d(pressure_upstream)
    # compute specific volumes
    v1 = 1.0 / np.atleast_1d(density_upstream)
    v2 = 1.0 / np.atleast_1d(density_downstream)
    p2 = np.atleast_1d(pressure_downstream)
    return 0.5 * (p1 + p2) * (v1 - v2)
