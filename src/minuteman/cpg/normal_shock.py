"""
This module computes flow parameters of 1D, stationary, calorically perfect normal shocks.

Enthalpy is constant across these shocks. For perfect (calorically+thermally) gases,
total temperature is also constant across the shock, and will not be output here.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import fsolve
import minuteman.cpg.isentropic_flow as isentropic_flow
from minuteman.utils.types import (
    ArrayOrScalarFloat,
    check_equal_shape,
    Floatlike,
    ndarray_f,
)


@dataclass
class NormalShockTable:
    """Normal shock table, for calorically perfect gas"""

    mach_upstream: ndarray_f
    """Upstream mach number, M1"""

    mach_downstream: ndarray_f
    """Downstream mach number, M2"""

    temperature_ratio: ndarray_f
    """Temperature ratio, T2/T1"""

    pressure_ratio: ndarray_f
    """Static pressure ratio, p2/p1"""

    density_ratio: ndarray_f
    """Density ratio, rho2/rho1"""

    total_pressure_ratio: ndarray_f
    """Total pressure ratio, p02/p01"""

    pitot_pressure_ratio: ndarray_f
    """Rayleigh Pitot tube pressure ratio, p02/p1"""

    specific_heat_ratio: ndarray_f
    """Ratio of specific heats, gamma"""


def lookup_table_by_upstream_mach(
    mach_upstream: ArrayOrScalarFloat,
    specific_heat_ratio: ArrayOrScalarFloat = 1.4,
) -> NormalShockTable:
    """Look up a normal shock table result from the upstream Mach number, M1

    Args:
        mach_upstream (ArrayOrScalarFloat): upstream Mach number, M1
        specific_heat_ratio (ArrayOrScalarFloat, optional): Ratio of specific
            heats, gamma. Defaults to 1.4.

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
    """Look up a normal shock table result from the temperature ratio, T2/T1

    Args:
        temperature_ratio (ArrayOrScalarFloat): temperature ratio, T2/T1
        specific_heat_ratio (ArrayOrScalarFloat, optional): Ratio of specific
            heats, gamma. Defaults to 1.4.

    Returns:
        NormalShockTable: normal shock table result
    """
    t21 = np.atleast_1d(temperature_ratio)
    if isinstance(specific_heat_ratio, Floatlike):
        gam = np.full_like(t21, specific_heat_ratio)
    else:
        gam = specific_heat_ratio
    check_equal_shape(t21.shape, gam.shape)

    # invert the T2/T1 -> M1 relationship
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
    """Look up a normal shock table result from the static pressure ratio, p2/p1

    Args:
        pressure_ratio (ArrayOrScalarFloat): static pressure ratio, p2/p1
        specific_heat_ratio (ArrayOrScalarFloat, optional): Ratio of specific
            heats, gamma. Defaults to 1.4.

    Returns:
        NormalShockTable: normal shock table result
    """
    p21 = np.atleast_1d(pressure_ratio)
    gam = specific_heat_ratio
    # invert the p2/p1 -> M1 relationship
    m1 = ((p21 - 1) * (gam + 1) / (2 * gam) + 1) ** 0.5
    return lookup_table_by_upstream_mach(
        mach_upstream=m1, specific_heat_ratio=gam
    )


def lookup_table_by_density(
    density_ratio: ArrayOrScalarFloat,
    specific_heat_ratio: ArrayOrScalarFloat = 1.4,
) -> NormalShockTable:
    """Look up a normal shock table result from the density ratio, rho2/rho1

    Args:
        density_ratio (ArrayOrScalarFloat): density ratio, rho2/rho1
        specific_heat_ratio (ArrayOrScalarFloat, optional): Ratio of specific
            heats, gamma. Defaults to 1.4.

    Returns:
        NormalShockTable: normal shock table result
    """
    r21 = np.atleast_1d(density_ratio)
    gam = specific_heat_ratio
    # invert the rho2/rho1 -> M1 relationship
    m1 = (2.0 / ((gam + 1) / r21 - gam + 1)) ** 0.5
    return lookup_table_by_upstream_mach(
        mach_upstream=m1, specific_heat_ratio=gam
    )


def lookup_table_by_total_pressure(
    total_pressure_ratio: ArrayOrScalarFloat,
    specific_heat_ratio: ArrayOrScalarFloat = 1.4,
) -> NormalShockTable:
    """Look up a normal shock table result from the total pressure ratio, p02/p01

    Args:
        total_pressure_ratio (ArrayOrScalarFloat): total pressure ratio, p02/p01
        specific_heat_ratio (ArrayOrScalarFloat, optional): Ratio of specific
            heats, gamma. Defaults to 1.4.

    Returns:
        NormalShockTable: normal shock table result
    """
    p02_p01 = np.atleast_1d(total_pressure_ratio)
    if isinstance(specific_heat_ratio, Floatlike):
        gam = np.full_like(p02_p01, specific_heat_ratio)
    else:
        gam = specific_heat_ratio
    check_equal_shape(p02_p01.shape, gam.shape)

    # invert the p02/p01 -> M1 relationship
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
    """Look up a normal shock table result from the Rayleigh Pitot tube pressure ratio, p02/p1

    Args:
        total_pressure_ratio (ArrayOrScalarFloat): Rayleigh Pitot tube pressure ratio, p02/p1
        specific_heat_ratio (ArrayOrScalarFloat, optional): Ratio of specific
            heats, gamma. Defaults to 1.4.

    Returns:
        NormalShockTable: normal shock table result
    """
    p02_p1 = np.atleast_1d(pitot_pressure_ratio)
    if isinstance(specific_heat_ratio, Floatlike):
        gam = np.full_like(p02_p1, specific_heat_ratio)
    else:
        gam = specific_heat_ratio
    check_equal_shape(p02_p1.shape, gam.shape)

    # invert the p02/p1 -> M1 relationship
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
    """Look up a normal shock table result from the downstream Mach number, M2

    Args:
        mach_downstream (ArrayOrScalarFloat): downstream Mach number, M2
        specific_heat_ratio (ArrayOrScalarFloat, optional): Ratio of specific
            heats, gamma. Defaults to 1.4.

    Returns:
        NormalShockTable: normal shock table result
    """
    m2 = np.atleast_1d(mach_downstream)
    gam = specific_heat_ratio

    # invert the M1 -> M2 relationship
    m1 = (
        (1 + m2**2 * 0.5 * (gam - 1)) / (gam * m2**2 - 0.5 * (gam - 1))
    ) ** 0.5
    return lookup_table_by_upstream_mach(
        mach_upstream=m1, specific_heat_ratio=gam
    )


def mach_downstream(
    mach_upstream: ArrayOrScalarFloat, specific_heat_ratio: ArrayOrScalarFloat
) -> ndarray_f:
    """Computes the Mach number downstream of a normal shock, M2.

    This form is valid for calorically perfect gases only, where `gamma`
    is constant (M1 < 5).

    Args:
        mach_upstream (ArrayOrScalarFloat): upstream Mach number, M1
        specific_heat_ratio (ArrayOrScalarFloat): ratio of specific heats, gamma

    Returns:
        ndarray_f: downstream Mach number, M2
    """
    m1 = np.atleast_1d(mach_upstream)
    gam = specific_heat_ratio
    return (
        (1 + 0.5 * (gam - 1) * m1**2) / (gam * m1**2 - 0.5 * (gam - 1))
    ) ** 0.5


def density_ratio(
    mach_upstream: ArrayOrScalarFloat, specific_heat_ratio: ArrayOrScalarFloat
) -> ndarray_f:
    """Computes the density ratio across a normal shock, rho2/rho1.

    This form is valid for calorically perfect gases only, where `gamma`
    is constant (M1 < 5).

    The density ratio rho2/rho1 is equivalent to the inverse velocity ratio
    u1/u2

    Args:
        mach_upstream (ArrayOrScalarFloat): upstream Mach number, M1
        specific_heat_ratio (ArrayOrScalarFloat): ratio of specific heats, gamma

    Returns:
        ndarray_f: density ratio, rho2/rho1
    """
    m1 = np.atleast_1d(mach_upstream)
    gam = specific_heat_ratio
    return (gam + 1) * m1**2 / (2 + (gam - 1) * m1**2)


def pressure_ratio(
    mach_upstream: ArrayOrScalarFloat, specific_heat_ratio: ArrayOrScalarFloat
) -> ndarray_f:
    """Computes the static pressure ratio across a normal shock, p2/p1.

    This form is valid for calorically perfect gases only, where `gamma`
    is constant (M1 < 5).

    Args:
        mach_upstream (ArrayOrScalarFloat): upstream Mach number, M1
        specific_heat_ratio (ArrayOrScalarFloat): ratio of specific heats, gamma

    Returns:
        ndarray_f: static pressure ratio, p2/p1
    """
    m1 = np.atleast_1d(mach_upstream)
    gam = specific_heat_ratio
    return 1 + 2 * gam / (gam + 1) * (m1**2 - 1)


def total_pressure_ratio_by_mach(
    mach_upstream: ArrayOrScalarFloat, specific_heat_ratio: ArrayOrScalarFloat
) -> ndarray_f:
    """Computes the total pressure ratio across a normal shock, p02/p01.

    This form is valid for calorically perfect gases only, where `gamma`
    is constant (M1 < 5).

    Args:
        mach_upstream (ArrayOrScalarFloat): upstream Mach number, M1
        specific_heat_ratio (ArrayOrScalarFloat): ratio of specific heats, gamma

    Returns:
        ndarray_f: total pressure ratio across shock, p02/p01
    """
    m1 = np.atleast_1d(mach_upstream)
    gam = specific_heat_ratio
    return ((gam + 1) / (2 * gam * m1**2 - gam + 1)) ** (1.0 / (gam - 1)) * (
        (gam + 1) * m1**2 / ((gam - 1) * m1**2 + 2)
    ) ** (gam / (gam - 1))


def temperature_ratio_by_upstream_mach(
    mach_upstream: ArrayOrScalarFloat, specific_heat_ratio: ArrayOrScalarFloat
) -> ndarray_f:
    """Computes the temperature ratio across a normal shock, T2/T1.

    This form is valid for calorically perfect gases only, where `gamma`
    is constant (M1 < 5). This is also equivalent to the enthalpy ratio
    across the normal shock, h2/h1

    Args:
        mach_upstream (ArrayOrScalarFloat): upstream Mach number, M1
        specific_heat_ratio (ArrayOrScalarFloat): ratio of specific heats, gamma

    Returns:
        ndarray_f: temperature ratio across shock, T2/T1
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
    """Compute the change in specific entropy across a normal shock, s2-s1

    This form is valid for calorically and thermally perfect gases over
    a stationary shock.

    Args:
        total_pressure_ratio (ArrayOrScalarFloat): total pressure ratio, p02/p01
        gas_constant (ArrayOrScalarFloat): specific gas constant, R

    Returns:
        ndarray_f: change in specific entropy, s2-s1
    """
    p02_p01 = np.atleast_1d(total_pressure_ratio)
    return -gas_constant * np.log(p02_p01)


def internal_energy_change(
    pressure_upstream: ArrayOrScalarFloat,
    pressure_downstream: ArrayOrScalarFloat,
    density_upstream: ArrayOrScalarFloat,
    density_downstream: ArrayOrScalarFloat,
) -> ndarray_f:
    """Computes the change in specific internal energy across a normal shock.
    This is the Hugoniot relation.

    This relates only thermodynamic quantities across a normal shock.
    This relation is valid for perfect, chemically reacting, and real gases.

    Args:
        pressure_upstream (ArrayOrScalarFloat): upstream static pressure
        pressure_downstream (ArrayOrScalarFloat): downstream static pressure
        density_upstream (ArrayOrScalarFloat): upstream static density
        density_downstream (ArrayOrScalarFloat): downstream static density

    Returns:
        ndarray_f: change in specific internal energy
    """
    p1 = np.atleast_1d(pressure_upstream)
    # compute specific volumes
    v1 = 1.0 / density_upstream
    v2 = 1.0 / density_downstream
    return 0.5 * (p1 + pressure_downstream) * (v1 - v2)
