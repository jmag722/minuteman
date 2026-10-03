# Copyright (c) 2022-2026 Jared Magnusson
# SPDX-License-Identifier: Apache-2.0

r"""```python
 import minuteman.eq.normal_shock as normal_shock
```

This module computes flow parameters of 1D, stationary,
calorically perfect normal shocks ($\gamma$ is constant).

For a stationary, nonreacting normal shock, total enthalpy is constant
(adiabatic flow).
For perfect (calorically+thermally) gases, total temperature is thus also
constant across the shock.
"""

from dataclasses import dataclass

import cantera as ct
import numpy as np
from scipy.optimize.elementwise import find_root

from minuteman.utils.bounds_check import (
    OutOfBoundsError,
)
from minuteman.utils.types import (
    ArraylikeFloat,
    InvalidArrayShapeError,
    NDArrayFloat,
    RootFindingError,
    broadcast_inputs,
)


@dataclass
class EqNormalShockSolution:
    """Normal shock solution for a stationary shock in chemical equilibrium."""

    states_upstream: ct.SolutionArray
    states_downstream: ct.SolutionArray

    velocity_upstream: NDArrayFloat
    r"""Upstream velocity, $u_1$"""

    velocity_downstream: NDArrayFloat
    r"""Downstream velocity, $u_2$"""

    @property
    def mach_upstream(self) -> NDArrayFloat:
        r"""Upstream mach number, $M_1$"""
        return self.velocity_upstream / self.states_upstream.sound_speed

    @property
    def mach_downstream(self) -> NDArrayFloat:
        r"""Downstream mach number, $M_2$"""
        return self.velocity_downstream / self.states_downstream.sound_speed

    @property
    def temperature_ratio(self) -> NDArrayFloat:
        r"""Temperature ratio, $T_2 / T_1$"""
        return self.states_downstream.T / self.states_upstream.T

    @property
    def pressure_ratio(self) -> NDArrayFloat:
        r"""Static pressure ratio, $p_2 / p_1$"""
        return self.states_downstream.P / self.states_upstream.P

    @property
    def density_ratio(self) -> NDArrayFloat:
        r"""Density ratio, $\rho_2 / \rho_1$"""
        return (
            self.states_downstream.density_mass
            / self.states_upstream.density_mass
        )

    @property
    def speed_of_sound_ratio(self) -> NDArrayFloat:
        r"""Speed of sound ratio, $\a_2 / \a_1$"""
        return (
            self.states_downstream.sound_speed
            / self.states_upstream.sound_speed
        )

    @property
    def specific_heat_ratio_downstream(self) -> NDArrayFloat:
        r"""Downstream ratio of specific heats, $\gamma_2$"""
        return self.states_downstream.cp_mass / self.states_downstream.cv_mass

    @property
    def specific_heat_ratio_upstream(self) -> NDArrayFloat:
        r"""Upstream ratio of specific heats, $\gamma_1$"""
        return self.states_upstream.cp_mass / self.states_upstream.cv_mass


def pressure_by_density(
    density_ratio: ArraylikeFloat,
    density_upstream: ArraylikeFloat,
    velocity_upstream: ArraylikeFloat,
    pressure_upstream: ArraylikeFloat,
) -> NDArrayFloat:
    r"""Compute downstream pressure $p_2$ from the density ratio.

    Args:
        density_ratio: Density ratio $\rho_1/\rho_2$
        density_upstream: Upstream density $\rho_1$
        velocity_upstream: Upstream velocity $u_1$
        pressure_upstream: Upstream pressure $p_1$

    Returns:
        Downstream pressure $p_2$

    """
    r12 = np.atleast_1d(density_ratio)
    r1 = np.atleast_1d(density_upstream)
    u1 = np.atleast_1d(velocity_upstream)
    p1 = np.atleast_1d(pressure_upstream)
    return p1 + r1 * u1**2 * (1.0 - r12)


def enthalpy_by_density(
    density_ratio: ArraylikeFloat,
    velocity_upstream: ArraylikeFloat,
    enthalpy_upstream: ArraylikeFloat,
) -> NDArrayFloat:
    r"""Compute downstream enthalpy $h_2$ from the density ratio.

    Args:
        density_ratio: Density ratio $\rho_1/\rho_2$
        velocity_upstream: Upstream velocity $u_1$
        enthalpy_upstream: Upstream enthalpy $1_1$

    Returns:
        Downstream enthalpy $h_2$

    """
    r12 = np.atleast_1d(density_ratio)
    u1 = np.atleast_1d(velocity_upstream)
    h1 = np.atleast_1d(enthalpy_upstream)
    return h1 + 0.5 * u1**2 * (1.0 - r12**2)


def _density_ratio_residual(r12, u1, _states1) -> NDArrayFloat:
    states1 = _states1.item()
    p2 = pressure_by_density(
        density_ratio=r12,
        density_upstream=states1.density_mass,
        velocity_upstream=u1,
        pressure_upstream=states1.P,
    )
    h2 = enthalpy_by_density(
        density_ratio=r12,
        velocity_upstream=u1,
        enthalpy_upstream=states1.enthalpy_mass,
    )
    gas = ct.Solution(states1.source)
    states2 = ct.SolutionArray(gas, shape=p2.shape)
    states2.HPX = h2, p2, states1.X
    states2.equilibrate("HP", max_iter=5000)
    return r12 - states1.density_mass / states2.density_mass


def lookup_table_by_pressure_temperature(
    pressure: ArraylikeFloat,
    temperature: ArraylikeFloat,
    velocity: ArraylikeFloat,
    mole_fraction: str = "N2: 0.79, O2: 0.21",
    mechanism: str = "airNASA9.yaml",
) -> EqNormalShockSolution:
    p1, t1, u1 = broadcast_inputs(pressure, temperature, velocity)

    if np.any(p1 <= 0.0):
        p_msg = "Upstream pressure p1 must be > 0"
        raise OutOfBoundsError(p_msg)
    if np.any(t1 <= 0.0):
        t_msg = "Upstream temperature T1 must be > 0"
        raise OutOfBoundsError(t_msg)

    gas1 = ct.Solution(mechanism)
    states1 = ct.SolutionArray(gas1, shape=p1.shape)
    states1.TPX = t1, p1, mole_fraction

    return lookup_table_by_states(velocity=u1, states1=states1)


def lookup_table_by_states(
    velocity: NDArrayFloat, states1: ct.SolutionArray
) -> EqNormalShockSolution:
    u1 = velocity
    if u1.shape != states1.shape:
        raise InvalidArrayShapeError

    if np.any(u1 / states1.sound_speed <= 1.0):
        u_msg = "Upstream Mach number M1 must be > 1"
        raise OutOfBoundsError(u_msg)

    # density ratio (rho1/rho2) limits for root-finding
    r12_min = 1e-9
    r12_max = 1.0 - r12_min
    res = find_root(
        _density_ratio_residual,
        (r12_min, r12_max),
        args=(u1, states1),
    )
    if not np.all(res.success):
        raise RootFindingError(f"find_root did not succeed: {res.status}")
    r12 = res.x

    p2 = pressure_by_density(
        density_ratio=r12,
        density_upstream=states1.density_mass,
        velocity_upstream=u1,
        pressure_upstream=states1.P,
    )
    h2 = enthalpy_by_density(
        density_ratio=r12,
        velocity_upstream=u1,
        enthalpy_upstream=states1.enthalpy_mass,
    )
    gas2 = ct.Solution(states1.source)
    states2 = ct.SolutionArray(gas2, shape=u1.shape)
    states2.HPX = h2, p2, states1.X
    states2.equilibrate("HP", max_iter=5000)

    u21 = r12
    u2 = u21 * u1

    return EqNormalShockSolution(
        states_upstream=states1,
        states_downstream=states2,
        velocity_upstream=u1,
        velocity_downstream=u2,
    )
