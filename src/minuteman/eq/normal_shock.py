# Copyright (c) 2022-2026 Jared Magnusson
# SPDX-License-Identifier: Apache-2.0

r"""```python
 import minuteman.eq.normal_shock as normal_shock
```

This module computes flow parameters of 1D, stationary,
equilibrium reaction normal shocks. The implementation follows that of
[1](normal_shock.md#references)

For a stationary normal shock, total enthalpy is constant (adiabatic flow).
"""

from dataclasses import dataclass

import cantera as ct
import numpy as np
import scipy.constants as sc
from scipy.optimize.elementwise import find_root

from minuteman.eq.thermo import isentropic_exponent, speed_of_sound
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
class EqFlowStates:
    """Equilibrium flow state."""

    states: ct.SolutionArray
    """Thermodynamic states"""

    velocity: NDArrayFloat
    """Flow velocity $u$ [m/s]"""

    speed_of_sound_eq: NDArrayFloat
    """Equilibrium speed of sound $a_{eq}$ [m/s]"""

    @property
    def speed_of_sound_fr(self) -> NDArrayFloat:
        r"""Frozen speed of sound $a_{f}$ [m/s]"""
        return self.states.sound_speed

    isentropic_exponent: NDArrayFloat
    r"""Isentropic exponent $\gamma_s$"""

    @property
    def pressure(self) -> NDArrayFloat:
        r"""Pressure, $p$ [Pa]"""
        return self.states.P

    @property
    def temperature(self) -> NDArrayFloat:
        r"""Temperature, $T$ [K]"""
        return self.states.T

    @property
    def density(self) -> NDArrayFloat:
        r"""Density, $\rho$ [kg/$m^{-3}$]"""
        return self.states.density_mass

    @property
    def mach(self) -> NDArrayFloat:
        r"""Mach number, $M$ (using equilibrium speed of sound)"""
        return self.velocity / self.speed_of_sound_eq

    def mass_fraction(self, name: str) -> NDArrayFloat:
        """Mass fraction $Y_i$ for species $i$.

        Args:
            name: Species name.

        Returns:
            Mass fraction $Y_i$.

        """
        idx = self.states.species_index(name)
        return self.states.Y[..., idx]

    def mole_fraction(self, name: str) -> NDArrayFloat:
        """Mole fraction $X_i$ for species $i$.

        Args:
            name: Species name.

        Returns:
            Mole fraction $X_i$.

        """
        idx = self.states.species_index(name)
        return self.states.X[..., idx]

    def number_density(self, name: str) -> NDArrayFloat:
        r"""Return number density $n_i$ for species $i$.

        Args:
            name: Species name

        Returns:
            Number density $n_i$ [particles/$m^{-3}$]

        """
        idx = self.states.species_index(name)
        return self.states.concentrations[..., idx] * sc.kilo * sc.Avogadro


@dataclass
class EqNormalShockSolution:
    """Normal shock solution for a stationary shock in chemical equilibrium."""

    upstream: EqFlowStates
    """Upstream flow properties."""

    downstream: EqFlowStates
    """Downstream flow properties."""

    @property
    def temperature_ratio(self) -> NDArrayFloat:
        r"""Temperature ratio, $T_2 / T_1$"""
        return self.downstream.temperature / self.upstream.temperature

    @property
    def pressure_ratio(self) -> NDArrayFloat:
        r"""Static pressure ratio, $p_2 / p_1$"""
        return self.downstream.pressure / self.upstream.pressure

    @property
    def density_ratio(self) -> NDArrayFloat:
        r"""Density ratio, $\rho_2 / \rho_1$"""
        return self.downstream.density / self.upstream.density

    @property
    def speed_of_sound_ratio(self) -> NDArrayFloat:
        r"""Speed of sound ratio, $a_2 / a_1$"""
        return (
            self.downstream.speed_of_sound_eq / self.upstream.speed_of_sound_eq
        )


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
        enthalpy_upstream: Upstream enthalpy $h_1$

    Returns:
        Downstream enthalpy $h_2$

    """
    r12 = np.atleast_1d(density_ratio)
    u1 = np.atleast_1d(velocity_upstream)
    h1 = np.atleast_1d(enthalpy_upstream)
    return h1 + 0.5 * u1**2 * (1.0 - r12**2)


def lookup_table_by_pressure_temperature(
    pressure: ArraylikeFloat,
    temperature: ArraylikeFloat,
    velocity: ArraylikeFloat,
    mole_fraction: str = "N2: 0.79, O2: 0.21",
    mechanism: str = "airNASA9.yaml",
) -> EqNormalShockSolution:
    r"""Lookup equilibrium normal shock table properties by upstream pressure
    $p_1$ and temperature $T_1$

    Args:
        pressure: Upstream pressure $p_1$. Bounds: (0, $\infty$)
        temperature: Upstream temperature $T_1$. Bounds: (0, $\infty$)
        velocity: Upstream velocity $u_1$. Bounds: (Mach 1, $\infty$)
        mole_fraction: Upstream mole fractions $X_1$.
        mechanism: Reaction mechanism file.

    Returns:
        Equilibrium normal shock table.

    Raises:
        OutOfBoundsError: Pressure or temperature are out of bounds.
    """
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

    return lookup_table_by_states(states1=states1, velocity=u1)


def lookup_table_by_states(
    states1: ct.SolutionArray, velocity: NDArrayFloat
) -> EqNormalShockSolution:
    r"""Lookup equilibrium normal shock table properties by preinitialized
    upstream thermodynamic states

    Args:
        states1: Upstream thermodynamic states.
        velocity: Upstream velocity $u_1$. Bounds: (Mach 1, $\infty$)

    Returns:
        Equilibrium normal shock table.

    Raises:
        OutOfBoundsError: Upstream velocity (Mach) is out of bounds
        RootFindingError: Root finder failed to converge.
    """
    u1 = velocity
    if u1.shape != states1.shape:
        raise InvalidArrayShapeError

    if np.any(u1 / states1.sound_speed <= 1.0):
        u_msg = "Upstream Mach number M1 must be > 1"
        raise OutOfBoundsError(u_msg)

    # density ratio (rho1/rho2) limits for root-finding
    r12_min = np.zeros_like(u1) + 1e-9
    r12_max = np.ones_like(u1) - r12_min

    p1 = states1.P
    r1 = states1.density_mass
    h1 = states1.enthalpy_mass
    x1 = states1.X

    # WHEN SCIPY 1.19 is out! Replace the monstrosity below, either by
    # passing preserve_shape=True with plane multidimensional mole fracs,
    # or by passing the entire upstream state directly if that works
    # with slicing SolutionArray

    # passing species mole fractions,as a multidimensional array of objects
    # holding 1D arrays, instead of as a multidimensional arrays,
    # to keep it from having shape issues with find_root
    @dataclass
    class Xobj:
        xarr: NDArrayFloat

    xnew = np.full(u1.shape, fill_value=None, dtype=Xobj)
    # x1 will always have one more dimension than u1
    for idx in np.ndindex(u1.shape):
        xnew[idx] = Xobj(xarr=x1[idx])

    def _density_ratio_residual(_r12, _r1, _u1, _p1, _h1, _x1):
        # can't pass states1 because it won't get sliced right.
        # states1.X needs to be broadcastable because it's used with h2, p2
        # x1 can't be multidimensional because currently it gets flattened
        # after the first 2 iterations. so it needs to have the same shape
        # as u1/p1/h1 essentially. We're cheating here by putting its last
        # dimension in an object to hide it
        _p2 = pressure_by_density(
            density_ratio=_r12,
            density_upstream=_r1,
            velocity_upstream=_u1,
            pressure_upstream=_p1,
        )
        _h2 = enthalpy_by_density(
            density_ratio=_r12,
            velocity_upstream=_u1,
            enthalpy_upstream=_h1,
        )
        gas = ct.Solution(states1.source)
        st2 = ct.SolutionArray(gas, shape=_p2.shape)

        # undo the class, get back a multidimensional array to pass to cantera
        # must query shape every step because find_root
        # drops converged indices!
        xa = np.zeros_like(_p2)
        newshape = list(_p2.shape)
        newshape.append(_x1[..., 0].flatten()[0].xarr.size)
        xa = np.zeros(newshape)
        for idx in np.ndindex(_p2.shape):
            xa[idx, :] = _x1[idx].xarr

        st2.HPX = _h2, _p2, xa
        st2.equilibrate("HP", max_iter=5000)
        return _r12 - _r1 / st2.density_mass

    res = find_root(
        _density_ratio_residual,
        (r12_min, r12_max),
        args=(r1, u1, p1, h1, xnew),
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

    a1 = speed_of_sound(states1)
    a2 = speed_of_sound(states2)
    gam1 = isentropic_exponent(states1)
    gam2 = isentropic_exponent(states2)

    return EqNormalShockSolution(
        upstream=EqFlowStates(
            states=states1,
            velocity=u1,
            speed_of_sound_eq=a1,
            isentropic_exponent=gam1,
        ),
        downstream=EqFlowStates(
            states=states2,
            velocity=u2,
            speed_of_sound_eq=a2,
            isentropic_exponent=gam2,
        ),
    )
