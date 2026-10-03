# Copyright (c) 2022-2026 Jared Magnusson
# SPDX-License-Identifier: Apache-2.0

r"""```python
 import minuteman.eq.thermo as thermo
```
High-temperature equilibrium gases are reacting gases in chemical equilibrium
and where intermolecular forces are neglected.

"""

import cantera as ct
import numpy as np

from minuteman.utils.types import (
    EqSolutionArraylike,
    NDArrayFloat,
    eq_soln_toarray,
)


def speed_of_sound(states: EqSolutionArraylike) -> NDArrayFloat:
    r"""Compute the equilibrium sound speed $a_{eq}$ from a given set
    of thermodynamic states.

    Implementation derived from Eq. 7.96-7.102 of
    Kao, S., Ziegler, J., Bitter, N., Schmidt, B., Lawson, J., & Shepherd,
    J. E. (2023). *SDToolbox: Numerical tools for shock and
    detonation wave modeling*. *Explosion Dynamics Laboratory*,
    *California Institute of Technology*, Pasadena, CA USA, 911125.

    Args:
        states: Equilibrium states. There is no check that the input states
            are in equilibrium.

    Returns:
        Equilibrium speed of sound $a_{eq}$

    """
    st1 = eq_soln_toarray(states)
    p0 = st1.P
    t0 = st1.T
    dfrac = 1e-4
    fac1 = 1 - dfrac
    fac2 = 1 + dfrac

    p1 = p0 * fac1
    t1 = t0 * fac1
    p2 = p0 * fac2
    t2 = t0 * fac2

    def _get_state(p, t):
        gas_new = ct.Solution(st1.source)
        states_new = ct.SolutionArray(gas_new, shape=p0.shape)
        states_new.TPX = t, p, st1.X
        states_new.equilibrate("TP")
        return states_new

    states_p1t0 = _get_state(p=p1, t=t0)
    states_p2t0 = _get_state(p=p2, t=t0)
    ds_dp = (states_p2t0.entropy_mass - states_p1t0.entropy_mass) / (p2 - p1)

    states_p0t1 = _get_state(p=p0, t=t1)
    states_p0t2 = _get_state(p=p0, t=t2)
    ds_dt = (states_p0t2.entropy_mass - states_p0t1.entropy_mass) / (t2 - t1)

    dt_dp = -ds_dp / ds_dt
    ta = t0 + dt_dp * (p2 - p0)
    tb = t0 + dt_dp * (p1 - p0)

    states_p2ta = _get_state(p=p2, t=ta)
    states_p1tb = _get_state(p=p1, t=tb)

    return np.sqrt(
        (p2 - p1) / (states_p2ta.density_mass - states_p1tb.density_mass)
    )


def specific_heat_ratio(states: EqSolutionArraylike) -> NDArrayFloat:
    r"""Compute the equilibrium specific heat ratio or isentropic
    expansion factor $\gamma_s$.

    Args:
        states: Equilibrium states

    Returns:
        Equilibrium specific heat ratio $\gamma_s$

    """
    st = eq_soln_toarray(states)
    a_eq = speed_of_sound(st)
    return a_eq**2 * st.density_mass / st.P
