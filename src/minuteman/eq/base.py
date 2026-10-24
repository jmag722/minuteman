# Copyright (c) 2022-2026 Jared Magnusson
# SPDX-License-Identifier: Apache-2.0

r"""```python
from minuteman import eq
```

Base equilibrium flow helper functions.
"""

from typing import Literal

import cantera as ct
import numpy as np
import numpy.typing as npt

from minuteman.utils.bounds_check import check_positive
from minuteman.utils.types import ArraylikeFloat


def initialize_states(
    state_pair: Literal["TP", "TD", "HP", "SP", "SV", "UV"],
    property1: ArraylikeFloat,
    property2: ArraylikeFloat,
    mole_fraction: str
    | dict[str, npt.ArrayLike]
    | npt.ArrayLike = "N2:0.79, O2:0.21",
    mass_fraction: str
    | dict[str, npt.ArrayLike]
    | npt.ArrayLike
    | None = None,
    mechanism: str = "airNASA9.yaml",
) -> ct.SolutionArray:
    r"""Initialize the equilibrium state based upon thermodynamic inputs.

    Args:
        state_pair: Thermodynamic property pairs used for initialization.
            Individual state variables are defined as follows:

            - "T": temperature $T$ [K]. Bounds: (0.0, $\infty$)

            - "P": pressure $p$ [Pa]. Bounds: (0.0, $\infty$)

            - "D": density $\rho$ [kg/m3]. Bounds: (0.0, $\infty$)

            - "H": specific enthalpy $h$ [J/kg]

            - "S": specific entropy $s$ [J/kg-K]

            - "U": specific internal energy $u$ [J/kg]
        property1: Values for first item in state_pair
        property2: Values for second item in state_pair
        mole_fraction: Mole fractions used for initialization
        mass_fraction: Mass fractions used for initialization
        mechanism: Mechanism input file (YAML)

    Returns:
        Equilibrium states
    """
    a1 = np.atleast_1d(property1)
    a2 = np.atleast_1d(property2)
    if state_pair.startswith("T"):
        check_positive(a1)
    check_positive(a2)

    use_x = mass_fraction is None
    frac_str = "X" if use_x else "Y"
    fraction = mole_fraction if use_x else mass_fraction

    equil_str = "TV" if state_pair == "TD" else state_pair
    init_str = state_pair + frac_str
    gas = ct.Solution(mechanism)
    states = ct.SolutionArray(gas, shape=a1.shape)

    setattr(states, init_str, (a1, a2, fraction))
    states.equilibrate(equil_str, max_iter=5000)
    return states
