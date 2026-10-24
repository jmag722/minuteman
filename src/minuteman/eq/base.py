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

from minuteman.utils.types import ArraylikeFloat


def initialize_states(
    property_pair: Literal["TP", "TD", "HP", "SP", "SV", "UV"],
    arr1: ArraylikeFloat,
    arr2: ArraylikeFloat,
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
        property_pair: Thermodynamic property pairs used for initialization.
            Individual properties are defined as follows:

            - "T": temperature $T$ [K]

            - "P": pressure $p$ [Pa]

            - "D": density $\rho$ [kg/m3]

            - "H": specific enthalpy $h$ [J/kg]

            - "S": specific entropy $s$ [J/kg-K]

            - "U": specific internal energy $u$ [J/kg]
        arr1: Values for first item in property_pair
        arr2: Values for second item in property_pair
        mole_fraction: Mole fractions used for initialization
        mass_fraction: Mass fractions used for initialization
        mechanism: Mechanism input file (YAML)

    Returns:
        Equilibrium states
    """
    gas = ct.Solution(mechanism)
    a1 = np.atleast_1d(arr1)
    a2 = np.atleast_1d(arr2)
    states = ct.SolutionArray(gas, shape=a1.shape)

    use_x = mass_fraction is None
    frac_str = "X" if use_x else "Y"
    fraction = mole_fraction if use_x else mass_fraction

    equil_str = "TV" if property_pair == "TD" else property_pair
    init_str = property_pair + frac_str
    setattr(states, init_str, (a1, a2, fraction))
    states.equilibrate(equil_str, max_iter=5000)
    return states
