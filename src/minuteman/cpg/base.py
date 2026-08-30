# Copyright (c) 2022-2026 Jared Magnusson
# SPDX-License-Identifier: Apache-2.0

r"""```python
 import minuteman.cpg.base
```

Base compressible flow relations.
"""

from enum import Enum, auto
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from minuteman.utils.types import (
    ArraylikeFloat,
    NDArrayFloat,
)


class FlowSpeedRegime(Enum):
    """Is flowfield subsonic or supersonic"""

    subsonic = auto()
    r"""Subsonic flow, $M < 1$"""
    supersonic = auto()
    r"""Supersonic flow, $M > 1$"""


ArraylikeFlowSpeedRegime: TypeAlias = (
    npt.NDArray[np.object_] | list[FlowSpeedRegime] | FlowSpeedRegime
)
"""Arraylike of FlowSpeedRegime objects"""


def mach_number(
    velocity: ArraylikeFloat,
    speed_of_sound: ArraylikeFloat,
) -> NDArrayFloat:
    r"""Compute Mach number, $M$

    Args:
        velocity: velocity, $v$
        speed_of_sound: speed of sound, $a$

    Returns:
        Mach number, $M$

    """
    return np.atleast_1d(velocity) / np.atleast_1d(speed_of_sound)


def speed_of_sound_by_temperature(
    gas_constant: ArraylikeFloat,
    temperature: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat,
) -> NDArrayFloat:
    r"""Compute the speed of sound from the specific gas constant $R$ and
    temperature $T$

    Args:
        gas_constant: specific gas constant, $R$
        temperature: temperature $T$
        specific_heat_ratio: ratio of specific heats, $\gamma$

    Returns:
        Speed of sound, $a$

    """
    gam = np.atleast_1d(specific_heat_ratio)
    gc = np.atleast_1d(gas_constant)
    t = np.atleast_1d(temperature)
    return (gam * gc * t) ** 0.5


def speed_of_sound_by_pressure(
    pressure: ArraylikeFloat,
    density: ArraylikeFloat,
    specific_heat_ratio: ArraylikeFloat,
) -> NDArrayFloat:
    r"""Compute the speed of sound $a$ from the pressure $p$ and density $\rho$

    Args:
        pressure: pressure, $p$
        density: density, $\rho$
        specific_heat_ratio: ratio of specific heats, $\gamma$

    Returns:
        Speed of sound, $a$

    """
    gam = np.atleast_1d(specific_heat_ratio)
    p = np.atleast_1d(pressure)
    rho = np.atleast_1d(density)
    return (gam * p / rho) ** 0.5


class InvalidFlowRegimeError(Exception):
    """FlowSpeedRegime is invalid"""


def bracket_mach_from_flow_regime(
    flow_regime: ArraylikeFlowSpeedRegime,
) -> tuple[NDArrayFloat, NDArrayFloat]:
    subsonic_mach_min = 1e-50
    subsonic_mach_max = 1.0 - subsonic_mach_min
    supersonic_mach_min = 1.0
    supersonic_mach_max = 1e10
    if flow_regime is FlowSpeedRegime.supersonic:
        return (
            np.atleast_1d(supersonic_mach_min),
            np.atleast_1d(supersonic_mach_max),
        )
    if flow_regime is FlowSpeedRegime.subsonic:
        return (
            np.atleast_1d(subsonic_mach_min),
            np.atleast_1d(subsonic_mach_max),
        )
    if isinstance(flow_regime, np.ndarray | list):
        return (
            np.where(
                flow_regime == FlowSpeedRegime.supersonic,
                supersonic_mach_min,
                subsonic_mach_min,
            ),
            np.where(
                flow_regime == FlowSpeedRegime.supersonic,
                supersonic_mach_max,
                subsonic_mach_max,
            ),
        )
    raise InvalidFlowRegimeError(
        "Use ArraylikeFlowSpeedRegime to set flow_regime",
    )
