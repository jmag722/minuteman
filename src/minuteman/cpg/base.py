"""Base compressible flow relations."""

from enum import Enum, auto
from typing import Annotated, TypeAlias

import numpy as np
import numpy.typing as npt

from minuteman.utils.types import (
    ArrayOrScalarFloat,
    check_equal_shape,
    ndarray_f,
)


class FlowSpeedRegime(Enum):
    """Is flowfield subsonic or supersonic"""

    subsonic = auto()
    r"""Subsonic flow, $M < 1$"""
    supersonic = auto()
    r"""Supersonic flow, $M > 1$"""


ndarray_FlowSpeedRegime: TypeAlias = Annotated[
    npt.NDArray[np.object_], FlowSpeedRegime
]
"""Array of FlowSpeedRegime objects"""


def mach_number(
    velocity: ArrayOrScalarFloat, speed_of_sound: ArrayOrScalarFloat
) -> ndarray_f:
    r"""Compute Mach number, $M$

    Args:
        velocity (ArrayOrScalarFloat): velocity, $v$
        speed_of_sound (ArrayOrScalarFloat): speed of sound, $a$

    Returns:
        ndarray_f: Mach number, $M$
    """
    return np.atleast_1d(velocity) / speed_of_sound


def speed_of_sound_from_temperature(
    gas_constant: ArrayOrScalarFloat,
    temperature: ArrayOrScalarFloat,
    specific_heat_ratio: ArrayOrScalarFloat,
) -> ndarray_f:
    r"""Compute the speed of sound from the specific gas constant $R$ and
    temperature $T$

    Args:
        gas_constant (ArrayOrScalarFloat): specific gas constant, $R$
        temperature (ArrayOrScalarFloat): temperature $T$
        specific_heat_ratio (ArrayOrScalarFloat): ratio of specific heats,
            $\gamma$

    Returns:
        ndarray_f: speed of sound, $a$
    """
    gam = np.atleast_1d(specific_heat_ratio)
    return (gam * gas_constant * temperature) ** 0.5


def speed_of_sound_from_pressure(
    pressure: ArrayOrScalarFloat,
    density: ArrayOrScalarFloat,
    specific_heat_ratio: ArrayOrScalarFloat,
) -> ndarray_f:
    r"""Compute the speed of sound $a$ from the pressure $p$ and density $\rho$

    Args:
        pressure (ArrayOrScalarFloat): pressure, $p$
        density (ArrayOrScalarFloat): density, $\rho$
        specific_heat_ratio (ArrayOrScalarFloat): ratio of specific heats,
            $\gamma$

    Returns:
        ndarray_f: speed of sound, $a$
    """
    gam = np.atleast_1d(specific_heat_ratio)
    return (gam * pressure / density) ** 0.5


class InvalidFlowRegime(Exception):
    """FlowSpeedRegime is invalid"""

    pass


def mach_guess_from_flow_regime(
    flow_regime: ndarray_FlowSpeedRegime | FlowSpeedRegime,
    shape: tuple[int, ...],
    mach_subsonic: float,
    mach_supersonic: float,
) -> ndarray_f:
    if flow_regime is FlowSpeedRegime.supersonic:
        return np.full(shape, mach_supersonic)
    elif flow_regime is FlowSpeedRegime.subsonic:
        return np.full(shape, mach_subsonic)
    elif isinstance(flow_regime, np.ndarray):
        check_equal_shape(flow_regime.shape, shape)
        return np.where(
            flow_regime == FlowSpeedRegime.supersonic,
            mach_supersonic,
            mach_subsonic,
        )
    else:
        raise InvalidFlowRegime(
            "Use FlowSpeedRegime or ndarray_FlowSpeedRegime to set flow_regime"
        )
