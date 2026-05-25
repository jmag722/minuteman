from enum import Enum, auto
from typing import Annotated, TypeAlias
import numpy as np
import numpy.typing as npt

ndarray_f: TypeAlias = npt.NDArray[np.floating]
"""Numpy array of any floating type"""

Floatlike: TypeAlias = float | np.floating
"""Any float type"""

ArrayOrScalarFloat: TypeAlias = ndarray_f | Floatlike
"""Scalar or array-like float"""


class FlowSpeedRegime(Enum):
    """Is flowfield subsonic or supersonic"""

    subsonic = auto()
    """Subsonic flow (M < 1)"""
    supersonic = auto()
    """Supersonic flow (M > 1)"""


ndarray_FlowSpeedRegime: TypeAlias = Annotated[
    npt.NDArray[np.object_], FlowSpeedRegime
]


class InvalidArrayShape(Exception):
    """Array shapes do not match"""

    pass


def check_equal_shape(
    actual: tuple[int, ...], expected: tuple[int, ...]
) -> None:
    if actual != expected:
        raise InvalidArrayShape(
            f"Actual array shape ({actual}) "
            f"is not equal to the expected ({expected})"
        )


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
            flow_regime is FlowSpeedRegime.supersonic,
            mach_supersonic,
            mach_subsonic,
        )
    else:
        raise InvalidFlowRegime(
            "Use FlowSpeedRegime or ndarray_FlowSpeedRegime to set flow_regime"
        )
