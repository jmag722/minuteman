from typing import TypeAlias

import numpy as np
import numpy.typing as npt

ndarray_f: TypeAlias = npt.NDArray[np.floating]
"""Numpy array of any floating type"""

ndarray_b: TypeAlias = npt.NDArray[np.bool]
"""Numpy array of boolean"""

Floatlike: TypeAlias = float | np.floating
"""Any float type"""

ArrayOrScalarFloat: TypeAlias = ndarray_f | Floatlike
"""Scalar or array-like float"""


class InvalidArrayShape(Exception):
    """Array shapes do not match"""

    pass


class DeveloperError(Exception):
    """Error meaning the developer needs to fix something"""

    pass


def check_equal_shape(
    actual: tuple[int, ...], expected: tuple[int, ...]
) -> None:
    if actual != expected:
        raise InvalidArrayShape(
            f"Actual array shape ({actual}) "
            f"is not equal to the expected ({expected})"
        )
