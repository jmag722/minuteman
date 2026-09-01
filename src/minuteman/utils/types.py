# Copyright (c) 2022-2026 Jared Magnusson
# SPDX-License-Identifier: Apache-2.0

r"""```python
 import minuteman.utils.types
```

Basic minuteman types
"""

from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

NDArrayFloat: TypeAlias = npt.NDArray[np.floating]
"""Numpy array of any floating type"""

NDArrayBool: TypeAlias = npt.NDArray[np.bool]
"""Numpy array of boolean"""

Floatlike: TypeAlias = float | np.floating
"""Any float type"""

ArraylikeFloat: TypeAlias = (
    NDArrayFloat | list[float] | tuple[float, ...] | Floatlike
)
"""Scalar or array-like float"""


class InvalidArrayShapeError(Exception):
    """Array shapes do not match"""


def broadcast_inputs(*args: Any) -> tuple[Any, ...]:
    """Broadcast inputs to be same shape, but not 0D scalar arrays

    Args:
        *args: Arraylike input arguments

    Returns:
        Tuple of contiguous copies of arraylike inputs, at least 1D

    """
    return tuple(
        np.atleast_1d(np.array(a)) for a in np.broadcast_arrays(*args)
    )


class DeveloperError(Exception):
    """Error meaning the developer needs to fix something"""


class RootFindingError(Exception):
    """Error from fsolve, find_root, or a similar optimization function"""
