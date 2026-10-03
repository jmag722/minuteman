# Copyright (c) 2022-2026 Jared Magnusson
# SPDX-License-Identifier: Apache-2.0

r"""```python
 import minuteman.utils.types
```

Basic minuteman types
"""

from typing import Any, TypeAlias

import cantera as ct
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

EqSolutionArraylike: TypeAlias = ct.Solution | ct.SolutionArray


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


def eq_soln_toarray(st: EqSolutionArraylike) -> ct.SolutionArray:
    """Convert gas state Solution to a SolutionArray

    Args:
        st: Gas state object, Solution or SolutionArray

    Returns:
        SolutionArray

    """
    if isinstance(st, ct.Solution):
        gas2 = ct.Solution(st.source)
        gas2.state = st.state
        return ct.SolutionArray(gas2, shape=1)
    return st


class DeveloperError(Exception):
    """Error meaning the developer needs to fix something"""


class RootFindingError(Exception):
    """Error from fsolve, find_root, or a similar optimization function"""
