# Copyright (c) 2022-2026 Jared Magnusson
# SPDX-License-Identifier: Apache-2.0

r"""```python
 import minuteman.utils.types
```

Basic minuteman types
"""

from typing import TypeAlias

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


class DeveloperError(Exception):
    """Error meaning the developer needs to fix something"""


class RootFindingError(Exception):
    """Error from fsolve, find_root, or a similar optimization function"""
