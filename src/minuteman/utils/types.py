r"""

```python
 import minuteman.utils.types
```

Basic minuteman types
"""

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

ndarray_f: TypeAlias = npt.NDArray[np.floating]
"""Numpy array of any floating type"""

ndarray_b: TypeAlias = npt.NDArray[np.bool]
"""Numpy array of boolean"""

Floatlike: TypeAlias = float | np.floating
"""Any float type"""

ArraylikeFloat: TypeAlias = (
    ndarray_f | list[float] | tuple[float] | Floatlike
)
"""Scalar or array-like float"""


class InvalidArrayShapeError(Exception):
    """Array shapes do not match"""

    pass


class DeveloperError(Exception):
    """Error meaning the developer needs to fix something"""

    pass

class RootFindingError(Exception):
    """Error from fsolve, find_root, or a similar optimization function"""

    pass
