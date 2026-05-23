from typing import TypeAlias
import numpy as np
import numpy.typing as npt

ndarray: TypeAlias = npt.NDArray[np.generic]


def scalar2array(scalar: np.generic, shape: tuple[int, ...]) -> ndarray:
    """Compute argument to an array of the given shape, if not already an array
    of that shape

    Args:
        scalar (float): scalar
        shape (tuple[int, ...]): desired shape

    Returns:
        ndarray: scalar converted to array
    """
    return np.zeros(shape=shape, dtype=type(scalar)) + scalar
