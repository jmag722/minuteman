from numpy.typing import ArrayLike


def is1known(known, unknowns: ArrayLike):
    return (
        known is not None and all(var is None for var in unknowns)
    )
