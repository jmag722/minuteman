# Copyright (c) 2022-2026 Jared Magnusson
# SPDX-License-Identifier: Apache-2.0

r"""
Bounds checking for inputs
"""

import numpy as np

from minuteman.utils.types import ArraylikeFloat


class OutOfBoundsError(Exception):
    """Input value is out of bounds"""


def check_positive(val: ArraylikeFloat) -> None:
    v = np.atleast_1d(val)
    if np.any(v <= 0.0):
        raise OutOfBoundsError("Value must be positive")


def check_nonnegative(val: ArraylikeFloat) -> None:
    v = np.atleast_1d(val)
    if np.any(v < 0.0):
        raise OutOfBoundsError("Value must not be negative")


def check_specific_heat_ratio(specific_heat_ratio: ArraylikeFloat) -> None:
    gam_min = 1.0
    # I know gam_max should be 5/3=1.66666666..., but to prevent against
    # user frustration I'll let them round up
    gam_max = 1.67
    gam = np.atleast_1d(specific_heat_ratio)
    if np.any(gam < gam_min) or np.any(gam > gam_max):
        raise OutOfBoundsError(
            f"Specific heat ratio must be within [{gam_min}, {gam_max}]"
        )


def check_mach_supersonic(mach: ArraylikeFloat) -> None:
    m = np.atleast_1d(mach)
    if np.any(m <= 1.0):
        raise OutOfBoundsError("Mach number must be > 1.0")
