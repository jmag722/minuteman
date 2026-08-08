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
    gam_max = 5.0 / 3.0
    gam = np.atleast_1d(specific_heat_ratio)
    if np.any(gam < gam_min) or np.any(gam > gam_max):
        raise OutOfBoundsError(
            f"Specific heat ratio must be within [{gam_min}, {gam_max}]"
        )
