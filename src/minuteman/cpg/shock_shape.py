# Copyright (c) 2022-2026 Jared Magnusson
# SPDX-License-Identifier: Apache-2.0

r"""This module computes the shock shape for the detached shock of a blunt body
for a calorically perfect gas ($\gamma=1.4$).

The solutions follow that of Billig as described in
[1-2](shock_shape.md#references))
"""

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from minuteman.cpg.cone_flow import lookup_solution_by_cone_angle
from minuteman.cpg.oblique_shock import (
    ObliqueShockType,
    lookup_table_by_deflection_angle,
)
from minuteman.utils.bounds_check import OutOfBoundsError
from minuteman.utils.types import ArraylikeFloat, Floatlike, NDArrayFloat


class BluntBodyType(Enum):
    """Blunt body type (sphere-cone or cylinder-wedge)"""

    sphere_cone = auto()
    """Sphere-cone shape"""

    cylinder_wedge = auto()
    """Cylinder-wedge shape"""


@dataclass
class ShockShapeSolution:
    r"""Shock shape solution for a calorically perfect gas"""

    x: NDArrayFloat
    r"""x-coordinates, $x$. Zero is defined at the tip of the nose."""

    y: NDArrayFloat
    r"""y-coordinates, $y$. Zero is defined at the axis of symmetry."""

    mach: float
    r"""Upstream Mach number, $M$."""

    nose_radius: float
    """Nose radius $R$"""

    shock_angle: float
    r"""Attached shock angle $\theta$ [radians]"""

    half_angle: float
    r"""Wedge or cone half angle [radians]"""

    standoff_distance: float
    r"""Shock standoff distance $\Delta$"""

    curvature_radius: float
    """Vertex radius of curvature $R_c$"""

    body_type: BluntBodyType
    """Blunt body type"""


def estimate_coords_cylinder_wedge(
    y: ArraylikeFloat,
    mach: Floatlike,
    nose_radius: Floatlike,
    half_angle: Floatlike,
) -> ShockShapeSolution:
    r"""Estimate the shock-shape for a cylinder-wedge at a given freestream
    Mach number.

    Args:
        y: y-coordinates $y$. A value of $y=0$ is defined at the axis of
            symmetry. Bounds: $[0, \infty)$
        mach: Freestream Mach number $M$. Bounds: $(1, \infty)$
        nose_radius: Nose radius $R$. Bounds: $(0, \infty)$
        half_angle: Wedge half angle $\theta$ [radians]. Bounds: $(0, \pi/2)$

    Returns:
        Shock shape solution

    """
    _y = np.atleast_1d(y)
    m = float(mach)
    r = float(nose_radius)
    theta_w = float(half_angle)

    if np.any(_y < 0.0):
        raise OutOfBoundsError("y-coords must be >= 0")
    if m <= 1:
        raise OutOfBoundsError("Mach number M must be > 1")
    if r <= 0:
        raise OutOfBoundsError("Nose radius R must be > 0")
    if theta_w <= 0 or theta_w >= 0.5 * np.pi:
        raise OutOfBoundsError("Wedge angle must be within (0, 90) deg")

    delta = standoff_distance_cylinder_wedge(mach=m, nose_radius=r).item()
    rc = curvature_radius_cylinder_wedge(mach=m, nose_radius=r).item()
    # assuming weak solution, unsure if Billig correlations apply for
    # strong-shock solution
    wedge_soln = lookup_table_by_deflection_angle(
        deflection_angle=theta_w,
        mach_upstream=m,
        specific_heat_ratio=1.4,
        shock_type=ObliqueShockType.weak,
    )
    theta_s = wedge_soln.shock_angle.item()
    x = x_coordinates(
        y=_y,
        shock_angle=theta_s,
        nose_radius=r,
        standoff_distance=delta,
        curvature_radius=rc,
    )

    return ShockShapeSolution(
        x=x,
        y=_y,
        mach=m,
        nose_radius=r,
        shock_angle=theta_s,
        half_angle=theta_w,
        standoff_distance=delta,
        curvature_radius=rc,
        body_type=BluntBodyType.cylinder_wedge,
    )


def estimate_coords_sphere_cone(
    y: ArraylikeFloat,
    mach: Floatlike,
    nose_radius: Floatlike,
    half_angle: Floatlike,
) -> ShockShapeSolution:
    r"""Estimate the shock-shape for a sphere-cone at a given freestream Mach
    number.

    Args:
        y: y-coordinates $y$. A value of $y=0$ is defined at the axis of
            symmetry. Bounds: $[0, \infty)$
        mach: Freestream Mach number $M$. Bounds: $(1, \infty)$
        nose_radius: Nose radius $R$. Bounds: $(0, \infty)$
        half_angle: Cone half angle $\theta_c$ [radians]. Bounds: $(0, \pi/2)$

    Returns:
        Shock shape solution

    """
    _y = np.atleast_1d(y)
    m = float(mach)
    r = float(nose_radius)
    theta_c = float(half_angle)

    if np.any(_y < 0.0):
        raise OutOfBoundsError("y-coords must be >= 0")
    if m <= 1:
        raise OutOfBoundsError("Mach number M must be > 1")
    if r <= 0:
        raise OutOfBoundsError("Nose radius R must be > 0")
    if theta_c <= 0 or theta_c >= 0.5 * np.pi:
        raise OutOfBoundsError("Cone angle must be within (0, 90) deg")

    delta = standoff_distance_sphere_cone(mach=m, nose_radius=r).item()
    rc = curvature_radius_sphere_cone(mach=m, nose_radius=r).item()
    # assuming weak solution, unsure if Billig correlations apply for
    # strong-shock solution
    cone_soln = lookup_solution_by_cone_angle(
        cone_angle=theta_c,
        mach_upstream=m,
        specific_heat_ratio=1.4,
        shock_type=ObliqueShockType.weak,
    )
    theta_s = cone_soln.shock_angle
    x = x_coordinates(
        y=_y,
        shock_angle=theta_s,
        nose_radius=r,
        standoff_distance=delta,
        curvature_radius=rc,
    )

    return ShockShapeSolution(
        x=x,
        y=_y,
        mach=m,
        nose_radius=r,
        shock_angle=theta_s,
        half_angle=theta_c,
        standoff_distance=delta,
        curvature_radius=rc,
        body_type=BluntBodyType.sphere_cone,
    )


def x_coordinates(
    y: ArraylikeFloat,
    shock_angle: Floatlike,
    nose_radius: Floatlike,
    standoff_distance: Floatlike,
    curvature_radius: Floatlike,
) -> NDArrayFloat:
    r"""Compute the x-coordinates of the shock shape of a blunt body $x$ for a
    given set of y-coordinates.

    Differing from the relation this function is based on, the positive x-axis
    is defined downstream of the vehicle, with $x=0$ centered at the nose tip,
    not the center of curvature

    Args:
        y: y-coordinates $y$. A value of $y=0$ is defined at the axis of
            symmetry.
        shock_angle: attached shock angle $\theta$ for a cone or
            wedge [radians]
        nose_radius: nose radius $R$
        standoff_distance: shock standoff distance $\Delta$
        curvature_radius: vertex radius of curvature $R_c$

    Returns:
        x-coordinates of shock shape, $x$.

    """
    _y = np.atleast_1d(y)
    theta = float(shock_angle)
    r = float(nose_radius)
    delta = float(standoff_distance)
    rc = float(curvature_radius)

    # flip axis direction and add nose radius to get to more usable coordinate
    return r - (
        r
        + delta
        - rc
        * np.tan(theta) ** -2.0
        * ((1 + (_y * np.tan(theta) / rc) ** 2) ** 0.5 - 1)
    )


def standoff_distance_sphere_cone(
    mach: ArraylikeFloat, nose_radius: ArraylikeFloat
) -> NDArrayFloat:
    r"""Compute the shock standoff distance $\Delta$ for a sphere-cone.

    Args:
        mach: Mach number $M$
        nose_radius: Sphere-cone radius $R$

    Returns:
        Shock standoff distance $\Delta$
    """
    m = np.atleast_1d(mach)
    r = np.atleast_1d(nose_radius)
    return r * 0.143 * np.exp(3.24 / m**2)


def standoff_distance_cylinder_wedge(
    mach: ArraylikeFloat, nose_radius: ArraylikeFloat
) -> NDArrayFloat:
    r"""Compute the shock standoff distance $\Delta$ for a cylinder-wedge.

    Args:
        mach: Mach number $M$
        nose_radius: Cylinder-wedge radius $R$

    Returns:
        Shock standoff distance $\Delta$
    """
    m = np.atleast_1d(mach)
    r = np.atleast_1d(nose_radius)
    return r * 0.386 * np.exp(4.67 / m**2)


def curvature_radius_sphere_cone(
    mach: ArraylikeFloat, nose_radius: ArraylikeFloat
) -> NDArrayFloat:
    """Compute the vertex radius of curvature for a sphere-cone, $R_c$.

    Args:
        mach: Mach number $M$
        nose_radius: Sphere-cone radius $R$

    Returns:
        Radius of curvature $R_c$
    """
    m = np.atleast_1d(mach)
    r = np.atleast_1d(nose_radius)
    return r * 1.143 * np.exp(0.54 / (m - 1.0) ** 1.2)


def curvature_radius_cylinder_wedge(
    mach: ArraylikeFloat, nose_radius: ArraylikeFloat
) -> NDArrayFloat:
    """Compute the vertex radius of curvature for a cylinder-wedge, $R_c$.

    Args:
        mach: Mach number $M$
        nose_radius: Cylinder-wedge radius $R$

    Returns:
        Radius of curvature $R_c$
    """
    m = np.atleast_1d(mach)
    r = np.atleast_1d(nose_radius)
    return r * 1.386 * np.exp(1.8 / (m - 1.0) ** 0.75)
