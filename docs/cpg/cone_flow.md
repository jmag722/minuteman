# Cone Flow Relations

::: minuteman.cpg.cone_flow
    options:
      show_root_heading: false
      members: no

## High-Level API

::: minuteman.cpg.cone_flow.lookup_solution_by_cone_angle

::: minuteman.cpg.cone_flow.lookup_solution_by_shock_angle

::: minuteman.cpg.cone_flow.lookup_solution_by_surface_mach

## Low-Level API

::: minuteman.cpg.cone_flow.solve_taylor_maccoll_by_cone_angle

::: minuteman.cpg.cone_flow.solve_taylor_maccoll_by_shock_angle

::: minuteman.cpg.cone_flow.solve_taylor_maccoll_by_surface_mach

::: minuteman.cpg.cone_flow.cone_shock_angle_maxes

::: minuteman.cpg.cone_flow.deflection_angle_by_velocity_components

::: minuteman.cpg.cone_flow.mach_from_nondimensional_velocity

::: minuteman.cpg.cone_flow.nondimensional_velocity_from_components

::: minuteman.cpg.cone_flow.nondimensional_velocity_from_mach

::: minuteman.cpg.cone_flow.nondimensional_velocity_polar

::: minuteman.cpg.cone_flow.nondimensional_velocity_radial

## Data Structures

::: minuteman.cpg.cone_flow.ConeFlowSolution
    options:
      separate_signature: false

## Error Types

::: minuteman.cpg.cone_flow.InvalidConeAngleError

::: minuteman.cpg.cone_flow.InvalidSurfaceMachError

::: minuteman.cpg.cone_flow.InvalidPolarVelocityError

::: minuteman.cpg.cone_flow.SolveIVPError

## Theory

## References

1. Anderson, J. D., Jr. (2003). *Modern compressible flow: With historical perspective* (3rd ed.). McGraw-Hill.
2. Sims, J. L. (1964). Tables for supersonic flow around right circular cones at zero angle of attack.
