# Copyright (c) 2022-2026 Jared Magnusson
# SPDX-License-Identifier: Apache-2.0

"""
```bash
usage: cshock [-h] x.x {subcommand}

Conical shock table lookup for a calorically perfect gas

positional arguments:
  x.x                   Upstream Mach number

options:
  -h, --help            show this help message and exit

subcommands:
  choose one of the following
    surface-mach (mc, Mc)         Surface Mach number Mc
    shock-angle (beta, theta_s)   Shock angle theta_s
    cone-angle (theta_c)          Cone angle theta_c
```

Running help on any of these subcommands will give further details of their
required inputs.

For instance, running `cshock 5 cone-angle --help` will yield:

```bash
usage: cshock x.x cone-angle [-h] [--specific-heat-ratio x.x] [--radians] \
[--shock-type <weak, strong>] x.x

Compute conical shock outputs by cone angle

positional arguments:
  x.x                   Cone angle theta_c

options:
  -h, --help            show this help message and exit
  --specific-heat-ratio x.x, --gamma x.x, -g x.x
                        Specific heat ratio
  --radians, -rad       Angle is specified in radians (default is degrees)
  --shock-type <weak, strong>, -st <weak, strong>
                        Oblique shock is weak or strong
```
"""

import argparse

import numpy as np

from minuteman.cpg import cone_flow
from minuteman.utils.cli import (
    print_line_split,
    print_table_angle_line,
    print_table_line,
    radians_parser,
    shock_type_parser,
    specific_heat_ratio_parser,
)


def main(args: list[str] | None = None) -> None:
    cshock_parser = argparse.ArgumentParser(
        prog="cshock",
        description="Conical shock table lookup for a calorically perfect gas",
    )
    cshock_parser.add_argument(
        "mach_upstream", type=float, metavar="x.x", help="Upstream Mach number"
    )

    subparsers = cshock_parser.add_subparsers(
        description="choose one of the following", required=True
    )

    mc_parser = subparsers.add_parser(
        "surface-mach",
        description="Compute conical shock outputs by the surface Mach number",
        aliases=["mc", "Mc"],
        parents=[specific_heat_ratio_parser()],
        help="Surface Mach number Mc",
    )
    mc_parser.add_argument(
        "surface_mach",
        type=float,
        metavar="x.x",
        help="Surface Mach number Mc",
    )
    mc_parser.set_defaults(func=cone_flow.lookup_solution_by_surface_mach)

    beta_parser = subparsers.add_parser(
        "shock-angle",
        description="Compute conical shock outputs by shock angle",
        aliases=["beta", "theta_s"],
        parents=[specific_heat_ratio_parser(), radians_parser()],
        help="Shock angle theta_s",
    )
    beta_parser.add_argument(
        "shock_angle",
        type=float,
        metavar="x.x",
        help="Shock angle theta_s",
    )
    beta_parser.set_defaults(func=cone_flow.lookup_solution_by_shock_angle)

    theta_parser = subparsers.add_parser(
        "cone-angle",
        description="Compute conical shock outputs by cone angle",
        aliases=["theta_c"],
        parents=[
            specific_heat_ratio_parser(),
            radians_parser(),
            shock_type_parser(),
        ],
        help="Cone angle theta_c",
    )
    theta_parser.add_argument(
        "cone_angle",
        type=float,
        metavar="x.x",
        help="Cone angle theta_c",
    )
    theta_parser.set_defaults(func=cone_flow.lookup_solution_by_cone_angle)

    parsed_args = cshock_parser.parse_args(args)

    kwargs = vars(parsed_args).copy()
    if not kwargs.get("radians", False):
        if "shock_angle" in kwargs:
            kwargs["shock_angle"] = np.radians(kwargs["shock_angle"])
        elif "cone_angle" in kwargs:
            kwargs["cone_angle"] = np.radians(kwargs["cone_angle"])
    kwargs.pop("func")
    kwargs.pop("radians", None)

    res = parsed_args.func(**kwargs)

    print_line_split()
    print("MinuteMAN: Conical Shock Lookup Table")
    print_line_split()

    name_width = 23
    sym_width = 9
    print_table_line(
        name="Upstream Mach",
        symbol="M1",
        value=res.mach_upstream,
        name_width=name_width,
        sym_width=sym_width,
    )
    print_table_line(
        name="Downstream Mach",
        symbol="M2",
        value=res.mach[0],
        name_width=name_width,
        sym_width=sym_width,
    )
    print_table_line(
        name="Surface Mach",
        symbol="Mc",
        value=res.mach[-1],
        name_width=name_width,
        sym_width=sym_width,
    )
    print_table_angle_line(
        name="Shock angle",
        symbol="theta_s",
        val_rad=res.shock_angle,
        name_width=name_width,
        sym_width=sym_width,
    )
    print_table_angle_line(
        name="Cone angle",
        symbol="theta_c",
        val_rad=res.cone_angle,
        name_width=name_width,
        sym_width=sym_width,
    )
    print_table_angle_line(
        name="Shock turn angle",
        symbol="psi",
        val_rad=res.flow_angle[0],
        name_width=name_width,
        sym_width=sym_width,
    )
    print_table_line(
        name="Temperature Ratio",
        symbol="T2/T1",
        value=res.temperature_ratio[0],
        name_width=name_width,
        sym_width=sym_width,
    )
    print_table_line(
        name="Pressure Ratio",
        symbol="p2/p1",
        value=res.pressure_ratio[0],
        name_width=name_width,
        sym_width=sym_width,
    )
    print_table_line(
        name="Density Ratio",
        symbol="rho2/rho1",
        value=res.density_ratio[0],
        name_width=name_width,
        sym_width=sym_width,
    )
    print_table_line(
        name="Total Pressure Ratio",
        symbol="p02/p01",
        value=res.total_pressure_ratio,
        name_width=name_width,
        sym_width=sym_width,
    )
    print_table_line(
        name="Surf. Temperature Ratio",
        symbol="Tc/T1",
        value=res.temperature_ratio[-1],
        name_width=name_width,
        sym_width=sym_width,
    )
    print_table_line(
        name="Surf. Pressure Ratio",
        symbol="pc/p1",
        value=res.pressure_ratio[-1],
        name_width=name_width,
        sym_width=sym_width,
    )
    print_table_line(
        name="Surf. Density Ratio",
        symbol="rhoc/rho1",
        value=res.density_ratio[-1],
        name_width=name_width,
        sym_width=sym_width,
    )
    print_table_line(
        name="Specific Heat Ratio",
        symbol="gamma",
        value=res.specific_heat_ratio,
        name_width=name_width,
        sym_width=sym_width,
    )
    print_line_split()


if __name__ == "__main__":
    main()
