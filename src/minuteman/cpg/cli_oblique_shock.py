# Copyright (c) 2022-2026 Jared Magnusson
# SPDX-License-Identifier: Apache-2.0

"""
```bash
usage: oshock [-h] x.x {subcommand}

Oblique shock table lookup for a calorically perfect gas

positional arguments:
  x.x                   Upstream Mach number

options:
  -h, --help            show this help message and exit

subcommands:
  choose one of the following
    mach-upstream-normal (mn1, Mn1) Normal component of \
upstream Mach number Mn1
    shock-angle (beta)              Shock angle beta
    deflection-angle (theta)        Deflection angle theta
```

Running help on any of these subcommands will give further details of their
required inputs.

For instance, running `oshock 5 beta --help` will yield:

```bash
usage: oshock x.x shock-angle [-h] [--specific-heat-ratio x.x] [--radians] x.x

Compute oblique shock outputs by shock angle

positional arguments:
  x.x                   Shock angle beta

options:
  -h, --help            show this help message and exit
  --specific-heat-ratio x.x, --gamma x.x, -g x.x
                        Specific heat ratio
  --radians, -rad       Angle is specified in radians (default is degrees)
```
"""

import argparse

import numpy as np

from minuteman.cpg import oblique_shock
from minuteman.utils.cli import (
    print_line_split,
    print_table_angle_line,
    print_table_line,
    radians_parser,
    shock_type_parser,
    specific_heat_ratio_parser,
)


def main(args: list[str] | None = None) -> None:
    oshock_parser = argparse.ArgumentParser(
        prog="oshock",
        description="Oblique shock table lookup for a calorically perfect gas",
    )
    oshock_parser.add_argument(
        "mach_upstream", type=float, metavar="x.x", help="Upstream Mach number"
    )

    subparsers = oshock_parser.add_subparsers(
        description="choose one of the following", required=True
    )

    mn1_parser = subparsers.add_parser(
        "mach-upstream-normal",
        description="Compute oblique shock outputs by the normal component "
        "of the upstream Mach number",
        aliases=["mn1", "Mn1"],
        parents=[specific_heat_ratio_parser()],
        help="Normal component of upstream Mach number Mn1",
    )
    mn1_parser.add_argument(
        "mach_upstream_normal",
        type=float,
        metavar="x.x",
        help="Normal component of upstream Mach number Mn1",
    )
    mn1_parser.set_defaults(
        func=oblique_shock.lookup_table_by_mach_upstream_normal
    )

    beta_parser = subparsers.add_parser(
        "shock-angle",
        description="Compute oblique shock outputs by shock angle",
        aliases=["beta"],
        parents=[specific_heat_ratio_parser(), radians_parser()],
        help="Shock angle beta",
    )
    beta_parser.add_argument(
        "shock_angle",
        type=float,
        metavar="x.x",
        help="Shock angle beta",
    )
    beta_parser.set_defaults(func=oblique_shock.lookup_table_by_shock_angle)

    theta_parser = subparsers.add_parser(
        "deflection-angle",
        description="Compute oblique shock outputs by deflection angle",
        aliases=["theta"],
        parents=[
            specific_heat_ratio_parser(),
            radians_parser(),
            shock_type_parser(),
        ],
        help="Deflection angle theta",
    )
    theta_parser.add_argument(
        "deflection_angle",
        type=float,
        metavar="x.x",
        help="Deflection angle theta",
    )
    theta_parser.set_defaults(
        func=oblique_shock.lookup_table_by_deflection_angle
    )

    parsed_args = oshock_parser.parse_args(args)

    kwargs = vars(parsed_args).copy()
    if not kwargs.get("radians", False):
        if "shock_angle" in kwargs:
            kwargs["shock_angle"] = np.radians(kwargs["shock_angle"])
        elif "deflection_angle" in kwargs:
            kwargs["deflection_angle"] = np.radians(kwargs["deflection_angle"])
    kwargs.pop("func")
    kwargs.pop("radians", None)

    res = parsed_args.func(**kwargs)

    print_line_split()
    print("MinuteMAN: Oblique Shock Lookup Table")
    print_line_split()

    name_width = 23
    sym_width = 9
    print_table_line(
        name="Upstream Mach",
        symbol="M1",
        value=res.mach_upstream.item(),
        name_width=name_width,
        sym_width=sym_width,
    )
    print_table_line(
        name="Downstream Mach",
        symbol="M2",
        value=res.mach_downstream.item(),
        name_width=name_width,
        sym_width=sym_width,
    )
    print_table_line(
        name="Norm. Upstream Mach",
        symbol="Mn1",
        value=res.mach_upstream_normal.item(),
        name_width=name_width,
        sym_width=sym_width,
    )
    print_table_line(
        name="Norm. Downstream Mach",
        symbol="Mn2",
        value=res.mach_downstream_normal.item(),
        name_width=name_width,
        sym_width=sym_width,
    )
    print_table_angle_line(
        name="Shock angle",
        symbol="beta",
        val_rad=res.shock_angle.item(),
        name_width=name_width,
        sym_width=sym_width,
    )
    print_table_angle_line(
        name="Deflection angle",
        symbol="theta",
        val_rad=res.deflection_angle.item(),
        name_width=name_width,
        sym_width=sym_width,
    )
    print_table_line(
        name="Temperature Ratio",
        symbol="T2/T1",
        value=res.temperature_ratio.item(),
        name_width=name_width,
        sym_width=sym_width,
    )
    print_table_line(
        name="Pressure Ratio",
        symbol="p2/p1",
        value=res.pressure_ratio.item(),
        name_width=name_width,
        sym_width=sym_width,
    )
    print_table_line(
        name="Density Ratio",
        symbol="rho2/rho1",
        value=res.density_ratio.item(),
        name_width=name_width,
        sym_width=sym_width,
    )
    print_table_line(
        name="Total Pressure Ratio",
        symbol="p02/p01",
        value=res.total_pressure_ratio.item(),
        name_width=name_width,
        sym_width=sym_width,
    )
    print_table_line(
        name="Specific Heat Ratio",
        symbol="gamma",
        value=res.specific_heat_ratio.item(),
        name_width=name_width,
        sym_width=sym_width,
    )
    print_line_split()


if __name__ == "__main__":
    main()
