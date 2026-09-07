# Copyright (c) 2022-2026 Jared Magnusson
# SPDX-License-Identifier: Apache-2.0

"""
```bash
usage: nshock [-h] {subcommand}

Normal shock table lookup for a calorically perfect gas

options:
  -h, --help            show this help message and exit

subcommands:
  choose one of the following
    mach-upstream (m1, M1)   Downstream Mach Number
    mach-downstream (m2, M2) Downstream Mach Number
    temperature (T, t)       Temperature ratio T2/T1
    pressure (p, P)          Pressure ratio p2/p1
    density (r, rho)         Density ratio rho2/rho1
    total-pressure (p0)      Total pressure ratio p02/p01
    pitot-pressure           Rayleigh Pitot pressure ratio p02/p1
```

Running help on any of these subcommands will give further details of their
required inputs.

For instance, running `nshock pitot-pressure --help` will yield:

```bash
usage: nshock pitot-pressure [-h] [--specific-heat-ratio x.x] x.x

Compute normal shock outputs by Rayleigh Pitot tube pressure ratio

positional arguments:
  x.x                   Rayleigh Pitot pressure ratio p02/p1

options:
  -h, --help            show this help message and exit
  --specific-heat-ratio x.x, --gamma x.x, -g x.x Specific heat ratio
```
"""

import argparse

from minuteman.cpg import normal_shock
from minuteman.utils.cli import (
    print_line_split,
    print_table_line,
    specific_heat_ratio_parser,
)


def main(args: list[str] | None = None) -> None:
    nshock_parser = argparse.ArgumentParser(
        prog="nshock",
        description="Normal shock table lookup for a calorically perfect gas",
    )
    subparsers = nshock_parser.add_subparsers(
        description="choose one of the following", required=True
    )

    m1_parser = subparsers.add_parser(
        "mach-upstream",
        description="Compute normal shock outputs by upstream Mach",
        aliases=["m1", "M1"],
        parents=[specific_heat_ratio_parser()],
        help="Downstream Mach Number",
    )
    m1_parser.add_argument(
        "mach_upstream", type=float, metavar="x.x", help="Upstream Mach number"
    )
    m1_parser.set_defaults(func=normal_shock.lookup_table_by_upstream_mach)

    m2_parser = subparsers.add_parser(
        "mach-downstream",
        description="Compute normal shock outputs by downstream Mach",
        aliases=["m2", "M2"],
        parents=[specific_heat_ratio_parser()],
        help="Downstream Mach Number",
    )
    m2_parser.add_argument(
        "mach_downstream",
        type=float,
        metavar="x.x",
        help="Downstream Mach number",
    )
    m2_parser.set_defaults(func=normal_shock.lookup_table_by_downstream_mach)

    t_parser = subparsers.add_parser(
        "temperature",
        description="Compute normal shock outputs by temperature ratio",
        aliases=["T", "t"],
        parents=[specific_heat_ratio_parser()],
        help="Temperature ratio T2/T1",
    )
    t_parser.add_argument(
        "temperature_ratio",
        type=float,
        metavar="x.x",
        help="Temperature ratio T2/T1",
    )
    t_parser.set_defaults(func=normal_shock.lookup_table_by_temperature)

    p_parser = subparsers.add_parser(
        "pressure",
        description="Compute normal shock outputs by pressure ratio",
        aliases=["p", "P"],
        parents=[specific_heat_ratio_parser()],
        help="Pressure ratio p2/p1",
    )
    p_parser.add_argument(
        "pressure_ratio",
        type=float,
        metavar="x.x",
        help="Pressure ratio p2/p1",
    )
    p_parser.set_defaults(func=normal_shock.lookup_table_by_pressure)

    r_parser = subparsers.add_parser(
        "density",
        description="Compute normal shock outputs by density ratio",
        aliases=["r", "rho"],
        parents=[specific_heat_ratio_parser()],
        help="Density ratio rho2/rho1",
    )
    r_parser.add_argument(
        "density_ratio",
        type=float,
        metavar="x.x",
        help="Density ratio rho2/rho1",
    )
    r_parser.set_defaults(func=normal_shock.lookup_table_by_density)

    p0_parser = subparsers.add_parser(
        "total-pressure",
        description="Compute normal shock outputs by total pressure ratio",
        aliases=["p0"],
        parents=[specific_heat_ratio_parser()],
        help="Total pressure ratio p02/p01",
    )
    p0_parser.add_argument(
        "total_pressure_ratio",
        type=float,
        metavar="x.x",
        help="Total pressure ratio p02/p01",
    )
    p0_parser.set_defaults(func=normal_shock.lookup_table_by_total_pressure)

    pit_parser = subparsers.add_parser(
        "pitot-pressure",
        description="Compute normal shock outputs by "
        "Rayleigh Pitot tube pressure ratio",
        parents=[specific_heat_ratio_parser()],
        help="Rayleigh Pitot pressure ratio p02/p1",
    )
    pit_parser.add_argument(
        "pitot_pressure_ratio",
        type=float,
        metavar="x.x",
        help="Rayleigh Pitot pressure ratio p02/p1",
    )
    pit_parser.set_defaults(func=normal_shock.lookup_table_by_pitot_pressure)

    parsed_args = nshock_parser.parse_args(args)

    kwargs = vars(parsed_args).copy()
    kwargs.pop("func")
    res = parsed_args.func(**kwargs)

    sym_width = 9
    print_line_split()
    print("MinuteMAN: Normal Shock Lookup Table")
    print_line_split()
    print_table_line(
        name="Upstream Mach Number",
        symbol="M1",
        value=res.mach_upstream.item(),
        sym_width=sym_width,
    )
    print_table_line(
        name="Downstream Mach Number",
        symbol="M2",
        value=res.mach_downstream.item(),
        sym_width=sym_width,
    )
    print_table_line(
        name="Temperature Ratio",
        symbol="T2/T1",
        value=res.temperature_ratio.item(),
        sym_width=sym_width,
    )
    print_table_line(
        name="Pressure Ratio",
        symbol="p2/p1",
        value=res.pressure_ratio.item(),
        sym_width=sym_width,
    )
    print_table_line(
        name="Density Ratio",
        symbol="rho2/rho1",
        value=res.density_ratio.item(),
        sym_width=sym_width,
    )
    print_table_line(
        name="Total Pressure Ratio",
        symbol="p02/p01",
        value=res.total_pressure_ratio.item(),
        sym_width=sym_width,
    )
    print_table_line(
        name="Pitot Pressure Ratio",
        symbol="p02/p1",
        value=res.pitot_pressure_ratio.item(),
        sym_width=sym_width,
    )
    print_table_line(
        name="Specific Heat Ratio",
        symbol="gamma",
        value=res.specific_heat_ratio.item(),
        sym_width=sym_width,
    )
    print_line_split()


if __name__ == "__main__":
    main()
