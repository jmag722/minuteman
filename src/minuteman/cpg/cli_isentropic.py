# Copyright (c) 2022-2026 Jared Magnusson
# SPDX-License-Identifier: Apache-2.0

"""
```bash
isen --help

usage: isen [-h] {subcommand}

Isentropic flow table lookup for a calorically perfect gas

options:
  -h, --help            show this help message and exit

subcommands:
  choose one of the following
    mach (m, M)              Mach Number
    temperature (T, t)       Temperature ratio T0/T
    pressure (p, P)          Pressure ratio p0/p
    density (r, rho)         Density ratio rho0/rho
    area (A)                 Area ratio A/A*
    mach-angle (mu)          Mach angle mu
    prandtl-meyer (nu, pm)   Prandtl-Meyer function nu
```

Running help on any of these subcommands will give further details of their
required inputs.

For instance, running `isen mach --help` will yield:

```bash
usage: isen mach [-h] [--specific-heat-ratio x.x] x.x

Compute isentropic flow outputs by Mach

positional arguments:
  x.x                   Mach number

options:
  -h, --help            show this help message and exit
  --specific-heat-ratio x.x, --gamma x.x, -g x.x Specific heat ratio
```
"""

import argparse

import numpy as np

from minuteman.cpg import isentropic_flow
from minuteman.utils.cli import (
    flow_regime_parser,
    print_line_split,
    print_table_angle_line,
    print_table_line,
    radians_parser,
    specific_heat_ratio_parser,
)


def main(args: list[str] | None = None) -> None:
    isen_parser = argparse.ArgumentParser(
        prog="isen",
        description="Isentropic flow table lookup for a "
        "calorically perfect gas",
    )
    subparsers = isen_parser.add_subparsers(
        description="choose one of the following", required=True
    )

    mach_parser = subparsers.add_parser(
        "mach",
        description="Compute isentropic flow outputs by Mach",
        aliases=["m", "M"],
        parents=[specific_heat_ratio_parser()],
        help="Mach Number",
    )
    mach_parser.add_argument(
        "mach", type=float, metavar="x.x", help="Mach number"
    )
    mach_parser.set_defaults(func=isentropic_flow.lookup_table_by_mach)

    t_parser = subparsers.add_parser(
        "temperature",
        description="Compute isentropic flow outputs by temperature ratio",
        aliases=["T", "t"],
        parents=[specific_heat_ratio_parser()],
        help="Temperature ratio T0/T",
    )
    t_parser.add_argument(
        "temperature_ratio",
        type=float,
        metavar="x.x",
        help="Temperature ratio T0/T",
    )
    t_parser.set_defaults(func=isentropic_flow.lookup_table_by_temperature)

    p_parser = subparsers.add_parser(
        "pressure",
        description="Compute isentropic flow outputs by pressure ratio",
        aliases=["p", "P"],
        parents=[specific_heat_ratio_parser()],
        help="Pressure ratio p0/p",
    )
    p_parser.add_argument(
        "pressure_ratio", type=float, metavar="x.x", help="Pressure ratio p0/p"
    )
    p_parser.set_defaults(func=isentropic_flow.lookup_table_by_pressure)

    r_parser = subparsers.add_parser(
        "density",
        description="Compute isentropic flow outputs by density ratio",
        aliases=["r", "rho"],
        parents=[specific_heat_ratio_parser()],
        help="Density ratio rho0/rho",
    )
    r_parser.add_argument(
        "density_ratio",
        type=float,
        metavar="x.x",
        help="Density ratio rho0/rho",
    )
    r_parser.set_defaults(func=isentropic_flow.lookup_table_by_density)

    a_parser = subparsers.add_parser(
        "area",
        description="Compute isentropic flow outputs by area ratio",
        aliases=["A"],
        parents=[specific_heat_ratio_parser(), flow_regime_parser()],
        help="Area ratio A/A*",
    )
    a_parser.add_argument(
        "area_ratio", type=float, metavar="x.x", help="Area ratio A/A*"
    )
    a_parser.set_defaults(func=isentropic_flow.lookup_table_by_area_ratio)

    mu_parser = subparsers.add_parser(
        "mach-angle",
        description="Compute isentropic flow outputs by mach angle",
        aliases=["mu"],
        parents=[specific_heat_ratio_parser(), radians_parser()],
        help="Mach angle mu",
    )
    mu_parser.add_argument(
        "mach_angle",
        type=float,
        metavar="x.x",
        help="Mach angle mu",
    )
    mu_parser.set_defaults(func=isentropic_flow.lookup_table_by_mach_angle)

    nu_parser = subparsers.add_parser(
        "prandtl-meyer",
        description="Compute isentropic flow outputs by Prandtl-Meyer func",
        aliases=["nu", "pm"],
        parents=[specific_heat_ratio_parser(), radians_parser()],
        help="Prandtl-Meyer function nu",
    )
    nu_parser.add_argument(
        "prandtl_meyer",
        type=float,
        metavar="x.x",
        help="Prandtl-Meyer function nu",
    )
    nu_parser.set_defaults(func=isentropic_flow.lookup_table_by_prandtl_meyer)

    parsed_args = isen_parser.parse_args(args)

    kwargs = vars(parsed_args).copy()
    if not kwargs.get("radians", False):
        if "prandtl_meyer" in kwargs:
            kwargs["prandtl_meyer"] = np.radians(kwargs["prandtl_meyer"])
        elif "mach_angle" in kwargs:
            kwargs["mach_angle"] = np.radians(kwargs["mach_angle"])
    kwargs.pop("func")
    kwargs.pop("radians", None)

    res = parsed_args.func(**kwargs)

    print_line_split()
    print("MinuteMAN: Isentropic Flow Lookup Table")
    print_line_split()
    print_table_line(name="Mach Number", symbol="M", value=res.mach.item())
    print_table_line(
        name="Temperature Ratio", symbol="T0/T", value=res.temperature.item()
    )
    print_table_line(
        name="Pressure Ratio", symbol="p0/p", value=res.pressure.item()
    )
    print_table_line(
        name="Density Ratio", symbol="rho0/rho", value=res.density.item()
    )
    print_table_line(
        name="Speed of Sound Ratio",
        symbol="a0/a",
        value=res.speed_of_sound.item(),
    )
    print_table_line(
        name="Area Ratio", symbol="A/A*", value=res.area_ratio.item()
    )
    print_table_angle_line(
        name="Mach Angle",
        symbol="mu",
        val_rad=res.mach_angle.item(),
    )
    print_table_angle_line(
        name="Prandtl-Meyer Function",
        symbol="nu",
        val_rad=res.prandtl_meyer_func.item(),
    )
    print_table_line(
        name="Specific Heat Ratio",
        symbol="gamma",
        value=res.specific_heat_ratio.item(),
    )
    print_line_split()


if __name__ == "__main__":
    main()
