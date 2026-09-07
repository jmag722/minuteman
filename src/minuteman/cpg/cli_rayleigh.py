# Copyright (c) 2022-2026 Jared Magnusson
# SPDX-License-Identifier: Apache-2.0

"""
```bash
usage: rayleigh [-h] {subcommand}

Rayleigh flow table lookup for a calorically perfect gas

options:
  -h, --help            show this help message and exit

subcommands:
  choose one of the following
    mach (m, M)                Mach Number
    temperature (T, t)         Temperature ratio T/T*
    pressure (p, P)            Pressure ratio p/p*
    density (r, rho)           Density ratio rho/rho*
    total-pressure (p0)        Total pressure ratio p0/p0*
    total-temperature (T0, t0) Total temperature ratio T0/T0*
    entropy (s)                Entropy ratio (s*-s)/R
```

Running help on any of these subcommands will give further details of their
required inputs.

For instance, running `rayleigh temperature --help` will yield:

```bash
usage: rayleigh temperature [-h] [--specific-heat-ratio x.x] \
[--flow-regime <lowspeed, highspeed>] x.x

Compute Rayleigh flow outputs by temperature ratio

positional arguments:
  x.x                   Temperature ratio T/T*

options:
  -h, --help            show this help message and exit
  --specific-heat-ratio x.x, --gamma x.x, -g x.x
                        Specific heat ratio
  --flow-regime <lowspeed, highspeed>, -fr <lowspeed, highspeed>
                        Rayleigh flow regime is left or right of Tmax
                        on Rayleigh curve
```
"""

import argparse
from argparse import ArgumentParser

from minuteman.cpg import rayleigh
from minuteman.cpg.rayleigh import RayleighTemperatureRegime
from minuteman.utils.cli import (
    flow_regime_parser,
    print_line_split,
    print_table_line,
    specific_heat_ratio_parser,
)


def ray_flow_regime_parser() -> ArgumentParser:
    fr = argparse.ArgumentParser(add_help=False)
    fr.add_argument(
        "--flow-regime",
        "-fr",
        type=lambda x: rayleigh.RayleighTemperatureRegime[x.lower()],
        choices=RayleighTemperatureRegime,
        default=RayleighTemperatureRegime.highspeed,
        metavar="<lowspeed, highspeed>",
        help="Rayleigh flow regime is left or right of Tmax on Rayleigh curve",
    )
    return fr


def main(args: list[str] | None = None) -> None:
    ray_parser = argparse.ArgumentParser(
        prog="rayleigh",
        description="Rayleigh flow table lookup for a calorically perfect gas",
    )
    subparsers = ray_parser.add_subparsers(
        description="choose one of the following", required=True
    )

    mach_parser = subparsers.add_parser(
        "mach",
        description="Compute Rayleigh flow outputs by Mach",
        aliases=["m", "M"],
        parents=[specific_heat_ratio_parser()],
        help="Mach Number",
    )
    mach_parser.add_argument(
        "mach", type=float, metavar="x.x", help="Mach number"
    )
    mach_parser.set_defaults(func=rayleigh.lookup_table_by_mach)

    t_parser = subparsers.add_parser(
        "temperature",
        description="Compute Rayleigh flow outputs by temperature ratio",
        aliases=["T", "t"],
        parents=[specific_heat_ratio_parser(), ray_flow_regime_parser()],
        help="Temperature ratio T/T*",
    )
    t_parser.add_argument(
        "temperature_ratio",
        type=float,
        metavar="x.x",
        help="Temperature ratio T/T*",
    )
    t_parser.set_defaults(func=rayleigh.lookup_table_by_temperature)

    p_parser = subparsers.add_parser(
        "pressure",
        description="Compute Rayleigh flow outputs by pressure ratio",
        aliases=["p", "P"],
        parents=[specific_heat_ratio_parser()],
        help="Pressure ratio p/p*",
    )
    p_parser.add_argument(
        "pressure_ratio", type=float, metavar="x.x", help="Pressure ratio p/p*"
    )
    p_parser.set_defaults(func=rayleigh.lookup_table_by_pressure)

    r_parser = subparsers.add_parser(
        "density",
        description="Compute Rayleigh flow outputs by density ratio",
        aliases=["r", "rho"],
        parents=[specific_heat_ratio_parser()],
        help="Density ratio rho/rho*",
    )
    r_parser.add_argument(
        "density_ratio",
        type=float,
        metavar="x.x",
        help="Density ratio rho/rho*",
    )
    r_parser.set_defaults(func=rayleigh.lookup_table_by_density)

    p0_parser = subparsers.add_parser(
        "total-pressure",
        description="Compute Rayleigh flow outputs by total pressure ratio",
        aliases=["p0"],
        parents=[specific_heat_ratio_parser(), flow_regime_parser()],
        help="Total pressure ratio p0/p0*",
    )
    p0_parser.add_argument(
        "total_pressure_ratio",
        type=float,
        metavar="x.x",
        help="Total pressure ratio p0/p0*",
    )
    p0_parser.set_defaults(func=rayleigh.lookup_table_by_total_pressure)

    t0_parser = subparsers.add_parser(
        "total-temperature",
        description="Compute Rayleigh flow outputs by total temperature ratio",
        aliases=["T0", "t0"],
        parents=[specific_heat_ratio_parser(), flow_regime_parser()],
        help="Total temperature ratio T0/T0*",
    )
    t0_parser.add_argument(
        "total_temperature_ratio",
        type=float,
        metavar="x.x",
        help="Total temperature ratio T0/T0*",
    )
    t0_parser.set_defaults(func=rayleigh.lookup_table_by_total_temperature)

    s_parser = subparsers.add_parser(
        "entropy",
        description="Compute Rayleigh flow outputs by entropy ratio",
        aliases=["s"],
        parents=[specific_heat_ratio_parser(), flow_regime_parser()],
        help="Entropy ratio (s*-s)/R",
    )
    s_parser.add_argument(
        "entropy_ratio",
        type=float,
        metavar="x.x",
        help="Entropy ratio, (s*-s)/R",
    )
    s_parser.set_defaults(func=rayleigh.lookup_table_by_entropy)

    parsed_args = ray_parser.parse_args(args)

    kwargs = vars(parsed_args).copy()
    kwargs.pop("func")

    res = parsed_args.func(**kwargs)

    print_line_split()
    print("MinuteMAN: Rayleigh Flow Lookup Table")
    print_line_split()
    print_table_line(name="Mach Number", symbol="M", value=res.mach.item())
    print_table_line(
        name="Temperature Ratio",
        symbol="T/T*",
        value=res.temperature_ratio.item(),
    )
    print_table_line(
        name="Pressure Ratio", symbol="p/p*", value=res.pressure_ratio.item()
    )
    print_table_line(
        name="Density Ratio", symbol="rho/rho*", value=res.density_ratio.item()
    )
    print_table_line(
        name="Total Pressure Ratio",
        symbol="p0/p0*",
        value=res.total_pressure_ratio.item(),
    )
    print_table_line(
        name="Total Temperature Ratio",
        symbol="T0/T0*",
        value=res.total_temperature_ratio.item(),
    )
    print_table_line(
        name="Entropy Ratio",
        symbol="(s*-s)/R",
        value=res.entropy_ratio.item(),
    )
    print_table_line(
        name="Specific Heat Ratio",
        symbol="gamma",
        value=res.specific_heat_ratio.item(),
    )
    print_line_split()


if __name__ == "__main__":
    main()
