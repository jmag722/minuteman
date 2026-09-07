# Copyright (c) 2022-2026 Jared Magnusson
# SPDX-License-Identifier: Apache-2.0

"""
```bash
usage: fanno [-h] {subcommand}

Fanno flow table lookup for a calorically perfect gas

options:
  -h, --help            show this help message and exit

subcommands:
  choose one of the following
    mach (m, M)            Mach Number
    temperature (T, t)     Temperature ratio T/T*
    pressure (p, P)        Pressure ratio p/p*
    density (r, rho)       Density ratio rho/rho*
    total-pressure (p0)    Total pressure ratio p0/p0*
    fanno-parameter (fp)   Fanno parameter 4fL*/D
    entropy (s)            Entropy ratio (s*-s)/R
```

Running help on any of these subcommands will give further details of their
required inputs.

For instance, running `fanno entropy --help` will yield:

```bash
usage: fanno entropy [-h] [--specific-heat-ratio x.x] \
[--flow-regime <subsonic, supersonic>] x.x

Compute Fanno flow outputs by entropy ratio

positional arguments:
  x.x                   Entropy ratio (s*-s)/R

options:
  -h, --help            show this help message and exit
  --specific-heat-ratio x.x, --gamma x.x, -g x.x
                        Specific heat ratio
  --flow-regime <subsonic, supersonic>, -fr <subsonic, supersonic>
                        Flow regime is subsonic or supersonic
```
"""

import argparse

from minuteman.cpg import fanno
from minuteman.utils.cli import (
    flow_regime_parser,
    print_line_split,
    print_table_line,
    specific_heat_ratio_parser,
)


def main(args: list[str] | None = None) -> None:
    fanno_parser = argparse.ArgumentParser(
        prog="fanno",
        description="Fanno flow table lookup for a calorically perfect gas",
    )
    subparsers = fanno_parser.add_subparsers(
        description="choose one of the following", required=True
    )

    mach_parser = subparsers.add_parser(
        "mach",
        description="Compute Fanno flow outputs by Mach",
        aliases=["m", "M"],
        parents=[specific_heat_ratio_parser()],
        help="Mach Number",
    )
    mach_parser.add_argument(
        "mach", type=float, metavar="x.x", help="Mach number"
    )
    mach_parser.set_defaults(func=fanno.lookup_table_by_mach)

    t_parser = subparsers.add_parser(
        "temperature",
        description="Compute Fanno flow outputs by temperature ratio",
        aliases=["T", "t"],
        parents=[specific_heat_ratio_parser()],
        help="Temperature ratio T/T*",
    )
    t_parser.add_argument(
        "temperature_ratio",
        type=float,
        metavar="x.x",
        help="Temperature ratio T/T*",
    )
    t_parser.set_defaults(func=fanno.lookup_table_by_temperature)

    p_parser = subparsers.add_parser(
        "pressure",
        description="Compute Fanno flow outputs by pressure ratio",
        aliases=["p", "P"],
        parents=[specific_heat_ratio_parser()],
        help="Pressure ratio p/p*",
    )
    p_parser.add_argument(
        "pressure_ratio", type=float, metavar="x.x", help="Pressure ratio p/p*"
    )
    p_parser.set_defaults(func=fanno.lookup_table_by_pressure)

    r_parser = subparsers.add_parser(
        "density",
        description="Compute Fanno flow outputs by density ratio",
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
    r_parser.set_defaults(func=fanno.lookup_table_by_density)

    p0_parser = subparsers.add_parser(
        "total-pressure",
        description="Compute Fanno flow outputs by total pressure ratio",
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
    p0_parser.set_defaults(func=fanno.lookup_table_by_total_pressure)

    fp_parser = subparsers.add_parser(
        "fanno-parameter",
        description="Compute Fanno flow outputs by Fanno parameter",
        aliases=["fp"],
        parents=[specific_heat_ratio_parser(), flow_regime_parser()],
        help="Fanno parameter 4fL*/D",
    )
    fp_parser.add_argument(
        "fanno_parameter",
        type=float,
        metavar="x.x",
        help="Fanno parameter 4fL*/D",
    )
    fp_parser.set_defaults(func=fanno.lookup_table_by_fanno_parameter)

    s_parser = subparsers.add_parser(
        "entropy",
        description="Compute Fanno flow outputs by entropy ratio",
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
    s_parser.set_defaults(func=fanno.lookup_table_by_entropy)

    parsed_args = fanno_parser.parse_args(args)

    kwargs = vars(parsed_args).copy()
    kwargs.pop("func")

    res = parsed_args.func(**kwargs)

    print_line_split()
    print("MinuteMAN: Fanno Flow Lookup Table")
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
        name="Fanno Parameter",
        symbol="4fL*/D",
        value=res.fanno_parameter.item(),
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
