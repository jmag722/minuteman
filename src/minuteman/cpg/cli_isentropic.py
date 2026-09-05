# Copyright (c) 2022-2026 Jared Magnusson
# SPDX-License-Identifier: Apache-2.0

import argparse

import numpy as np

from minuteman.cpg import isentropic_flow
from minuteman.utils.cli import (
    check_zero_args,
    flow_regime_parser,
    radians_parser,
    specific_heat_ratio_parser,
)


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="isen",
        description="Isentropic flow table lookup for a "
        "calorically perfect gas",
    )
    subparsers = parser.add_subparsers(help="subcommands")

    mach_parser = subparsers.add_parser(
        "mach",
        aliases=["m"],
        parents=[specific_heat_ratio_parser()],
        help="Mach Number",
    )
    mach_parser.add_argument(
        "mach", type=float, metavar="x.x", help="Mach number"
    )
    mach_parser.set_defaults(func=isentropic_flow.lookup_table_by_mach)

    t_parser = subparsers.add_parser(
        "temperature",
        aliases=["T"],
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
        aliases=["p"],
        parents=[specific_heat_ratio_parser()],
        help="Pressure ratio p0/p",
    )
    p_parser.add_argument(
        "pressure_ratio", type=float, metavar="x.x", help="Pressure ratio p0/p"
    )
    p_parser.set_defaults(func=isentropic_flow.lookup_table_by_pressure)

    r_parser = subparsers.add_parser(
        "density",
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
        "prandtl-meyer-function",
        aliases=["nu"],
        parents=[specific_heat_ratio_parser(), radians_parser()],
        help="Prandtl-Meyer function nu",
    )
    nu_parser.add_argument(
        "prandtl_meyer_func",
        type=float,
        metavar="x.x",
        help="Prandtl-Meyer function nu",
    )
    nu_parser.set_defaults(func=isentropic_flow.lookup_table_by_prandtl_meyer)

    check_zero_args(parser)

    parsed_args = parser.parse_args(args)

    kwargs = vars(parsed_args).copy()
    if not kwargs.get("radians", False):
        if "prandtl_meyer_func" in kwargs:
            kwargs["prandtl_meyer_func"] = np.radians(
                kwargs["prandtl_meyer_func"]
            )
        elif "mach_angle" in kwargs:
            kwargs["mach_angle"] = np.radians(kwargs["mach_angle"])
    kwargs.pop("func")
    kwargs.pop("radians", None)

    res = parsed_args.func(**kwargs)

    key_width = 23
    sym_width = 8
    value_width = 18
    decimals_width = 9
    line_split = "-" * 79
    key_fmt = f"<{key_width}"
    sym_fmt = f">{sym_width}"
    num_fmt = f">{value_width}.{decimals_width}g"
    print(line_split)
    print("MinuteMAN: Isentropic Flow Lookup Table")
    print(line_split)
    print(
        f"{'Mach Number':{key_fmt}}{'M':{sym_fmt}}{res.mach.item():{num_fmt}}"
    )
    print(
        f"{'Temperature Ratio':{key_fmt}}"
        f"{'T0/T':{sym_fmt}}"
        f"{res.temperature.item():{num_fmt}}"
    )
    print(
        f"{'Pressure Ratio':{key_fmt}}"
        f"{'p0/p':{sym_fmt}}"
        f"{res.pressure.item():{num_fmt}}"
    )
    print(
        f"{'Density Ratio':{key_fmt}}"
        f"{'rho0/rho':{sym_fmt}}"
        f"{res.density.item():{num_fmt}}"
    )
    print(
        f"{'Speed of Sound Ratio':{key_fmt}}"
        f"{'a0/a':{sym_fmt}}"
        f"{res.speed_of_sound.item():{num_fmt}}"
    )
    print(
        f"{'Area Ratio':{key_fmt}}"
        f"{'A/A*':{sym_fmt}}"
        f"{res.area_ratio.item():{num_fmt}}"
    )
    print(
        f"{'Mach Angle':{key_fmt}}"
        f"{'mu':{sym_fmt}}"
        f"{np.degrees(res.mach_angle.item()):{num_fmt}} [deg]"
        f"{res.mach_angle.item():{num_fmt}} [rad]"
    )
    print(
        f"{'Prandtl-Meyer function':{key_fmt}}"
        f"{'nu':{sym_fmt}}"
        f"{np.degrees(res.prandtl_meyer_func.item()):{num_fmt}} [deg]"
        f"{res.prandtl_meyer_func.item():{num_fmt}} [rad]"
    )
    print(
        f"{'Specific Heat Ratio':{key_fmt}}"
        f"{'gamma':{sym_fmt}}"
        f"{res.specific_heat_ratio.item():{num_fmt}}"
    )
    print(line_split)


if __name__ == "__main__":
    main()
