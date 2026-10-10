# Copyright (c) 2022-2026 Jared Magnusson
# SPDX-License-Identifier: Apache-2.0

"""
```bash
usage: nshock-eq [-h] x.x {subcommand}

Normal shock table lookup for an equilibrium reacting gas

positional arguments:
  x.x                   Upstream velocity [m/s]

options:
  -h, --help            show this help message and exit

subcommands:
  choose one of the following

    pressure-temperature (pt, PT, pT)    pressure & temperature
    density-temperature (rt, rhoT, rhot) density & temperature
```

Running help on any of these subcommands will give further details of their
required inputs.

For instance, running `nshock-eq 1e3 PT --help` will yield:

```bash
usage: nshock-eq x.x pressure-temperature [-h]
       [--mole-fraction MOLE_FRACTION] [--mechanism MECHANISM] x.x x.x

Compute normal shock outputs by pressure and temperature

positional arguments:
  x.x                   Upstream pressure [Pa]
  x.x                   Upstream temperature [K]

options:
  -h, --help            show this help message and exit
  --mole-fraction, -X MOLE_FRACTION Mole fractions for upstream species
  --mechanism MECHANISM Reaction mechanism
```
"""

import argparse

from minuteman.eq import normal_shock
from minuteman.utils.cli import (
    mole_fraction_parser,
    print_line_split,
    print_table_line,
    reaction_mechanism_parser,
)


def main(args: list[str] | None = None) -> None:
    nshock_parser = argparse.ArgumentParser(
        prog="nshock-eq",
        description="Normal shock table lookup for "
        "an equilibrium reacting gas",
    )
    nshock_parser.add_argument(
        "velocity", type=float, metavar="x.x", help="Upstream velocity [m/s]"
    )

    subparsers = nshock_parser.add_subparsers(
        description="choose one of the following", required=True
    )

    pt_parser = subparsers.add_parser(
        "pressure-temperature",
        description="Compute normal shock outputs by pressure and temperature",
        aliases=["pt", "PT", "pT"],
        parents=[mole_fraction_parser(), reaction_mechanism_parser()],
        help="pressure & temperature",
    )
    pt_parser.add_argument(
        "pressure", type=float, metavar="x.x", help="Upstream pressure [Pa]"
    )
    pt_parser.add_argument(
        "temperature",
        type=float,
        metavar="x.x",
        help="Upstream temperature [K]",
    )
    pt_parser.set_defaults(
        func=normal_shock.lookup_table_by_pressure_temperature
    )

    rt_parser = subparsers.add_parser(
        "density-temperature",
        description="Compute normal shock outputs by density and temperature",
        aliases=["rt", "rhoT", "rhot"],
        parents=[mole_fraction_parser(), reaction_mechanism_parser()],
        help="density & temperature",
    )
    rt_parser.add_argument(
        "density", type=float, metavar="x.x", help="Upstream density [kg/m3]"
    )
    rt_parser.add_argument(
        "temperature",
        type=float,
        metavar="x.x",
        help="Upstream temperature [K]",
    )
    rt_parser.set_defaults(
        func=normal_shock.lookup_table_by_density_temperature
    )

    parsed_args = nshock_parser.parse_args(args)

    kwargs = vars(parsed_args).copy()
    kwargs.pop("func")
    res = parsed_args.func(**kwargs)

    sym_width = 9
    print_line_split()
    print("MinuteMAN: Equilibrium Normal Shock Lookup Table")
    print_line_split()

    print("Upstream Conditions")
    print_line_split()
    print_table_line(
        name="Mach Number",
        symbol="M1",
        value=res.upstream.mach.item(),
        sym_width=sym_width,
    )
    print_table_line(
        name="Velocity",
        symbol="u1",
        value=res.upstream.velocity.item(),
        sym_width=sym_width,
        unit="m/s",
    )
    print_table_line(
        name="Pressure",
        symbol="p1",
        value=res.upstream.pressure.item(),
        sym_width=sym_width,
        unit="Pa",
    )
    print_table_line(
        name="Temperature",
        symbol="T1",
        value=res.upstream.temperature.item(),
        sym_width=sym_width,
        unit="K",
    )
    print_table_line(
        name="Density",
        symbol="rho1",
        value=res.upstream.density.item(),
        sym_width=sym_width,
        unit="kg/m3",
    )
    print_table_line(
        name="Eq. Sound Speed",
        symbol="a1_eq",
        value=res.upstream.speed_of_sound_eq.item(),
        sym_width=sym_width,
        unit="m/s",
    )
    print_table_line(
        name="Frozen Sound Speed",
        symbol="a1_f",
        value=res.upstream.speed_of_sound_fr.item(),
        sym_width=sym_width,
        unit="m/s",
    )
    print_table_line(
        name="Isentropic Exponent",
        symbol="gamma_s1",
        value=res.upstream.isentropic_exponent.item(),
        sym_width=sym_width,
    )
    print_line_split()

    print("Downstream Conditions")
    print_line_split()
    print_table_line(
        name="Mach Number",
        symbol="M2",
        value=res.downstream.mach.item(),
        sym_width=sym_width,
    )
    print_table_line(
        name="Velocity",
        symbol="u2",
        value=res.downstream.velocity.item(),
        sym_width=sym_width,
        unit="m/s",
    )
    print_table_line(
        name="Pressure",
        symbol="p2",
        value=res.downstream.pressure.item(),
        sym_width=sym_width,
        unit="Pa",
    )
    print_table_line(
        name="Temperature",
        symbol="T2",
        value=res.downstream.temperature.item(),
        sym_width=sym_width,
        unit="K",
    )
    print_table_line(
        name="Density",
        symbol="rho2",
        value=res.downstream.density.item(),
        sym_width=sym_width,
        unit="kg/m3",
    )
    print_table_line(
        name="Eq. Sound Speed",
        symbol="a2_eq",
        value=res.downstream.speed_of_sound_eq.item(),
        sym_width=sym_width,
        unit="m/s",
    )
    print_table_line(
        name="Frozen Sound Speed",
        symbol="a2_f",
        value=res.downstream.speed_of_sound_fr.item(),
        sym_width=sym_width,
        unit="m/s",
    )
    print_table_line(
        name="Isentropic Exponent",
        symbol="gamma_s2",
        value=res.downstream.isentropic_exponent.item(),
        sym_width=sym_width,
    )
    print_line_split()

    print("Ratios")
    print_line_split()
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
        name="Sound Speed Ratio",
        symbol="a2/a1",
        value=res.speed_of_sound_ratio.item(),
        sym_width=sym_width,
    )
    print_line_split()


if __name__ == "__main__":
    main()
