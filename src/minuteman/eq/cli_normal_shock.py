# Copyright (c) 2022-2026 Jared Magnusson
# SPDX-License-Identifier: Apache-2.0

"""
```bash
usage: nshock-eq [-h] velocity state_pair x.x x.x
    [--mole-fraction MOLE_FRACTION | --mass-fraction MASS_FRACTION]
    [--mechanism MECHANISM]

Normal shock table lookup for an equilibrium reacting gas (SI units)

positional arguments:
  velocity              Upstream velocity [m/s]
  state_pair            Thermodynamic property pair, choose: TP, TD, HP,
                            SP, SV, UV
  x.x                   First thermodynamic property value
  x.x                   Second thermodynamic property value

options:
  -h, --help            show this help message and exit
  --mole-fraction, -X MOLE_FRACTION
                        Mole fractions for upstream species
                            (default: N2: 0.79, O2:0.21)
  --mass-fraction, -Y MASS_FRACTION
                        Mass fractions for upstream species (default: None)
  --mechanism MECHANISM
                        Reaction mechanism (default: airNASA9.yaml)
```
"""

import argparse

from minuteman import eq
from minuteman.eq import normal_shock
from minuteman.utils.cli import (
    add_mass_fraction_parser,
    add_mole_fraction_parser,
    add_reaction_mechanism_parser,
    print_line_split,
    print_table_line,
)


def main(args: list[str] | None = None) -> None:
    nshock_parser = argparse.ArgumentParser(
        prog="nshock-eq",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Normal shock table lookup for "
        "an equilibrium reacting gas (SI units)",
    )
    nshock_parser.add_argument(
        "velocity", type=float, help="Upstream velocity [m/s]"
    )

    nshock_parser.add_argument(
        "state_pair",
        type=str,
        help="Thermodynamic state pair, choose: TP, TD, HP, SP, SV, UV",
    )
    nshock_parser.add_argument(
        "property1",
        type=float,
        metavar="x.x",
        help="First thermodynamic state variable",
    )
    nshock_parser.add_argument(
        "property2",
        type=float,
        metavar="x.x",
        help="Second thermodynamic state variable",
    )
    frac_parser = nshock_parser.add_mutually_exclusive_group(required=False)
    add_mole_fraction_parser(frac_parser)
    add_mass_fraction_parser(frac_parser)
    add_reaction_mechanism_parser(nshock_parser)

    parsed_args = nshock_parser.parse_args(args)
    states1 = eq.initialize_states(
        state_pair=parsed_args.state_pair,
        property1=parsed_args.property1,
        property2=parsed_args.property2,
        mole_fraction=parsed_args.mole_fraction,
        mass_fraction=parsed_args.mass_fraction,
        mechanism=parsed_args.mechanism,
    )
    res = normal_shock.lookup_table_by_states(
        velocity=parsed_args.velocity,
        states1=states1,
    )

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
