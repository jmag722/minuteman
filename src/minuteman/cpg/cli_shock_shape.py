# Copyright (c) 2022-2026 Jared Magnusson
# SPDX-License-Identifier: Apache-2.0

"""
```bash
usage: shock-standoff [-h] {subcommand}

Shock standoff distance calculator for a calorically perfect gas

options:
  -h, --help            show this help message and exit

subcommands:
  choose one of the following
    cylinder-wedge (cw) Compute shock standoff distance for a cylinder-wedge
    sphere-cone (sc)    Compute shock standoff distance for a sphere-cone
```

Running help on any of these subcommands will give further details of their
required inputs.

For instance, running `shock-standoff cylinder-wedge --help` will yield:

```bash
usage: shock-standoff cylinder-wedge [-h] x.x x.x

Compute shock standoff distance for a cylinder-wedge

positional arguments:
  x.x         Upstream Mach number
  x.x         Cylinder radius

options:
  -h, --help  show this help message and exit
```
"""

import argparse

from minuteman.cpg import shock_shape
from minuteman.utils.cli import (
    print_line_split,
    print_table_line,
)


def main(args: list[str] | None = None) -> None:
    sshape_parser = argparse.ArgumentParser(
        prog="shock-standoff",
        description="Shock standoff distance calculator for a "
                    "calorically perfect gas",
    )

    subparsers = sshape_parser.add_subparsers(
        description="choose one of the following", required=True
    )

    cw_parser = subparsers.add_parser(
        "cylinder-wedge",
        description="Compute shock standoff distance for a cylinder-wedge",
        aliases=["cw"],
        help="Compute shock standoff distance for a cylinder-wedge",
    )
    cw_parser.add_argument(
        "mach",
        type=float,
        metavar="x.x",
        help="Upstream Mach number",
    )
    cw_parser.add_argument(
        "nose_radius",
        type=float,
        metavar="x.x",
        help="Cylinder radius",
    )

    def cw_func(mach, nose_radius):
        sd = shock_shape.standoff_distance_cylinder_wedge(
            mach=mach, nose_radius=nose_radius
        )
        rc = shock_shape.curvature_radius_cylinder_wedge(
            mach=mach, nose_radius=nose_radius
        )
        typ = "Cylinder-Wedge"
        return (sd, rc, typ)

    cw_parser.set_defaults(func=cw_func)

    sc_parser = subparsers.add_parser(
        "sphere-cone",
        description="Compute shock standoff distance for a sphere-cone",
        aliases=["sc"],
        help="Compute shock standoff distance for a sphere-cone",
    )
    sc_parser.add_argument(
        "mach",
        type=float,
        metavar="x.x",
        help="Upstream Mach number",
    )
    sc_parser.add_argument(
        "nose_radius",
        type=float,
        metavar="x.x",
        help="Cylinder radius",
    )

    def sc_func(mach, nose_radius):
        sd = shock_shape.standoff_distance_sphere_cone(
            mach=mach, nose_radius=nose_radius
        )
        rc = shock_shape.curvature_radius_sphere_cone(
            mach=mach, nose_radius=nose_radius
        )
        typ = "Sphere-Cone"
        return (sd, rc, typ)

    sc_parser.set_defaults(func=sc_func)

    parsed_args = sshape_parser.parse_args(args)
    kwargs = vars(parsed_args).copy()
    kwargs.pop("func")
    sd, rc, typ = parsed_args.func(**kwargs)

    print_line_split()
    print(f"MinuteMAN: Shock Standoff Distance Calculator ({typ})")
    print_line_split()

    print_table_line(name="Upstream Mach", symbol="M1", value=parsed_args.mach)
    print_table_line(
        name="Nose Radius", symbol="R", value=parsed_args.nose_radius
    )
    print_table_line(
        name="Standoff Distance",
        symbol="Delta",
        value=sd.item(),
    )
    print_table_line(
        name="Radius of Curvature",
        symbol="Rc",
        value=rc.item(),
    )
    print_line_split()


if __name__ == "__main__":
    main()
