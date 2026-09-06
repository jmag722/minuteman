# Copyright (c) 2022-2026 Jared Magnusson
# SPDX-License-Identifier: Apache-2.0

import argparse
from argparse import ArgumentParser

import numpy as np

from minuteman.cpg.base import FlowSpeedRegime

name_width = 23
sym_width = 8
value_width = 18
decimals_width = 9
line_split = "-" * 79

name_fmt = f"<{name_width}"
sym_fmt = f">{sym_width}"
num_fmt = f">{value_width}.{decimals_width}g"


def print_line_split() -> None:
    print(line_split)


def print_table_line(
    name: str,
    symbol: str,
    value: float,
    unit: str = "",
) -> None:
    unit_str = f"[{unit}]" if unit else unit
    print(
        f"{name:{name_fmt}}{symbol:{sym_fmt}}{value:{num_fmt}}{unit_str}",
    )


def print_table_angle_line(
    name: str,
    symbol: str,
    val_rad: float,
) -> None:
    val_deg = np.degrees(val_rad)
    print(
        f"{name:{name_fmt}}{symbol:{sym_fmt}}"
        f"{val_deg:{num_fmt}}[deg]{val_rad:{num_fmt}}[rad]",
    )


def specific_heat_ratio_parser() -> ArgumentParser:
    gam = argparse.ArgumentParser(add_help=False)
    gam.add_argument(
        "--specific-heat-ratio",
        "--gamma",
        "-g",
        type=float,
        default=1.4,
        metavar="x.x",
        help="Specific heat ratio",
    )
    return gam


def flow_regime_parser() -> ArgumentParser:
    fr = argparse.ArgumentParser(add_help=False)
    fr.add_argument(
        "--flow-regime",
        "-fr",
        type=lambda x: FlowSpeedRegime[x.lower()],
        choices=FlowSpeedRegime,
        default=FlowSpeedRegime.supersonic,
        help="Flow regime is subsonic or supersonic",
    )
    return fr


def radians_parser() -> ArgumentParser:
    rad = argparse.ArgumentParser(add_help=False)
    rad.add_argument(
        "--radians",
        "-rad",
        action="store_true",
        help="Angle is specified in radians (default is degrees)",
    )
    return rad
