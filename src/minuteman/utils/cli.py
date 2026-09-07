# Copyright (c) 2022-2026 Jared Magnusson
# SPDX-License-Identifier: Apache-2.0

import argparse
from argparse import ArgumentParser

import numpy as np

from minuteman.cpg.base import FlowSpeedRegime
from minuteman.cpg.oblique_shock import ObliqueShockType

default_name_width = 23
default_sym_width = 8
default_value_width = 18
default_decimal_width = 9
line_split = "-" * 79


def print_line_split() -> None:
    print(line_split)


def _get_name_fmt(name_width: int) -> str:
    return f"<{name_width}"


def _get_sym_fmt(sym_width: int) -> str:
    return f">{sym_width}"


def _get_num_fmt(value_width: int, decimal_width: int) -> str:
    return f">{value_width}.{decimal_width}g"


def print_table_line(
    name: str,
    symbol: str,
    value: float,
    unit: str = "",
    name_width: int = default_name_width,
    sym_width: int = default_sym_width,
    value_width: int = default_value_width,
    decimal_width: int = default_decimal_width,
) -> None:
    unit_str = f"[{unit}]" if unit else unit
    name_fmt = _get_name_fmt(name_width)
    sym_fmt = _get_sym_fmt(sym_width)
    num_fmt = _get_num_fmt(
        value_width=value_width, decimal_width=decimal_width
    )
    print(
        f"{name:{name_fmt}}{symbol:{sym_fmt}}{value:{num_fmt}}{unit_str}",
    )


def print_table_angle_line(
    name: str,
    symbol: str,
    val_rad: float,
    name_width: int = default_name_width,
    sym_width: int = default_sym_width,
    value_width: int = default_value_width,
    decimal_width: int = default_decimal_width,
) -> None:
    val_deg = np.degrees(val_rad)
    name_fmt = _get_name_fmt(name_width)
    sym_fmt = _get_sym_fmt(sym_width)
    num_fmt = _get_num_fmt(
        value_width=value_width, decimal_width=decimal_width
    )
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
        metavar="<subsonic, supersonic>",
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


def shock_type_parser() -> ArgumentParser:
    st = argparse.ArgumentParser(add_help=False)
    st.add_argument(
        "--shock-type",
        "-st",
        type=lambda x: ObliqueShockType[x.lower()],
        choices=ObliqueShockType,
        default=ObliqueShockType.weak,
        metavar="<weak, strong>",
        help="Oblique shock is weak or strong",
    )
    return st
