# Copyright (c) 2022-2026 Jared Magnusson
# SPDX-License-Identifier: Apache-2.0

import argparse
import sys
from argparse import ArgumentParser

from minuteman.cpg.base import FlowSpeedRegime


def check_zero_args(parser: ArgumentParser) -> None:
    if len(sys.argv) == 1:
        print(parser.print_help())
        sys.exit(1)


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
