# Copyright (c) 2022-2026 Jared Magnusson
# SPDX-License-Identifier: Apache-2.0

import pytest

from minuteman.cpg import cli_rayleigh


@pytest.mark.parametrize(
    "args",
    [
        ["m", "5.0", "-g", "1.5"],
        ["t", "0.7", "-fr", "lowspeed"],
        ["rho", "3.0"],
        ["pressure", "1.0"],
        ["p0", "1.1", "-g", "1.41", "-fr", "subsonic"],
        ["T0", "0.85", "-fr", "supersonic"],
        ["s", "6.1"],
    ],
)
def test_main(args):
    # really just testing that the API works, logic tested elsewhere
    cli_rayleigh.main(args)
