# Copyright (c) 2022-2026 Jared Magnusson
# SPDX-License-Identifier: Apache-2.0

import pytest

from minuteman.cpg import cli_isentropic


@pytest.mark.parametrize(
    "args",
    [
        ["mach", "5.0", "-g", "1.5"],
        ["t", "5.0"],
        ["rho", "3.0"],
        ["area", "2.0", "--flow-regime", "supersonic"],
        ["A", "2.0", "-fr", "subsonic"],
        ["mach-angle", "45.0"],
        ["prandtl-meyer", "45.0"],
    ],
)
def test_main(args):
    # really just testing that the API works, logic tested elsewhere
    cli_isentropic.main(args)
