# Copyright (c) 2022-2026 Jared Magnusson
# SPDX-License-Identifier: Apache-2.0

import pytest

from minuteman.cpg import cli_normal_shock


@pytest.mark.parametrize(
    "args",
    [
        ["mach-upstream", "5.0", "-g", "1.5"],
        ["m2", "0.9", "-g", "1.5"],
        ["t", "5.0"],
        ["rho", "3.0"],
        ["pressure", "45.0"],
        ["p0", "0.5", "-g", "1.41"],
        ["pitot-pressure", "6.1"],
    ],
)
def test_main(args):
    # really just testing that the API works, logic tested elsewhere
    cli_normal_shock.main(args)
