# Copyright (c) 2022-2026 Jared Magnusson
# SPDX-License-Identifier: Apache-2.0

import pytest

from minuteman.cpg import cli_shock_shape


@pytest.mark.parametrize(
    "args",
    [
        ["cw", "3.0", "1.0"],
        ["cylinder-wedge", "3.0", "1.0"],
        ["sc", "2.0", "0.1"],
        ["sphere-cone", "1.1", "5.0"],
    ],
)
def test_main(args):
    # really just testing that the API works, logic tested elsewhere
    cli_shock_shape.main(args)
