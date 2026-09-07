# Copyright (c) 2022-2026 Jared Magnusson
# SPDX-License-Identifier: Apache-2.0

import pytest

from minuteman.cpg import cli_fanno


@pytest.mark.parametrize(
    "args",
    [
        ["m", "5.0", "-g", "1.5"],
        ["t", "0.7"],
        ["rho", "3.0"],
        ["pressure", "45.0"],
        ["p0", "2.5", "-g", "1.41", "-fr", "subsonic"],
        ["s", "6.1"],
        ["fanno-parameter", "0.6", "--flow-regime", "supersonic"],
    ],
)
def test_main(args):
    # really just testing that the API works, logic tested elsewhere
    cli_fanno.main(args)
