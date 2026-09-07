# Copyright (c) 2022-2026 Jared Magnusson
# SPDX-License-Identifier: Apache-2.0

import pytest

from minuteman.cpg import cli_oblique_shock


@pytest.mark.parametrize(
    "args",
    [
        ["5.0", "mn1", "1.2", "-g", "1.5"],
        ["2.0", "theta", "20", "-st", "weak"],
        ["12.0", "theta", "35", "--shock-type", "strong"],
        ["1.3", "beta", "1.4", "-rad"],
    ],
)
def test_main(args):
    # really just testing that the API works, logic tested elsewhere
    cli_oblique_shock.main(args)
