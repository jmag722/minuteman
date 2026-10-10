# Copyright (c) 2022-2026 Jared Magnusson
# SPDX-License-Identifier: Apache-2.0

import pytest

from minuteman.eq import cli_normal_shock


@pytest.mark.parametrize(
    "args",
    [
        ["2e3", "pt", "50.0", "320.0"],
        ["4e3", "rhoT", "0.05", "420.0"],
    ],
)
def test_main(args):
    # really just testing that the API works, logic tested elsewhere
    cli_normal_shock.main(args)
