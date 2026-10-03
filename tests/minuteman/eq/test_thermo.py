# Copyright (c) 2022-2026 Jared Magnusson
# SPDX-License-Identifier: Apache-2.0

import cantera as ct
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as sc

from minuteman.eq import thermo


def test_specific_heat_ratio_hansen():
    """
    Test taken from Table IV.f of
    Hansen, C. F. (1959). Approximations for the thermodynamic and
    transport properties of high-temperature air. National Aeronautics
    and Space Administration.

    The error is really low in some areas (see plot below) but in other
    temperature bands it's much worse, exceeding 3%. I believe this is due
    to approximations made by Hansen at the time, which caused his sound speed
    to overpredict those here.

    Hansen's values are really just specific heat ratio, so this test works
    for both
    """
    gas = ct.Solution("airNASA9.yaml")
    # convert pressure from atm to Pa
    p1d = np.array([1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4]) * sc.atm
    t1d = np.linspace(5e2, 15e3, num=30)  # Kelvin
    t, p = np.meshgrid(t1d, p1d, indexing="ij")
    x = "N2:0.79, O2:0.21"
    states = ct.SolutionArray(gas, shape=p.shape)
    states.TPX = t, p, x
    states.equilibrate("TP")

    actual = thermo.specific_heat_ratio(states)
    expected = [
        [1.387, 1.387, 1.387, 1.387, 1.387, 1.387, 1.387],
        [1.337, 1.337, 1.337, 1.337, 1.337, 1.337, 1.337],
        [1.312, 1.312, 1.312, 1.312, 1.312, 1.310, 1.306],
        [1.300, 1.299, 1.296, 1.285, 1.259, 1.206, 1.144],
        [1.288, 1.276, 1.247, 1.196, 1.144, 1.119, 1.132],
        [1.265, 1.229, 1.181, 1.147, 1.150, 1.232, 1.302],
        [1.234, 1.192, 1.166, 1.187, 1.265, 1.250, 1.171],
        [1.212, 1.187, 1.204, 1.254, 1.208, 1.137, 1.097],
        [1.208, 1.210, 1.241, 1.196, 1.133, 1.101, 1.092],
        [1.217, 1.233, 1.202, 1.143, 1.111, 1.103, 1.124],
        [1.229, 1.217, 1.161, 1.124, 1.113, 1.133, 1.249],
        [1.230, 1.186, 1.141, 1.124, 1.135, 1.225, 1.263],
        [1.214, 1.165, 1.137, 1.136, 1.193, 1.260, 1.183],
        [1.195, 1.155, 1.142, 1.167, 1.249, 1.205, 1.140],
        [1.182, 1.154, 1.156, 1.216, 1.231, 1.162, 1.128],
        [1.175, 1.159, 1.181, 1.242, 1.193, 1.143, 1.130],
        [1.174, 1.169, 1.214, 1.228, 1.168, 1.140, 1.136],
        [1.176, 1.185, 1.237, 1.203, 1.157, 1.144, 1.145],
        [1.181, 1.206, 1.237, 1.185, 1.155, 1.151, 1.157],
        [1.190, 1.228, 1.225, 1.176, 1.159, 1.159, 1.185],
        [1.201, 1.242, 1.212, 1.173, 1.165, 1.169, 1.244],
        [1.216, 1.246, 1.202, 1.175, 1.173, 1.187, 1.334],
        [1.232, 1.242, 1.198, 1.180, 1.181, 1.221, 1.424],
        [1.247, 1.236, 1.197, 1.187, 1.192, 1.276, 1.489],
        [1.257, 1.231, 1.198, 1.194, 1.207, 1.346, 1.528],
        [1.263, 1.227, 1.202, 1.202, 1.232, 1.415, 1.548],
        [1.265, 1.226, 1.208, 1.210, 1.269, 1.470, 1.558],
        [1.265, 1.226, 1.214, 1.221, 1.317, 1.508, 1.563],
        [1.264, 1.228, 1.221, 1.235, 1.370, 1.532, 1.565],
        [1.263, 1.232, 1.228, 1.254, 1.420, 1.546, 1.567],
    ]
    expected = np.asarray(expected)
    plt_check = False
    if plt_check:
        for i, p1 in enumerate(p1d):
            plt.plot(
                t1d,
                actual[:, i],
                ls="--",
                c="r",
                label=f"{p1 / sc.atm}",
            )
            plt.plot(t1d, expected[:, i], ls="-", c="k")
            plt.legend()
            plt.show()
    np.testing.assert_allclose(actual, expected, rtol=3.3e-2)


def test_sound_speed_equilibrium_sdtoolbox():
    """
    Tested against SDToolbox 11-Feb-2023. Pressure-temperature values are
    for those regions where agreement with Hansen's values above were poor.
    """
    gas = ct.Solution("airNASA9.yaml")
    t = np.array([2e3, 14e3, 8e3])
    p = np.array([1e2, 1e1, 1e-4]) * sc.atm
    x = "N2:0.79, O2:0.21"
    states = ct.SolutionArray(gas, shape=p.shape)
    states.TPX = t, p, x
    states.equilibrate("TP")

    actual = thermo.speed_of_sound(states)
    expected = np.array(
        [858.0973021541372, 3292.5675719603146, 2499.8286326468346]
    )
    # we can match SDToolbox exactly if our "frac" is 1e-2 above/below
    # but we used a smaller value
    np.testing.assert_allclose(actual, expected, rtol=1.65e-5)
