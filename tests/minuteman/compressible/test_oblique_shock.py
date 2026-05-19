
import numpy as np
import pytest
import minuteman.compressible.oblique_shock as obs


@pytest.mark.parametrize("M1, beta, expected", [
    # expected values from https://devenport.aoe.vt.edu/aoe3114/calc.html
    (4, np.radians(30.0), 2.),
    (12, np.radians(60.0), 10.3923048),
])
def test_mach1_normal(M1, beta, expected):
    actual = obs.mach1_normal(M1=M1, beta=beta)
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize("M1, beta, gam, expected", [
    # expected values from https://devenport.aoe.vt.edu/aoe3114/calc.html
    (1.5, np.radians(60.0), 1.5, 1.48351648),
    (5, np.radians(33.0), 1.2, 4.68389331),
])
def test_density_ratio(M1, beta, gam, expected):
    Mn1 = obs.mach1_normal(M1=M1, beta=beta)
    actual = obs.density_ratio(Mn1=Mn1, gam=gam)
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize("M1, beta, gam, expected", [
    # expected values from https://devenport.aoe.vt.edu/aoe3114/calc.html
    (1.5, np.radians(60.0), 1.5, 1.82500000),
    (5, np.radians(33.0), 1.2, 7.99904577),
])
def test_pressure_ratio(M1, beta, gam, expected):
    Mn1 = obs.mach1_normal(M1=M1, beta=beta)
    actual = obs.pressure_ratio(Mn1=Mn1, gam=gam)
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize("M1, beta, gam, expected", [
    # expected values from https://devenport.aoe.vt.edu/aoe3114/calc.html
    (2.5, np.radians(65.0), 1.35, 0.53010844),
    (30, np.radians(18.0), 1.67, 0.45612707),
])
def test_mach2_normal(M1, beta, gam, expected):
    Mn1 = obs.mach1_normal(M1=M1, beta=beta)
    actual = obs.mach2_normal(Mn1=Mn1, gam=gam)
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize("M1, beta, gam, expected", [
    # expected values from https://devenport.aoe.vt.edu/aoe3114/calc.html
    (1.5, np.radians(60.0), 1.5, 1.23018518),
    (5, np.radians(33.0), 1.2, 1.70777710),
])
def test_temperature_ratio(M1, beta, gam, expected):
    Mn1 = obs.mach1_normal(M1=M1, beta=beta)
    actual = obs.temperature_ratio(Mn1=Mn1, gam=gam)
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize("M1, beta, gam, expected", [
    # expected values from https://devenport.aoe.vt.edu/aoe3114/calc.html
    (3.3, np.radians(35), 1.4, 2.21741000),
    (6.3, np.radians(87), 1.2, 0.35603772),
])
def test_mach2(M1, beta, gam, expected):
    Mn1 = obs.mach1_normal(M1=M1, beta=beta)
    Mn2 = obs.mach2_normal(Mn1=Mn1, gam=gam)
    theta = obs.deflection_angle(M1=M1, beta=beta, gam=gam)
    actual = obs.mach2(Mn2=Mn2, beta=beta, theta=theta)
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize("M1, beta, gam, expected", [
    # expected values from https://devenport.aoe.vt.edu/aoe3114/calc.html
    (2.0, np.radians(65.0), 1.4, np.radians(22.9704761)),
    (6.0, np.radians(15.0), 1.2, np.radians(7.85416103)),
])
def test_deflection_angle(M1, beta, gam, expected):
    assert obs.deflection_angle(
        M1=M1, beta=beta, gam=gam) == pytest.approx(expected)


@pytest.mark.parametrize("M1, theta, is_strong_shock, gam, expected", [
    # expected values from https://devenport.aoe.vt.edu/aoe3114/calc.html
    (6.0, np.radians(15.0), False, 1.2, np.radians(21.2816779)),
    (1.2, np.radians(3.0), False, 1.45, np.radians(64.6226700)),
    (3, np.radians(15.0), True, 1.45, np.radians(83.9497031)),
    (1.1, np.radians(5.0), True, 1.5, ValueError),
    (4, np.radians(89.0), False, 1.3, ValueError),
])
def test_shock_angle(M1, theta, is_strong_shock, gam, expected):
    if isinstance(expected, float):
        assert obs.shock_angle(
            M1=M1, theta=theta, is_strong_shock=is_strong_shock, gam=gam
        ) == pytest.approx(expected)
    else:
        with pytest.raises(expected):
            obs.shock_angle(
                M1=M1, theta=theta, is_strong_shock=is_strong_shock, gam=gam
            )


@pytest.mark.parametrize("theta", [
    90, 89, 0, 10, 45
])
def test_check_deflection_angle(theta):
    theta = np.radians(theta)
    obs.check_deflection_angle(theta)
    with pytest.raises(ValueError):
        obs.check_deflection_angle(np.radians(-5))
    with pytest.raises(ValueError):
        obs.check_deflection_angle(np.radians(90.1))


@pytest.mark.parametrize("beta, M", [
    (45, 2), (35, 4), (65, 1.2)
])
def test_check_shock_angle(beta, M):
    beta = np.radians(beta)
    obs.check_shock_angle(beta=beta, M=M)
    with pytest.raises(ValueError):
        obs.check_shock_angle(np.radians(5), M=2)
    with pytest.raises(ValueError):
        obs.check_shock_angle(np.radians(90.1), M=1.1)


@pytest.mark.parametrize("M1, gam", [
    (3, 1.35), (11, 1.45), (1.3, 1.6), (5, 1.4), (25, 1.25)
])
def test_max_shock_deflection_angle(M1, gam):
    actual = obs.max_shock_angle(M1=M1, gam=gam)

    # check that max_shock_angle agrees with shock_angle prediction
    #  (to numerical precision)
    theta_max = obs.max_deflection_angle(M1=M1, gam=gam)
    expected = obs.shock_angle(
        M1=M1, theta=theta_max - 1e-16, gam=gam)
    assert actual == pytest.approx(expected)

    # check that shock_angle errors out just above max
    with pytest.raises(ValueError):
        obs.shock_angle(M1=M1, theta=theta_max+1e-10, gam=gam)


@pytest.mark.parametrize("M1, gam, expected", [
    # expected values from https://www.pdas.com/flowcalc.html
    (3, 1.4, 63.76658), (11, 1.45, None),
    (1.3, 1.6, None), (5, 1.4, 66.08399), (25, 1.25, None)
])
def test_sonic_shock_angle(M1, gam, expected):
    beta = obs.sonic_shock_angle(M1=M1, gam=gam)
    if expected is not None:
        expected = expected * np.pi/180
        assert beta == pytest.approx(expected)
    # in case online calculator wrong or not gam=1.4, check downstream Mach
    mn1 = obs.mach1_normal(M1=M1, beta=beta)
    mn2 = obs.mach2_normal(Mn1=mn1, gam=gam)
    theta = obs.deflection_angle(M1=M1, beta=beta, gam=gam)
    M2 = obs.mach2(Mn2=mn2, beta=beta, theta=theta)
    assert M2 == pytest.approx(1.0)
