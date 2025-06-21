import pytest
import minuteman.compressible.isentropic as isen


def test_speed_sound():
    actual = isen.speed_sound(T=300, R=287, gam=1.4)
    expected = 347.188709494
    assert actual == pytest.approx(expected)
    actual = isen.speed_sound(gam=1.1, p=1e5, rho=1.2)
    expected = 302.76503541
    assert actual == pytest.approx(expected)


def test_mach():
    assert isen.mach(1, 4) == pytest.approx(0.25)


def test_lookup_table_mach(check_dicts):
    actual = isen.lookup_table(M=2.0, gam=1.4)
    expected = {
        "M": 2.0, "p0_ratio": 7.824, "T0_ratio": 1.8,
        "r0_ratio": 4.347, "a0_ratio": 1.342,
        "area_ratio": 1.687, "gam": 1.4
    }
    check_dicts(actual, expected, rtol=1e-3)


def test_lookup_table_T(check_dicts):
    actual = isen.lookup_table(T0_ratio=1.008, gam=1.4)
    expected = {
        "M": 0.2, "p0_ratio": 1.028, "T0_ratio": 1.008,
        "r0_ratio": 1.02, "a0_ratio": 1.00399203184,
        "area_ratio": 2.964, "gam": 1.4
    }
    check_dicts(actual, expected, rtol=1e-3)


def test_lookup_table_area(check_dicts):
    actual = isen.lookup_table(area_ratio=1.094, gam=1.4)
    expected = {
        "M": 1.360, "p0_ratio": 3.00932891965, "T0_ratio": 1.3698630137,
        "r0_ratio": 2.19669178218, "a0_ratio": 1.17041147196,
        "area_ratio": 1.094, "gam": 1.4
    }
    check_dicts(actual, expected, rtol=1e-4)


def test_lookup_table_area2(check_dicts):
    actual = isen.lookup_table(area_ratio=3.1, gam=1.3, is_supersonic=False)
    expected = {
        "M": 0.193, "p0_ratio": 1.02438024995, "T0_ratio": 1.0060362173,
        "r0_ratio": 1.01871377199, "a0_ratio": 1.00301356785,
        "area_ratio": 3.1, "gam": 1.3
    }
    check_dicts(actual, expected, rtol=1e-3)


def test_lookup_table_p(check_dicts):
    actual = isen.lookup_table(p0_ratio=39.59, gam=1.4)
    expected = {
        "M": 3.05, "p0_ratio": 39.59, "T0_ratio": 2.860,
        "r0_ratio": 13.84, "a0_ratio": 1.69115345253,
        "area_ratio": 4.441, "gam": 1.4
    }
    check_dicts(actual, expected, rtol=1e-3)
