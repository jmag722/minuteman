import pytest
import phypy.compressible.normal_shock as nsk

def test_lookup_table_m1(check_dicts):
    actual=nsk.lookup_table(M1=3.0)
    expected = {
        "M1":3.0,
        "p21":10.33,
        "p02_p01":0.3283,
        "T21":2.679,
        "r21":3.857,
        "M2":.4752,
        "p02_p1":12.06
    }
    check_dicts(actual, expected, rtol=1e-3)

def test_lookup_table_m2(check_dicts):
    actual=nsk.lookup_table(M2=0.9805)
    expected = {
        "M1":1.020,
        "p21":1.047,
        "p02_p01":1.0,
        "T21":1.013,
        "r21":1.033,
        "M2":0.9805,
        "p02_p1":1.938
    }
    check_dicts(actual, expected, rtol=1e-3)

def test_lookup_table_p21(check_dicts):
    actual=nsk.lookup_table(p21=2916)
    expected = {
        "M1":50,
        "p21":2916,
        "p02_p01":0.1144e-5,
        "T21":487.1,
        "r21":5.988,
        "M2":0.3784,
        "p02_p1":3219
    }
    check_dicts(actual, expected, rtol=1e-3)

def test_lookup_table_r21(check_dicts):
    actual=nsk.lookup_table(r21=2.637)
    expected = {
        "M1":1.98,
        "p21":4.407,
        "p02_p01":0.7302,
        "T21":1.671,
        "r21":2.637,
        "M2":0.5808,
        "p02_p1":5.539
    }
    check_dicts(actual, expected, rtol=1e-3)

def test_lookup_table_t21(check_dicts):
    actual=nsk.lookup_table(T21=2.0,gam=1.3)
    expected = {
        "M1":2.66849127,
        "p21":7.91921689,
        "p02_p01":0.39284291,
        "T21":2.0,
        "r21":3.95960844,
        "M2":0.47653910,
        "p02_p1":9.15630005635
    }
    check_dicts(actual, expected, rtol=1e-6)

def test_lookup_table_p0201(check_dicts):
    actual=nsk.lookup_table(p02_p01=0.6,gam=1.35)
    expected = {
        "M1":2.23100735,
        "p21":5.56977161,
        "p02_p01":0.6,
        "T21":1.78189249,
        "r21":3.12576186,
        "M2":0.53469283,
        "p02_p1":6.72385620146
    }
    check_dicts(actual, expected, rtol=1e-6)

def test_lookup_table_p021(check_dicts):
    actual=nsk.lookup_table(p02_p1=3.887,gam=1.35)
    expected = {
        "M1":1.64168094,
        "p21":2.94758044,
        "p02_p01":0.87573717,
        "T21":1.36978829,
        "r21":2.15185109,
        "M2":0.65185322,
        "p02_p1":3.887
    }
    check_dicts(actual, expected, rtol=1e-7)

def test_entropy():
    ds = nsk.entropy2(p02_p01=.5615, R=1716, s1=0.0)
    assert ds == pytest.approx(990.4, rel=1e-4)

def test_hugoniot():
    de = nsk.hugoniot(p1=1e5, p2=3e4, v1=0.8, v2=3.0, e1=0.0)
    assert de == pytest.approx(-143000.0)