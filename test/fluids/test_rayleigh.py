import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)

from fluids.compressible.inviscid.rayleigh import *
import pytest

def test_lookup_table_m_supersonic():
    ans = lookup_table(m2=2.0,m1=1.0)
    assert ans["m2"] == 2.0
    assert ans["p21"] == pytest.approx(0.36363636)
    assert ans["t21"] == pytest.approx(0.52892561)
    assert ans["r21"] == pytest.approx(0.68750000214)
    assert ans["t02_t01"] == pytest.approx(0.79338842)
    assert ans["p02_p01"] == pytest.approx(1.50309597)
    assert ans["s21"] == pytest.approx(-1.21757520)

def test_lookup_table_m_subsonic():
    ans = lookup_table(m2=0.98,m1=1.0)
    assert ans["m2"] == 0.98
    assert ans["p21"] == pytest.approx(1.02364622)
    assert ans["t21"] == pytest.approx(1.00635667)
    assert ans["r21"] == pytest.approx(1.01718034902)
    assert ans["t02_t01"] == pytest.approx(0.99971472)
    assert ans["p02_p01"] == pytest.approx(1.00019443)
    assert ans["s21"] == pytest.approx(-0.00119303,rel=1e-5)

def test_lookup_table_tsupersonic():
    ans = lookup_table(t21 = 0.5,m1=1.0,gam=1.3)
    assert ans["m2"] == pytest.approx(2.14314380)
    assert ans["p21"] == pytest.approx(0.32993902)
    assert ans["t21"] == pytest.approx(0.5)
    assert ans["r21"] == pytest.approx(0.65987805677)
    assert ans["t02_t01"] == pytest.approx(0.73433035)
    assert ans["p02_p01"] == pytest.approx(1.74485794)
    assert ans["s21"] == pytest.approx(-1.89479037)

def test_lookup_table_tsubsonic():
    ans = lookup_table(t21 = 0.1,m1=1.0,gam=1.35,supersonic=False)
    assert ans["m2"] == pytest.approx(0.13802589)
    assert ans["p21"] == pytest.approx(2.29107571)
    assert ans["t21"] == pytest.approx(0.09999999)
    assert ans["r21"] == pytest.approx(22.9107618951)
    assert ans["t02_t01"] == pytest.approx(0.08539012)
    assert ans["p02_p01"] == pytest.approx(1.24585906)
    assert ans["s21"] == pytest.approx(-9.71042109)

def test_lookup_table_pratio():
    ans = lookup_table(p21 = 1.8,m1=1.0,gam=1.4)
    assert ans["m2"] == pytest.approx(0.48795003)
    assert ans["p21"] == pytest.approx(1.8)
    assert ans["t21"] == pytest.approx(0.77142857)
    assert ans["r21"] == pytest.approx(2.33333338)
    assert ans["t02_t01"] == pytest.approx(0.67346938)
    assert ans["p02_p01"] == pytest.approx(1.11905129)
    assert ans["s21"] == pytest.approx(-1.49607584)

def test_lookup_table_pratio2():
    ans = lookup_table(p21 = 0.1,m1=1.0,gam=1.3)
    assert ans["m2"] == pytest.approx(4.11376675)
    assert ans["p21"] == pytest.approx(0.09999999)
    assert ans["t21"] == pytest.approx(0.16923076)
    assert ans["r21"] == pytest.approx(0.59090909171)
    assert ans["t02_t01"] == pytest.approx(0.52071005)
    assert ans["p02_p01"] == pytest.approx(13.0368330)
    assert ans["s21"] == pytest.approx(-5.39554689)

def test_lookup_table_ptotsupersonic():
    ans = lookup_table(p02_p01 = 90,m1=1.0,gam=1.2,supersonic=True)
    assert ans["m2"] == pytest.approx(5.15182760)
    assert ans["p21"] == pytest.approx(0.06697191)
    assert ans["t21"] == pytest.approx(0.11904413)
    assert ans["r21"] == pytest.approx(0.56258049823)
    assert ans["t02_t01"] == pytest.approx(0.39545734)
    assert ans["p02_p01"] == pytest.approx(89.9999999)
    assert ans["s21"] == pytest.approx(-10.0660836)

def test_lookup_table_ptotsubsonic():
    ans = lookup_table(p02_p01 = 1.2,m1=1.0,gam=1.33,supersonic=False)
    assert ans["m2"] == pytest.approx(0.28211298)
    assert ans["p21"] == pytest.approx(2.10697331)
    assert ans["t21"] == pytest.approx(0.35331674)
    assert ans["r21"] == pytest.approx(5.96341220006)
    assert ans["t02_t01"] == pytest.approx(0.30725879)
    assert ans["p02_p01"] == pytest.approx(1.19999999)
    assert ans["s21"] == pytest.approx(-4.93834072)

def test_lookup_table_rho():
    ans = lookup_table(r21=3.33333333333,m1=1.0,gam=1.4,supersonic=True)
    assert ans["m2"] == pytest.approx(0.38924947)
    assert ans["p21"] == pytest.approx(1.98)
    assert ans["t21"] == pytest.approx(0.59399999)
    assert ans["r21"] == pytest.approx(3.333333333)
    assert ans["t02_t01"] == pytest.approx(0.50999999)
    assert ans["p02_p01"] == pytest.approx(1.16120326)
    assert ans["s21"] == pytest.approx(-2.50616270)

def test_lookup_table_ttot_supersonic():
    ans = lookup_table(t02_t01=0.6,m1=1.0,gam=1.5,supersonic=True)
    assert ans["m2"] == pytest.approx(5.64015853)
    assert ans["p21"] == pytest.approx(0.05131670)
    assert ans["t21"] == pytest.approx(0.08377223)
    assert ans["r21"] == pytest.approx(0.61257411404)
    assert ans["t02_t01"] == pytest.approx(0.6)
    assert ans["p02_p01"] == pytest.approx(18.8543774)
    assert ans["s21"] == pytest.approx(-4.46922198)

def test_lookup_table_ttot_subsonic():
    ans = lookup_table(t02_t01=0.1,m1=1.0,gam=1.5,supersonic=False)
    assert ans["m2"] == pytest.approx(0.14552929)
    assert ans["p21"] == pytest.approx(2.42302494)
    assert ans["t21"] == pytest.approx(0.12434164)
    assert ans["r21"] == pytest.approx(19.4868337208)
    assert ans["t02_t01"] == pytest.approx(0.09999999)
    assert ans["p02_p01"] == pytest.approx(1.26039890)
    assert ans["s21"] == pytest.approx(-7.13918354)

def test_lookup_table_pressure_temperature():
    gam=1.4
    m1=1.5
    m2=3.0
    p1=10.0

    p1_pstar = lookup_table(m2=m1,gam=gam)["p21"]
    p2_pstar = lookup_table(m2=m2, gam=gam)["p21"]
    p2_from_ref = p2_pstar * (1 / p1_pstar) * p1

    p2_p1 = lookup_table(m2=m2,m1=m1,gam=gam)["p21"]
    p2_direct = p2_p1 * p1
    assert p2_from_ref == pytest.approx(3.05,rel=1e-3)
    assert p2_direct == pytest.approx(3.05,rel=1e-3)

    tt1_ttstar = lookup_table(m2=m1,gam=gam)["t02_t01"]
    tt2_ttstar = lookup_table(m2=m2,gam=gam)["t02_t01"]
    tt2_tt1_From_ref = tt2_ttstar / tt1_ttstar

    tt2_tt1_direct = lookup_table(m2=m2,m1=m1,gam=gam)["t02_t01"]
    assert tt2_tt1_From_ref == pytest.approx(.719,rel=1e-3)
    assert tt2_tt1_direct == pytest.approx(.719,rel=1e-3)

def test_entropy():
    # specifying (s1-s*)/R
    ans_sub = lookup_table(s21=-2,m1=1.0,gam=1.4,supersonic=False)
    assert ans_sub["m2"] == pytest.approx(0.43346186)
    assert ans_sub["p21"] == pytest.approx(1.90017003)
    assert ans_sub["t21"] == pytest.approx(0.67840136)
    assert ans_sub["r21"] == pytest.approx(2.80095254794)
    assert ans_sub["t02_t01"] == pytest.approx(0.58657852)
    assert ans_sub["p02_p01"] == pytest.approx(1.14216936)
    assert ans_sub["s21"] == pytest.approx(-1.99999999)
    # specifying (s2-s*)/R
    ans_sup = lookup_table(s21=-4,m1=1.0,gam=1.4,supersonic=True)
    assert ans_sup["m2"] == pytest.approx(4.03619180)
    assert ans_sup["p21"] == pytest.approx(0.10080991)
    assert ans_sup["t21"] == pytest.approx(0.16555796)
    assert ans_sup["r21"] == pytest.approx(0.6089100727)
    assert ans_sup["t02_t01"] == pytest.approx(0.58747815)
    assert ans_sup["p02_p01"] == pytest.approx(8.48494372)
    assert ans_sup["s21"] == pytest.approx(-4.00000000)
    # confirm direct calculation is same as solving with sonic condition
    my_s21 = ans_sup["s21"] - ans_sub["s21"] # (s2-s1)/R
    assert my_s21 == pytest.approx(-2.000000)
    calc_ans_s21 = lookup_table(m2=4.03619180,m1=0.43346186,gam=1.4)
    assert calc_ans_s21["s21"] == pytest.approx(-2.000000)
    assert my_s21 == pytest.approx(calc_ans_s21["s21"])

