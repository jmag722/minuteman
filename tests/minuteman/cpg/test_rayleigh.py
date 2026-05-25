import pytest
import minuteman.cpg.rayleigh as ray


def test_lookup_table_M_supersonic():
    actual = ray.lookup_table(M2=2.0, M1=1.0)
    assert actual["M2"] == 2.0
    assert actual["p21"] == pytest.approx(0.36363636)
    assert actual["T21"] == pytest.approx(0.52892561)
    assert actual["r21"] == pytest.approx(0.68750000214)
    assert actual["T02_T01"] == pytest.approx(0.79338842)
    assert actual["p02_p01"] == pytest.approx(1.50309597)
    assert actual["ds21_R"] == pytest.approx(1.21757520)


def test_lookup_table_M_subsonic():
    actual = ray.lookup_table(M2=0.98, M1=1.0)
    assert actual["M2"] == 0.98
    assert actual["p21"] == pytest.approx(1.02364622)
    assert actual["T21"] == pytest.approx(1.00635667)
    assert actual["r21"] == pytest.approx(1.01718034902)
    assert actual["T02_T01"] == pytest.approx(0.99971472)
    assert actual["p02_p01"] == pytest.approx(1.00019443)
    assert actual["ds21_R"] == pytest.approx(0.00119303, rel=1e-5)


def test_lookup_table_T_supersonic():
    actual = ray.lookup_table(T21=0.5, M1=1.0, gam=1.3)
    assert actual["M2"] == pytest.approx(2.14314380)
    assert actual["p21"] == pytest.approx(0.32993902)
    assert actual["T21"] == pytest.approx(0.5)
    assert actual["r21"] == pytest.approx(0.65987805677)
    assert actual["T02_T01"] == pytest.approx(0.73433035)
    assert actual["p02_p01"] == pytest.approx(1.74485794)
    assert actual["ds21_R"] == pytest.approx(1.89479037)


def test_lookup_table_T_subsonic():
    actual = ray.lookup_table(T21=0.1, M1=1.0, gam=1.35, is_supersonic=False)
    assert actual["M2"] == pytest.approx(0.13802589)
    assert actual["p21"] == pytest.approx(2.29107571)
    assert actual["T21"] == pytest.approx(0.09999999)
    assert actual["r21"] == pytest.approx(22.9107618951)
    assert actual["T02_T01"] == pytest.approx(0.08539012)
    assert actual["p02_p01"] == pytest.approx(1.24585906)
    assert actual["ds21_R"] == pytest.approx(9.71042109)


def test_lookup_table_p():
    actual = ray.lookup_table(p21=1.8, M1=1.0, gam=1.4)
    assert actual["M2"] == pytest.approx(0.48795003)
    assert actual["p21"] == pytest.approx(1.8)
    assert actual["T21"] == pytest.approx(0.77142857)
    assert actual["r21"] == pytest.approx(2.33333338)
    assert actual["T02_T01"] == pytest.approx(0.67346938)
    assert actual["p02_p01"] == pytest.approx(1.11905129)
    assert actual["ds21_R"] == pytest.approx(1.49607584)


def test_lookup_table_p2():
    actual = ray.lookup_table(p21=0.1, M1=1.0, gam=1.3)
    assert actual["M2"] == pytest.approx(4.11376675)
    assert actual["p21"] == pytest.approx(0.09999999)
    assert actual["T21"] == pytest.approx(0.16923076)
    assert actual["r21"] == pytest.approx(0.59090909171)
    assert actual["T02_T01"] == pytest.approx(0.52071005)
    assert actual["p02_p01"] == pytest.approx(13.0368330)
    assert actual["ds21_R"] == pytest.approx(5.39554689)


def test_lookup_table_ptot_supersonic():
    actual = ray.lookup_table(p02_p01=90, M1=1.0, gam=1.2, is_supersonic=True)
    assert actual["M2"] == pytest.approx(5.15182760)
    assert actual["p21"] == pytest.approx(0.06697191)
    assert actual["T21"] == pytest.approx(0.11904413)
    assert actual["r21"] == pytest.approx(0.56258049823)
    assert actual["T02_T01"] == pytest.approx(0.39545734)
    assert actual["p02_p01"] == pytest.approx(89.9999999)
    assert actual["ds21_R"] == pytest.approx(10.0660836)


def test_lookup_table_ptotsubsonic():
    actual = ray.lookup_table(
        p02_p01=1.2, M1=1.0, gam=1.33, is_supersonic=False)
    assert actual["M2"] == pytest.approx(0.28211298)
    assert actual["p21"] == pytest.approx(2.10697331)
    assert actual["T21"] == pytest.approx(0.35331674)
    assert actual["r21"] == pytest.approx(5.96341220006)
    assert actual["T02_T01"] == pytest.approx(0.30725879)
    assert actual["p02_p01"] == pytest.approx(1.19999999)
    assert actual["ds21_R"] == pytest.approx(4.93834072)


def test_lookup_table_rho_supersonic():
    actual = ray.lookup_table(r21=3.33333333333, M1=1.0, gam=1.4,
                              is_supersonic=True)
    assert actual["M2"] == pytest.approx(0.38924947)
    assert actual["p21"] == pytest.approx(1.98)
    assert actual["T21"] == pytest.approx(0.59399999)
    assert actual["r21"] == pytest.approx(3.333333333)
    assert actual["T02_T01"] == pytest.approx(0.50999999)
    assert actual["p02_p01"] == pytest.approx(1.16120326)
    assert actual["ds21_R"] == pytest.approx(2.50616270)


def test_lookup_table_Ttot_supersonic():
    actual = ray.lookup_table(T02_T01=0.6, M1=1.0, gam=1.5, is_supersonic=True)
    assert actual["M2"] == pytest.approx(5.64015853)
    assert actual["p21"] == pytest.approx(0.05131670)
    assert actual["T21"] == pytest.approx(0.08377223)
    assert actual["r21"] == pytest.approx(0.61257411404)
    assert actual["T02_T01"] == pytest.approx(0.6)
    assert actual["p02_p01"] == pytest.approx(18.8543774)
    assert actual["ds21_R"] == pytest.approx(4.46922198)


def test_lookup_table_Ttot_subsonic():
    actual = ray.lookup_table(
        T02_T01=0.1, M1=1.0, gam=1.5, is_supersonic=False)
    assert actual["M2"] == pytest.approx(0.14552929)
    assert actual["p21"] == pytest.approx(2.42302494)
    assert actual["T21"] == pytest.approx(0.12434164)
    assert actual["r21"] == pytest.approx(19.4868337208)
    assert actual["T02_T01"] == pytest.approx(0.09999999)
    assert actual["p02_p01"] == pytest.approx(1.26039890)
    assert actual["ds21_R"] == pytest.approx(7.13918354)


def test_lookup_table_pressure_temperature():
    gam = 1.4
    M1 = 1.5
    M2 = 3.0
    p1 = 10.0

    p1_pstar = ray.lookup_table(M2=M1, gam=gam)["p21"]
    p2_pstar = ray.lookup_table(M2=M2, gam=gam)["p21"]
    p2_from_ref = p2_pstar * (1 / p1_pstar) * p1

    p2_p1 = ray.lookup_table(M2=M2, M1=M1, gam=gam)["p21"]
    p2_direct = p2_p1 * p1
    assert p2_from_ref == pytest.approx(3.05, rel=1e-3)
    assert p2_direct == pytest.approx(3.05, rel=1e-3)

    tt1_ttstar = ray.lookup_table(M2=M1, gam=gam)["T02_T01"]
    tt2_ttstar = ray.lookup_table(M2=M2, gam=gam)["T02_T01"]
    tt2_tt1_From_ref = tt2_ttstar / tt1_ttstar

    tt2_tt1_direct = ray.lookup_table(M2=M2, M1=M1, gam=gam)["T02_T01"]
    assert tt2_tt1_From_ref == pytest.approx(.719, rel=1e-3)
    assert tt2_tt1_direct == pytest.approx(.719, rel=1e-3)


def test_entropy():
    # specifying (s1-s*)/R
    ans_sub = ray.lookup_table(ds21_R=2, M1=1.0, gam=1.4, is_supersonic=False)
    assert ans_sub["M2"] == pytest.approx(0.43346186)
    assert ans_sub["p21"] == pytest.approx(1.90017003)
    assert ans_sub["T21"] == pytest.approx(0.67840136)
    assert ans_sub["r21"] == pytest.approx(2.80095254794)
    assert ans_sub["T02_T01"] == pytest.approx(0.58657852)
    assert ans_sub["p02_p01"] == pytest.approx(1.14216936)
    assert ans_sub["ds21_R"] == pytest.approx(1.99999999)
    # specifying (s2-s*)/R
    ans_sup = ray.lookup_table(ds21_R=4, M1=1.0, gam=1.4, is_supersonic=True)
    assert ans_sup["M2"] == pytest.approx(4.03619180)
    assert ans_sup["p21"] == pytest.approx(0.10080991)
    assert ans_sup["T21"] == pytest.approx(0.16555796)
    assert ans_sup["r21"] == pytest.approx(0.6089100727)
    assert ans_sup["T02_T01"] == pytest.approx(0.58747815)
    assert ans_sup["p02_p01"] == pytest.approx(8.48494372)
    assert ans_sup["ds21_R"] == pytest.approx(4.00000000)
    # confirm direct calculation is same as solving with sonic condition
    my_s21 = ans_sup["ds21_R"] - ans_sub["ds21_R"]  # (s2-s1)/R
    assert my_s21 == pytest.approx(2.000000)
    calc_ans_s21 = ray.lookup_table(M2=4.03619180, M1=0.43346186, gam=1.4)
    assert calc_ans_s21["ds21_R"] == pytest.approx(2.000000)
    assert my_s21 == pytest.approx(calc_ans_s21["ds21_R"])
