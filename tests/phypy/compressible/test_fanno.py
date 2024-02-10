import pytest
import phypy.compressible.fanno as fan

def test_lookup_table_M(check_dicts):
    check_dicts(
        fan.lookup_table(M=0.4, gam=1.3),
        {
            "M": 0.4,
            "p_ratio": 2.64934764,
            "r_ratio": 2.35907132043,
            "T_ratio": 1.12304687,
            "ds_R_ratio": 0.47144997,
            "p0_ratio": 1.60231582,
            "fanno_param": 2.51998734 
        }
    )

def test_lookup_table_p(check_dicts):
    check_dicts(
        fan.lookup_table(p_ratio=0.9, gam=1.3), 
        {
            "M": 1.09670359,
            "p_ratio": 0.9,
            "r_ratio": 0.92380212195,
            "T_ratio": 0.97423461,
            "ds_R_ratio": 0.00775303,
            "p0_ratio": 1.00778316,
            "fanno_param": 0.01054915
        }
    )

def test_lookup_table_r(check_dicts):
    check_dicts(
        fan.lookup_table(r_ratio=6.66666666667, gam=1.2), 
        {
            "M": 0.14316588,
            "p_ratio": 7.31833333,
            "r_ratio": 6.66666666667,
            "T_ratio": 1.09775,
            "ds_R_ratio": 1.43080683,
            "p0_ratio": 4.18207206,
            "fanno_param": 36.3460207
        }
    )

def test_lookup_table_T(check_dicts):
    check_dicts(
        fan.lookup_table(T_ratio=0.01, gam=1.5), 
        {
            "M": 22.2710574,
            "p_ratio": 0.00449013,
            "r_ratio": 1/2.22710574,
            "T_ratio": 0.01,
            "ds_R_ratio": 8.40963750,
            "p0_ratio": 4490.13255,
            "fanno_param": 0.66918220
        }
    )

def test_lookup_table_s(check_dicts):
    check_dicts(
        fan.lookup_table(ds_R_ratio=1.1, is_supersonic=False, gam=1.5), 
        {
            "M": 0.19511262,
            "p_ratio": 5.70312323,
            "r_ratio": 1/0.21711184,
            "T_ratio": 1.23821561,
            "ds_R_ratio": 1.1,
            "p0_ratio": 3.00416602,
            "fanno_param": 14.2998521
        }
    )
    check_dicts(
        fan.lookup_table(ds_R_ratio=1.5, is_supersonic=True, gam=1.1), 
        {
            "M": 2.54543291,
            "p_ratio": 0.34986083,
            "r_ratio": 1/2.26682835,
            "T_ratio": 0.79307445,
            "ds_R_ratio": 1.5,
            "p0_ratio": 4.48168906,
            "fanno_param": 0.79358257
        }
    )

def test_lookup_table_p0(check_dicts):
    check_dicts(
        fan.lookup_table(p0_ratio=5, is_supersonic=False, gam=1.4), 
        {
            "M": 0.11668889,
            "p_ratio": 9.37498439,
            "r_ratio": 1/0.12765258,
            "T_ratio": 1.19674096,
            "ds_R_ratio": 1.60943791,
            "p0_ratio": 5,
            "fanno_param": 48.2150982
        }
    )
    check_dicts(
        fan.lookup_table(p0_ratio=1.5, is_supersonic=True, gam=1.1), 
        {
            "M": 1.73162005,
            "p_ratio": 0.55183241,
            "r_ratio": 1/1.65467392,
            "T_ratio": 0.91310271,
            "ds_R_ratio": 0.40546510,
            "p0_ratio": 1.5,
            "fanno_param": 0.35551592
        }
    )

def test_lookup_table_fan(check_dicts):
    check_dicts(
        fan.lookup_table(fanno_param=33, is_supersonic=False, gam=1.4), 
        {
            "M": 0.13904873,
            "p_ratio": 7.86294980,
            "r_ratio": 1/0.15202660,
            "T_ratio": 1.19537758,
            "ds_R_ratio": 1.43754446,
            "p0_ratio": 4.21034447,
            "fanno_param": 33
        }
    )
    check_dicts(
        fan.lookup_table(fanno_param=0.01, is_supersonic=True, gam=1.3), 
        {
            "M": 1.09393234,
            "p_ratio": 0.90262819,
            "r_ratio": 1/ 1.08016431,
            "T_ratio": 	0.97498675,
            "ds_R_ratio": 0.00732479,
            "p0_ratio": 1.00735168,
            "fanno_param": 0.01
        }
    )


def test_mach2_pressure():
    assert fan.mach2(M1=1, p21=4) == pytest.approx(0.27185940)
    assert fan.mach2(M1=1, p21=0.5, gam=1.3) == pytest.approx(1.76924837)
    assert fan.mach2(M1=2, p21=0.9) == pytest.approx(fan.mach2(p21=fan.pressure2(M2=2)*0.9))

def test_mach2_temperature():
    assert fan.mach2(M1=1, gam=1.3, T21=.5) == pytest.approx(2.94392028)
    assert fan.mach2(M1=1, gam=1.5, T21=1.1) == pytest.approx(0.73854894)
    assert fan.mach2(M1=4, T21=0.1) == pytest.approx(fan.mach2(T21=fan.temperature2(M2=4)*0.1))

def test_mach2_density():
    assert fan.mach2(M1=1, gam=1.3, r21=.5) == pytest.approx(2.69679944)
    assert fan.mach2(M1=1, gam=1.5, r21=5) == pytest.approx(0.17960530)
    assert fan.mach2(M1=.3, r21=.4) == pytest.approx(fan.mach2(r21=fan.density2(M2=0.3)*0.4))
    
def test_mach2_entropy():
    assert fan.mach2(M1=.3, is_supersonic=False, ds21_R=0.5) == pytest.approx(0.56421189)
    assert fan.mach2(M1=4, is_supersonic=True, ds21_R=0.3) == pytest.approx(3.66913016)
    with pytest.raises(ValueError):
        fan.mach2(ds21_R=0.3)
    
def test_mach1_entropy():
    assert fan.mach1(M2=1, is_supersonic=False, ds21_R=2, gam=1.5) == pytest.approx(0.07776356)
    assert fan.mach1(M2=1, is_supersonic=True, ds21_R=0.6, gam=1.3) == pytest.approx(2.02884749)
    assert fan.mach1(ds21_R=(fan.entropy2(M1=4)-0.3)) == pytest.approx(3.66913016)

def test_mach2_total_pressure():
    assert fan.mach2(M1=1, is_supersonic=False, p02_p01=2, gam=1.3) == pytest.approx(0.30900860)
    assert fan.mach2(M1=1, is_supersonic=True, p02_p01=3, gam=1.2) == pytest.approx(2.39713187)
    assert fan.mach2(M1=2.3, is_supersonic=True, p02_p01=0.5, gam=1.2) == pytest.approx(fan.mach2(
        p02_p01=fan.total_pressure2(M2=2.3,gam=1.2)*.5, gam=1.2)
    )

def test_mach2_fanno_param():
    #M0.77231902
    assert fan.mach2(M1=.4, fanno_param=.1) == pytest.approx(0.40563819)
    assert fan.mach2(M1=3, fanno_param=0.2) == pytest.approx(2.05872043)
    with pytest.raises(ValueError):
        fan.mach2(fanno_param=.5)

def test_mach1_fanno_param():
    assert fan.mach1(M2=1, is_supersonic=False, fanno_param=1.2, gam=1.3) == pytest.approx(0.49693601)
    assert fan.mach1(M2=1, is_supersonic=True, fanno_param=0.4, gam=1.35) == pytest.approx(2.23092883)
    # L1* - L2* = L
    assert fan.mach1(fanno_param=(.2+fan.fanno_parameter(M1=2))) == pytest.approx(2.89064091)

def test_temperature2():
    assert fan.temperature2(M2=3, M1=4, gam=1.35) == pytest.approx(0.45631067*1/0.30921052)
    assert fan.temperature2(M2=0.5, M1=1, gam=1.4) == pytest.approx(1.14285714)
    assert fan.temperature2(M2=1.02, M1=1, gam=1.4, T1=300) == pytest.approx(0.99331170*300)

def test_pressure2():
    assert fan.pressure2(M2=5, M1=1, gam=1.2) == pytest.approx(0.11212238)
    assert fan.pressure2(M2=0.3, M1=0.1, gam=1.35) == pytest.approx(3.58512467 * 1/10.8302693)
    assert fan.pressure2(M2=0.9, gam=1.4, p1=101325) == pytest.approx(1.12913286*101325)

def test_density2():
    assert fan.density2(M2=1.1, M1=1, gam=1.6, rho1=2) == pytest.approx(2/1.07427738)
    assert fan.density2(M2=.5, M1=1, gam=1.5) == pytest.approx(1/0.54232614)
    assert fan.density2(M2=2, M1=3, gam=1.3) == pytest.approx(2.09863177/1.69558249)

def test_total_pressure2():
    assert fan.total_pressure2(M2=2) == pytest.approx(1.6875)
    assert fan.total_pressure2(M2=2, M1=2.5) == pytest.approx(0.64)

def test_entropy2():
    assert fan.entropy2(M2=1,M1=2) == pytest.approx(0.52324814)
    assert fan.entropy2(M2=1,M1=0.3) == pytest.approx(0.71052788)
    assert fan.entropy2(M2=1.5, M1=1.8) == pytest.approx(0.20167506)

def test_fanno_parameter():
    assert fan.fanno_parameter(0.3, M2=1, gam=1.4) == pytest.approx(5.29925)
    assert fan.fanno_parameter(0.47444776, M2=1, gam=1.4) == pytest.approx(1.29925)
    assert fan.fanno_parameter(0.3, M2=0.47444776, gam=1.4) == pytest.approx(4)
    assert fan.fanno_parameter(3, M2=1, gam=1.35) == pytest.approx(0.57108656)
    assert fan.fanno_parameter(2, M2=1, gam=1.35) == pytest.approx(0.32955389)
    assert fan.fanno_parameter(3, M2=2, gam=1.35) == pytest.approx(0.24153267)
    with pytest.raises(ValueError):
        fan.fanno_parameter(0.3, 1.4)
        fan.fanno_parameter(1, 1.4)
        fan.fanno_parameter(1, .5)
        fan.fanno_parameter(0.3, 0.1)
        fan.fanno_parameter(3.3, 0.4)
        fan.fanno_parameter(3.3, 5.1)

def test_L_star():
    assert fan.L_star(3, D=.2) == pytest.approx(30)
    assert fan.L_star(.5, D=1.2, f=.002) == pytest.approx(75)