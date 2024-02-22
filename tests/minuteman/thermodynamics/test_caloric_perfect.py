import pytest
import minuteman.thermodynamics.caloric_perfect as calp

class TestIdealGasLawSolver:
    
    def test_rho_law(self):
        igls = calp.IdealGasLawSolver()
        actual = igls.solve("rho", knowns={"p": 2, "R": 1, "T":4},)
        expected = 0.5
        assert actual == pytest.approx(expected)

    def test_rho_law2(self):
        igls = calp.IdealGasLawSolver()
        actual = igls.solve("rho", knowns={"p": 2, "R": 3, "T":4},)
        expected = 0.16666666666
        assert actual == pytest.approx(expected)

    def test_rho_law3(self):
        igls = calp.IdealGasLawSolver()
        actual = igls.solve("p", knowns={"n": 3, "T":4})
        expected = 1.6567788e-22
        assert actual == pytest.approx(expected)

    def test_rho_law4(self):
        igls = calp.IdealGasLawSolver()
        actual = igls.solve("rho", knowns={"p": 2, "R": 1, "T":4})
        expected = 0.5
        assert actual == pytest.approx(expected)

    def test_rho_law5(self):
        igls = calp.IdealGasLawSolver()
        actual = igls.solve("T", knowns={"p": 2, "V":4, "Nm":2})
        expected = 0.48108942
        assert actual == pytest.approx(expected)

def test_R():
    actual = calp.R(53.0,True,)
    expected = 0.1568766
    assert actual == pytest.approx(expected)

class TestEntropy:
    def test_entropy_PT(self):
        actual = calp.entropy(t21=2,p21=1.5,cp=1.3,R=287)
        expected = -115.4673947
        assert actual == pytest.approx(expected)

    def test_entropy_TV(self):
        actual = calp.entropy(t21=11,v21=0.1,cv=0.9,R=300, s1=1000)
        expected = 311.3825778
        assert actual == pytest.approx(expected)

    def test_entropy_PV(self):
        actual = calp.entropy(p21=3,v21=0.5,cp=25,cv=1.4)
        expected = -15.79062231
        assert actual == pytest.approx(expected)

class TestIsentropicRelations:
    def test_p(self, check_dicts):
        actual = calp.isentropic_process(p21=1.2)
        expected = {"p21":1.2,"t21":1.053472524,"r21":1.139089983,
                    "a21":1.026388096,"gam":1.4}
        check_dicts(actual, expected)

    def test_a(self, check_dicts):
        actual = calp.isentropic_process(a21=1.125)
        expected = {"p21":2.280697346,"r21":1.802032471,"t21":1.265625,
                    "a21":1.125,"gam":1.4}
        check_dicts(actual, expected)

    def test_r(self, check_dicts):
        actual = calp.isentropic_process(r21=2,gam=1.3)
        expected = {"p21":2.462288827,"r21":2,"t21":1.231144413,
                    "a21":1.109569472,"gam":1.3}
        check_dicts(actual, expected)

    def test_t(self, check_dicts):
        actual = calp.isentropic_process(t21=0.7,gam=1.35)
        expected = {"p21":0.2526509942,"r21":0.3609299917,"t21":0.7,
                    "a21":0.8366600265,"gam":1.35}
        check_dicts(actual, expected)

def test_entropy_state():
    assert calp.entropy_state(100, 2.5, gam=1.3, R=200) == pytest.approx(2275.9948230344603)

def test_total_energy():
    assert calp.total_energy(100, 1.2, 5000, 1.3) == pytest.approx(15000333.333333334)

def test_specific_enthalpy():
    assert calp.specific_enthalpy(100, 1e4, 0.8) == pytest.approx(12600)