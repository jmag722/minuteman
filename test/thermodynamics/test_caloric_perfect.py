import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)


import unittest
from thermodynamics.caloric_perfect import *

class TestConstants(unittest.TestCase):
    def test_constants(self):
        actual = NA*kB_SI
        expected = RU_SI
        self.assertAlmostEqual(actual, expected)

class TestIdealGasLaw(unittest.TestCase):
    
    def test_rho_law(self):
        igls = IdealGasLawSolver()
        actual = igls.solve("rho",values_dict={"p": 2, "R": 1, "T":4},)
        expected = 0.5
        self.assertEqual(actual, expected)

    def test_rho_law2(self):
        igls = IdealGasLawSolver()
        actual = igls.solve("rho",values_dict={"p": 2, "R": 3, "T":4},)
        expected = 0.16666666666
        self.assertAlmostEqual(actual, expected)

    def test_rho_law3(self):
        igls = IdealGasLawSolver()
        actual = igls.solve("p",values_dict={"n": 3, "T":4})
        expected = 1.6567788e-22
        self.assertAlmostEqual(actual, expected)

    def test_rho_law4(self):
        igls = IdealGasLawSolver()
        actual = igls.solve("rho",values_dict={"p": 2, "R": 1, "T":4})
        expected = 0.5
        self.assertEqual(actual, expected)

    def test_rho_law5(self):
        igls = IdealGasLawSolver()
        actual = igls.solve("T",values_dict={"p": 2, "V":4, "Nm":2})
        expected = 0.48108942
        self.assertAlmostEqual(actual, expected,places=6)

class TestR(unittest.TestCase):
    def test_R(self):
        actual = R(53.0,True,)
        expected = 0.1568766
        self.assertAlmostEqual(actual, expected,places=6)


class TestEntropy(unittest.TestCase):
    def test_entropy_PT(self):
        actual = entropy(T2_T1=2,p2_p1=1.5,cp=1.3,R=287)
        expected = -115.4673947
        self.assertAlmostEqual(actual,expected,places=6)

    def test_entropy_TV(self):
        actual = entropy(T2_T1=11,v2_v1=0.1,cv=0.9,R=300, s1=1000)
        expected = 311.3825778
        self.assertAlmostEqual(actual,expected,places=6)

    def test_entropy_PV(self):
        actual = entropy(p2_p1=3,v2_v1=0.5,cp=25,cv=1.4)
        expected = -15.79062231
        self.assertAlmostEqual(actual,expected,places=6)

class TestIsentropicRelations(unittest.TestCase):
    def test_p(self):
        actual = isentropic_process(p21=1.2)
        expected = {"p21":1.2,"t21":1.053472524,"r21":1.139089983,
                    "a21":1.026388096,"gamma":1.4}
        self.assertAlmostEqual(actual["p21"],expected["p21"])
        self.assertAlmostEqual(actual["a21"],expected["a21"])
        self.assertAlmostEqual(actual["t21"],expected["t21"])
        self.assertAlmostEqual(actual["r21"],expected["r21"])
        self.assertAlmostEqual(actual["gamma"],expected["gamma"])
    def test_a(self):
        actual = isentropic_process(a21=1.125)
        expected = {"p21":2.280697346,"r21":1.802032471,"t21":1.265625,"a21":1.125,"gamma":1.4}
        self.assertAlmostEqual(actual["p21"],expected["p21"])
        self.assertAlmostEqual(actual["a21"],expected["a21"])
        self.assertAlmostEqual(actual["t21"],expected["t21"])
        self.assertAlmostEqual(actual["r21"],expected["r21"])
        self.assertAlmostEqual(actual["gamma"],expected["gamma"])
    def test_r(self):
        actual = isentropic_process(r21=2,gamma=1.3)
        expected = {"p21":2.462288827,"r21":2,"t21":1.231144413,"a21":1.109569472,"gamma":1.3}
        self.assertAlmostEqual(actual["p21"],expected["p21"])
        self.assertAlmostEqual(actual["a21"],expected["a21"])
        self.assertAlmostEqual(actual["t21"],expected["t21"])
        self.assertAlmostEqual(actual["r21"],expected["r21"])
        self.assertAlmostEqual(actual["gamma"],expected["gamma"])
    def test_t(self):
        actual = isentropic_process(t21=0.7,gamma=1.35)
        expected = {"p21":0.2526509942,"r21":0.3609299917,"t21":0.7,"a21":0.8366600265,"gamma":1.35}
        self.assertAlmostEqual(actual["p21"],expected["p21"])
        self.assertAlmostEqual(actual["a21"],expected["a21"])
        self.assertAlmostEqual(actual["t21"],expected["t21"])
        self.assertAlmostEqual(actual["r21"],expected["r21"])
        self.assertAlmostEqual(actual["gamma"],expected["gamma"])

if __name__ == '__main__':
    unittest.main(verbosity=2)