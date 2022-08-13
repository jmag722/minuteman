import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)


import unittest
from thermodynamics.caloric_perfect import *

class TestConstants(unittest.TestCase):
    def test_constants(self):
        actual = NA*kB
        expected = RU
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
        actual = igls.solve("p",values_dict={"n": 3, "T":4},eq=igls.PARTICLE_EQ)
        expected = 1.6567788e-22
        self.assertAlmostEqual(actual, expected)

    def test_rho_law4(self):
        igls = IdealGasLawSolver()
        actual = igls.solve("rho",values_dict={"p": 2, "R": 1, "T":4},
                            eq=igls.RHO_EQ)
        expected = 0.5
        self.assertEqual(actual, expected)

    def test_rho_law5(self):
        igls = IdealGasLawSolver()
        actual = igls.solve("T",values_dict={"p": 2, "V":4, "Nm":2},
                            eq=igls.VOL_EQ)
        expected = 0.48108942
        self.assertAlmostEqual(actual, expected,places=6)

class TestR(unittest.TestCase):
    def test_R(self):
        actual = R(53.0,True)
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

if __name__ == '__main__':
    unittest.main(verbosity=2)