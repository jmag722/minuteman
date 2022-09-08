# from cgitb import lookup
import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)

import unittest
from fluids.compressible.inviscid.flow_1d import *

class TestNormalShockLookup(unittest.TestCase):
    def test_m1(self):
        normal_shock=lookup_normal_shock(M1=3.0)
        self.assertAlmostEqual(normal_shock["p21"],10.33,places=2)
        self.assertAlmostEqual(normal_shock["p02_p01"],0.3283,places=4)
        self.assertAlmostEqual(normal_shock["t21"],2.679,places=3)
        self.assertAlmostEqual(normal_shock["r21"],3.857,places=3)
        self.assertAlmostEqual(normal_shock["M2"],.4752,places=4)
        self.assertAlmostEqual(normal_shock["p02_p1"],12.06,places=2)
    
    def test_m2(self):
        normal_shock=lookup_normal_shock(M2=0.9805)
        self.assertAlmostEqual(normal_shock["p21"],1.047,places=3)
        self.assertAlmostEqual(normal_shock["p02_p01"],1.0,places=4)
        self.assertAlmostEqual(normal_shock["t21"],1.013,places=3)
        self.assertAlmostEqual(normal_shock["r21"],1.033,places=3)
        self.assertAlmostEqual(normal_shock["M1"],1.020,places=4)
        self.assertAlmostEqual(normal_shock["p02_p1"],1.938,places=3)

    def test_p21(self):
        normal_shock=lookup_normal_shock(p21=2916)
        self.assertAlmostEqual(normal_shock["M2"],0.3784,places=3)
        self.assertAlmostEqual(normal_shock["p02_p01"],0.1144e-5,places=4)
        self.assertAlmostEqual(normal_shock["t21"],487.1,places=0)
        self.assertAlmostEqual(normal_shock["r21"],5.988,places=3)
        self.assertAlmostEqual(normal_shock["M1"],50,places=1)
        self.assertAlmostEqual(normal_shock["p02_p1"],3219,places=0)

    def test_r21(self):
        normal_shock=lookup_normal_shock(r21=2.637)
        self.assertAlmostEqual(normal_shock["M2"],0.5808,places=4)
        self.assertAlmostEqual(normal_shock["p02_p01"],0.7302,places=4)
        self.assertAlmostEqual(normal_shock["t21"],1.671,places=3)
        self.assertAlmostEqual(normal_shock["p21"],4.407,places=3)
        self.assertAlmostEqual(normal_shock["M1"],1.98,places=2)
        self.assertAlmostEqual(normal_shock["p02_p1"],5.539,places=3)

    def test_t21(self):
        normal_shock=lookup_normal_shock(t21=2.0,gam=1.3)
        self.assertAlmostEqual(normal_shock["M2"],0.47653910)
        self.assertAlmostEqual(normal_shock["p02_p01"],0.39284291)
        self.assertAlmostEqual(normal_shock["r21"],3.95960844)
        self.assertAlmostEqual(normal_shock["p21"],7.91921689)
        self.assertAlmostEqual(normal_shock["M1"],2.66849127)
        self.assertAlmostEqual(normal_shock["p02_p1"],9.15630005635,places=6)

    def test_p0201(self):
        normal_shock=lookup_normal_shock(p02_p01=0.6,gam=1.35)
        self.assertAlmostEqual(normal_shock["M2"],0.53469283,places=6)
        self.assertAlmostEqual(normal_shock["t21"],1.78189249,places=5)
        self.assertAlmostEqual(normal_shock["r21"],3.12576186,places=5)
        self.assertAlmostEqual(normal_shock["p21"],5.56977161,places=4)
        self.assertAlmostEqual(normal_shock["M1"],2.23100735,places=5)
        self.assertAlmostEqual(normal_shock["p02_p1"],6.72385620146,places=4)

    def test_p021(self):
        normal_shock=lookup_normal_shock(p02_p1=3.887,gam=1.35)
        self.assertAlmostEqual(normal_shock["M2"],0.65185322,places=7)
        self.assertAlmostEqual(normal_shock["t21"],1.36978829,places=7)
        self.assertAlmostEqual(normal_shock["r21"],2.15185109,places=6)
        self.assertAlmostEqual(normal_shock["p21"],2.94758044,places=6)
        self.assertAlmostEqual(normal_shock["M1"],1.64168094,places=7)
        self.assertAlmostEqual(normal_shock["p02_p01"],0.87573717,places=7)

class TestMisc(unittest.TestCase):
    def test_entropy(self):
        ds = normal_shock_entropy(R=1716,p02_p01=.5615,s1=0.0)
        self.assertAlmostEqual(ds,990.4,places=1)

    def test_hugoniot(self):
        de = hugoniot(p1=1e5,p2=3e4,v1=0.8,v2=3.0,e1=0.0)
        self.assertEqual(de,-143000.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)