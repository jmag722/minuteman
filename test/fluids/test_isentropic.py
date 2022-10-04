import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)


import unittest
from fluids.compressible.inviscid.isentropic import *

class TestIsentropicLookupTable(unittest.TestCase):
    def test_a(self):
        actual = speed_sound(t=300,R=287,gam=1.4)
        expected = 347.188709494
        self.assertAlmostEqual(actual,expected)
        actual = speed_sound(gam=1.1,p=1e5,rho=1.2)
        expected = 302.76503541
        self.assertAlmostEqual(actual,expected)

    def test_mach(self):
        self.assertEqual(mach(1,4),0.25)
        actual = lookup_table(M=2.0,gam=1.4)
        expected = {"M":2.0,"p0_ratio":7.824,"t0_ratio":1.8,
                    "r0_ratio":4.347,"a0_ratio":1.342,
                    "area_ratio":1.687,"gam":1.4}
        self.assertAlmostEqual(actual["M"],expected["M"])
        self.assertAlmostEqual(actual["p0_ratio"],expected["p0_ratio"],places=3)
        self.assertAlmostEqual(actual["t0_ratio"],expected["t0_ratio"])
        self.assertAlmostEqual(actual["r0_ratio"],expected["r0_ratio"],places=3)
        self.assertAlmostEqual(actual["a0_ratio"],expected["a0_ratio"],places=3)
        self.assertAlmostEqual(actual["area_ratio"],expected["area_ratio"],places=2)
        self.assertAlmostEqual(actual["gam"],expected["gam"])

    def test_t(self):
        actual = lookup_table(t0_ratio=1.008,gam=1.4)
        expected = {"M":0.2,"p0_ratio":1.028,"t0_ratio":1.008,
                    "r0_ratio":1.02,"a0_ratio":1.00399203184,
                    "area_ratio":2.964,"gam":1.4}
        self.assertAlmostEqual(actual["M"],expected["M"])
        self.assertAlmostEqual(actual["p0_ratio"],expected["p0_ratio"],places=3)
        self.assertAlmostEqual(actual["t0_ratio"],expected["t0_ratio"])
        self.assertAlmostEqual(actual["r0_ratio"],expected["r0_ratio"],places=3)
        self.assertAlmostEqual(actual["a0_ratio"],expected["a0_ratio"],places=3)
        self.assertAlmostEqual(actual["area_ratio"],expected["area_ratio"],places=3)
        self.assertAlmostEqual(actual["gam"],expected["gam"])

    def test_area(self):
        actual = lookup_table(area_ratio=1.094,gam=1.4)
        expected = {"M":1.360,"p0_ratio":3.00932891965,"t0_ratio":1.3698630137,
                    "r0_ratio":2.19669178218,"a0_ratio":1.17041147196,
                    "area_ratio":1.094,"gam":1.4}
        self.assertAlmostEqual(actual["M"],expected["M"],places=3)
        self.assertAlmostEqual(actual["p0_ratio"],expected["p0_ratio"],places=3)
        self.assertAlmostEqual(actual["t0_ratio"],expected["t0_ratio"],places=3)
        self.assertAlmostEqual(actual["r0_ratio"],expected["r0_ratio"],places=3)
        self.assertAlmostEqual(actual["a0_ratio"],expected["a0_ratio"],places=3)
        self.assertAlmostEqual(actual["area_ratio"],expected["area_ratio"])
        self.assertAlmostEqual(actual["gam"],expected["gam"])

    def test_area2(self):
        actual = lookup_table(area_ratio=3.1,gam=1.3,regime="subsonic")
        expected = {"M":0.193,"p0_ratio":1.02438024995,"t0_ratio":1.0060362173,
                    "r0_ratio":1.01871377199,"a0_ratio":1.00301356785,
                    "area_ratio":3.1,"gam":1.3,"regime":"subsonic"}
        self.assertAlmostEqual(actual["M"],expected["M"],places=3)
        self.assertAlmostEqual(actual["p0_ratio"],expected["p0_ratio"],places=3)
        self.assertAlmostEqual(actual["t0_ratio"],expected["t0_ratio"],places=3)
        self.assertAlmostEqual(actual["r0_ratio"],expected["r0_ratio"],places=3)
        self.assertAlmostEqual(actual["a0_ratio"],expected["a0_ratio"],places=3)
        self.assertAlmostEqual(actual["area_ratio"],expected["area_ratio"])
        self.assertAlmostEqual(actual["gam"],expected["gam"])

    def test_p(self):
        actual = lookup_table(p0_ratio=39.59,gam=1.4)
        expected = {"M":3.05,"p0_ratio":39.59,"t0_ratio":2.860,
                    "r0_ratio":13.84,"a0_ratio":1.69115345253,
                    "area_ratio":4.441,"gam":1.4}
        self.assertAlmostEqual(actual["M"],expected["M"],places=3)
        self.assertAlmostEqual(actual["p0_ratio"],expected["p0_ratio"],places=3)
        self.assertAlmostEqual(actual["t0_ratio"],expected["t0_ratio"],places=2)
        self.assertAlmostEqual(actual["r0_ratio"],expected["r0_ratio"],places=3)
        self.assertAlmostEqual(actual["a0_ratio"],expected["a0_ratio"],places=3)
        self.assertAlmostEqual(actual["area_ratio"],expected["area_ratio"],places=3)
        self.assertAlmostEqual(actual["gam"],expected["gam"])

if __name__ == '__main__':
    unittest.main(verbosity=2)