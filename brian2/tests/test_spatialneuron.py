from numpy.testing.utils import assert_equal, assert_allclose, assert_raises
from brian2.spatialneuron import *
from brian2.units.stdunits import um

def test_spatialneuron():
    morpho = Soma(diameter=30*um)
    morpho.L = Cylinder(length=10*um, diameter=1*um, n=10)
    morpho.LL = Cylinder(length=5*um, diameter=2*um, n=5)
    morpho.right = Cylinder(length=3*um, diameter=1*um, n=7)
    morpho.right.nextone = Cylinder(length=2*um, diameter=1*um, n=3)

if __name__ == '__main__':
    test_spatialneuron()