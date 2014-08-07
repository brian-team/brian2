from numpy.testing.utils import assert_equal, assert_allclose, assert_raises
from brian2.spatialneuron.morphology import *
from brian2.units.stdunits import um

def test_basicshapes():
    morpho = Soma(diameter=30*um)
    morpho.L = Cylinder(length=10*um, diameter=1*um, n=10)
    morpho.LL = Cylinder(length=5*um, diameter=2*um, n=5)
    morpho.right = Cylinder(length=3*um, diameter=1*um, n=7)
    morpho.right['nextone'] = Cylinder(length=2*um, diameter=1*um, n=3)
    # Check total number of compartments
    assert_equal(len(morpho),26)
    # Check that end point is at distance 15 um from soma
    assert_allclose(morpho.LL.distance[-1],15*um)

def test_subgroup():
    pass

if __name__ == '__main__':
    test_basicshapes()
    test_subgroup()