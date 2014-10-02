from nose.plugins.attrib import attr
from numpy.testing.utils import assert_equal, assert_allclose, assert_raises
from brian2.spatialneuron import *
from brian2.units.stdunits import um

@attr('codegen-independent')
def test_basicshapes():
    morpho = Soma(diameter=30*um)
    morpho.L = Cylinder(length=10*um, diameter=1*um, n=10)
    morpho.LL = Cylinder(length=5*um, diameter=2*um, n=5)
    morpho.right = Cylinder(length=3*um, diameter=1*um, n=7)
    morpho.right['nextone'] = Cylinder(length=2*um, diameter=1*um, n=3)
    # Check total number of compartments
    assert_equal(len(morpho),26)
    assert_equal(len(morpho.L.main),10)
    # Check that end point is at distance 15 um from soma
    assert_allclose(morpho.LL.distance[-1],15*um)

@attr('codegen-independent')
def test_subgroup():
    morpho = Soma(diameter=30*um)
    morpho.L = Cylinder(length=10*um, diameter=1*um, n=10)
    morpho.LL = Cylinder(length=5*um, diameter=2*um, n=5)
    morpho.right = Cylinder(length=3*um, diameter=1*um, n=7)
    # Getting a single compartment by index
    assert_allclose(morpho.L[2].distance,3*um)
    # Getting a single compartment by position
    assert_allclose(morpho.LL[0*um].distance,11*um)
    assert_allclose(morpho.LL[1*um].distance,11*um)
    assert_allclose(morpho.LL[1.5*um].distance,12*um)
    assert_allclose(morpho.LL[5*um].distance,15*um)
    # Getting a segment
    assert_allclose(morpho.L[3*um:5.1*um].distance,[3*um,4*um,5*um])
    # Indices cannot be obtained at this stage
    assert_raises(AttributeError,lambda :morpho.L.indices())
    # Compress the morphology and get absolute compartment indices
    neuron = SpatialNeuron(morphology=morpho,model='Im = 0*amp/meter**2 : amp/meter**2')
    assert_equal(morpho.LL.indices(),[11,12,13,14,15])
    assert_allclose(morpho.L[3*um:5.1*um].indices(),[3,4,5])
    # Main branch
    assert_equal(len(morpho.L.main),10)
    # Non-existing branch
    assert_raises(AttributeError,lambda :morpho.axon)

if __name__ == '__main__':
    test_basicshapes()
    test_subgroup()