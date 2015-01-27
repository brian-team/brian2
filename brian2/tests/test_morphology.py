from nose.plugins.attrib import attr
from numpy.testing.utils import assert_equal, assert_allclose, assert_raises
import numpy as np

from brian2.spatialneuron import *
from brian2.units import um, second

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
    assert_raises(AttributeError,lambda :morpho.L.indices[:])
    # Compress the morphology and get absolute compartment indices
    N = len(morpho)
    morpho.compress(MorphologyData(N))
    assert_equal(morpho.LL.indices[:], [11, 12, 13, 14, 15])
    assert_equal(morpho.L.indices[3*um:5.1*um], [3, 4, 5])
    assert_equal(morpho.L.indices[3*um:5.1*um],
                 morpho.L[3*um:5.1*um].indices[:])
    assert_equal(morpho.L.indices[:5.1*um], [1, 2, 3, 4, 5])
    assert_equal(morpho.L.indices[3*um:], [3, 4, 5, 6, 7, 8, 9, 10])
    assert_equal(morpho.L.indices[3.5*um], 4)
    assert_equal(morpho.L.indices[3], 4)
    assert_equal(morpho.L.indices[-1], 10)
    assert_equal(morpho.L.indices[3:5], [4, 5])
    assert_equal(morpho.L.indices[3:], [4, 5, 6, 7, 8, 9, 10])
    assert_equal(morpho.L.indices[:5], [1, 2, 3, 4, 5])

    # Main branch
    assert_equal(len(morpho.L.main), 10)

    # Non-existing branch
    assert_raises(AttributeError, lambda: morpho.axon)

    # Incorrect indexing
    #  wrong units or mixing units
    assert_raises(TypeError, lambda: morpho.indices[3*second:5*second])
    assert_raises(TypeError, lambda: morpho.indices[3.4:5.3])
    assert_raises(TypeError, lambda: morpho.indices[3:5*um])
    assert_raises(TypeError, lambda: morpho.indices[3*um:5])
    #   providing a step
    assert_raises(TypeError, lambda: morpho.indices[3*um:5*um:2*um])
    assert_raises(TypeError, lambda: morpho.indices[3:5:2])
    #   incorrect type
    assert_raises(TypeError, lambda: morpho.indices[object()])


if __name__ == '__main__':
    test_basicshapes()
    test_subgroup()
