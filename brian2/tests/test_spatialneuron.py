from numpy.testing.utils import assert_equal, assert_allclose, assert_raises
from brian2 import *

def test_construction():
    morpho = Soma(diameter=30*um)
    morpho.L = Cylinder(length=10*um, diameter=1*um, n=10)
    morpho.LL = Cylinder(length=5*um, diameter=2*um, n=5)
    morpho.right = Cylinder(length=3*um, diameter=1*um, n=7)
    morpho.right.nextone = Cylinder(length=2*um, diameter=1*um, n=3)
    gL=1e-4*siemens/cm**2
    EL=-70*mV
    eqs='''
    Im=gL*(EL-v) : amp/meter**2
    I : meter (point current)
    '''
    # Check units of currents
    assert_raises(DimensionMismatchError,lambda :SpatialNeuron(morphology=morpho, model=eqs))

    eqs='''
    Im=gL*(EL-v) : amp/meter**2
    '''
    neuron = SpatialNeuron(morphology=morpho, model=eqs, Cm=1 * uF / cm ** 2, Ri=100 * ohm * cm)
    # Test initialization of values
    neuron.LL.v = EL
    assert_allclose(neuron.L.main.v,0)
    assert_allclose(neuron.LL.v,EL)
    neuron.LL[2*um:3.1*um].v = 0*mV
    assert_allclose(neuron.LL.v,[EL,0,0,EL,EL])
    assert_allclose(neuron.Cm,1 * uF / cm ** 2)

    # Test morphological variables
    assert_allclose(neuron.main.x,morpho.x)
    assert_allclose(neuron.L.main.x,morpho.L.x)
    assert_allclose(neuron.LL.main.x,morpho.LL.x)
    assert_allclose(neuron.right.main.x,morpho.right.x)
    assert_allclose(neuron.L.main.distance,morpho.L.distance)
    assert_allclose(neuron.L.main.diameter,morpho.L.diameter)
    assert_allclose(neuron.L.main.area,morpho.L.area)
    assert_allclose(neuron.L.main.length,morpho.L.length)


def test_passive():
    '''
    Test simulations of neurons with passive conductances
    '''
    pass

if __name__ == '__main__':
    test_construction()
    test_passive()