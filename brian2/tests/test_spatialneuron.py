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


def test_infinitecable():
    '''
    Test simulation of an infinite cable vs. theory (Green function)
    '''
    defaultclock.dt = 0.001*ms

    # Morphology
    diameter = 1*um
    Cm = 1 * uF / cm ** 2
    Ri = 100 * ohm * cm
    N = 500
    morpho=Cylinder(diameter=diameter,length=3*mm,n=N)

    # Passive channels
    gL=1e-4*siemens/cm**2
    eqs='''
    Im=-gL*v : amp/meter**2
    I : amp (point current)
    '''

    neuron = SpatialNeuron(morphology=morpho, model=eqs, Cm=Cm, Ri=Ri)

    taum = Cm/gL # membrane time constant
    rm = 1/(gL * pi * diameter) # membrane resistance per unit length
    ra = (4 * Ri)/(pi * diameter**2) # axial resistance per unit length
    la = sqrt(rm/ra) # space length

    # Monitors
    mon=StateMonitor(neuron,'v',record=N/2-20)

    neuron.I[len(neuron)/2]=1*nA # injecting in the middle
    run(0.02*ms)
    neuron.I=0*amp
    run(3*ms,report='text')

    t = mon.t
    v = mon[N/2-20].v
    # Theory (incorrect near cable ends)
    x = 20*morpho.length[0] * meter
    theory = 1./(la*Cm*pi*diameter)*sqrt(taum/(4*pi*(t+defaultclock.dt)))*\
                 exp(-(t+defaultclock.dt)/taum-taum/(4*(t+defaultclock.dt))*(x/la)**2)
    theory = theory*1*nA*0.02*ms
    assert_allclose(v[t>0.5*ms],theory[t>0.5*ms],rtol=0.01) # 1% error tolerance

if __name__ == '__main__':
    test_construction()
    test_infinitecable()