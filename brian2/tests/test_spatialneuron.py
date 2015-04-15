from numpy.testing.utils import assert_equal, assert_allclose, assert_raises
from nose import with_setup
from nose.plugins.attrib import attr
from brian2 import *
from brian2.devices.device import restore_device

@attr('codegen-independent')
@with_setup(teardown=restore_device)
def test_construction():
    BrianLogger.suppress_name('resolution_conflict')
    morpho = Soma(diameter=30*um)
    morpho.L = Cylinder(length=10*um, diameter=1*um, n=10)
    morpho.LL = Cylinder(length=5*um, diameter=2*um, n=5)
    morpho.LR = Cylinder(length=5*um, diameter=2*um, n=10)
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

    # Check basic consistency of the flattened representation
    assert len(np.unique(neuron.diffusion_state_updater._morph_i[:])) == len(neuron.diffusion_state_updater._morph_i)
    assert all(neuron.diffusion_state_updater._ends[:].flat >=
               neuron.diffusion_state_updater._starts[:].flat)


@attr('long', 'standalone-compatible')
@with_setup(teardown=restore_device)
def test_infinitecable():
    '''
    Test simulation of an infinite cable vs. theory for current pulse (Green function)
    '''
    BrianLogger.suppress_name('resolution_conflict')

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

    # Monitors
    mon=StateMonitor(neuron,'v',record=N/2-20)

    net = Network(neuron, mon)

    neuron.I[len(neuron)//2]=1*nA # injecting in the middle
    net.run(0.02*ms)
    neuron.I=0*amp
    net.run(3*ms)
    t = mon.t
    v = mon[N//2-20].v
    # Theory (incorrect near cable ends)
    x = 20*morpho.length[0] * meter
    la = neuron.space_constant[0]
    taum = Cm/gL # membrane time constant
    theory = 1./(la*Cm*pi*diameter)*sqrt(taum/(4*pi*(t+defaultclock.dt)))*\
                 exp(-(t+defaultclock.dt)/taum-taum/(4*(t+defaultclock.dt))*(x/la)**2)
    theory = theory*1*nA*0.02*ms
    assert_allclose(v[t>0.5*ms],theory[t>0.5*ms],rtol=0.01) # 1% error tolerance (not exact because not infinite cable)

@attr('long', 'standalone-compatible')
@with_setup(teardown=restore_device)
def test_finitecable():
    '''
    Test simulation of short cylinder vs. theory for constant current.
    '''
    BrianLogger.suppress_name('resolution_conflict')

    defaultclock.dt = 0.01*ms

    # Morphology
    diameter = 1*um
    length = 300*um
    Cm = 1 * uF / cm ** 2
    Ri = 150 * ohm * cm
    N = 200
    morpho=Cylinder(diameter=diameter,length=length,n=N)

    # Passive channels
    gL=1e-4*siemens/cm**2
    EL=-70*mV
    eqs='''
    Im=gL*(EL-v) : amp/meter**2
    I : amp (point current)
    '''

    neuron = SpatialNeuron(morphology=morpho, model=eqs, Cm=Cm, Ri=Ri)
    neuron.v = EL

    neuron.I[0]=0.02*nA # injecting at the left end
    net = Network(neuron)
    net.run(100*ms)

    # Theory
    x = neuron.distance
    v = neuron.v
    la = neuron.space_constant[0]
    ra = la*4*Ri/(pi*diameter**2)
    theory = EL+ra*neuron.I[0]*cosh((length-x)/la)/sinh(length/la)
    assert_allclose(v-EL, theory-EL, rtol=0.01)

@attr('long', 'standalone-compatible')
@with_setup(teardown=restore_device)
def test_rall():
    '''
    Test simulation of a cylinder plus two branches, with diameters according to Rall's formula
    '''
    BrianLogger.suppress_name('resolution_conflict')

    defaultclock.dt = 0.01*ms

    # Passive channels
    gL=1e-4*siemens/cm**2
    EL=-70*mV

    # Morphology
    diameter = 1*um
    length = 300*um
    Cm = 1 * uF / cm ** 2
    Ri = 150 * ohm * cm
    N = 500
    rm = 1/(gL * pi * diameter) # membrane resistance per unit length
    ra = (4 * Ri)/(pi * diameter**2) # axial resistance per unit length
    la = sqrt(rm/ra) # space length
    morpho=Cylinder(diameter=diameter,length=length,n=N)
    d1 = 0.5*um
    L1 = 200*um
    rm = 1/(gL * pi * d1) # membrane resistance per unit length
    ra = (4 * Ri)/(pi * d1**2) # axial resistance per unit length
    l1 = sqrt(rm/ra) # space length
    morpho.L=Cylinder(diameter=d1,length=L1,n=N)
    d2 = (diameter**1.5-d1**1.5)**(1./1.5)
    rm = 1/(gL * pi * d2) # membrane resistance per unit length
    ra = (4 * Ri)/(pi * d2**2) # axial resistance per unit length
    l2 = sqrt(rm/ra) # space length
    L2 = (L1/l1)*l2
    morpho.R=Cylinder(diameter=d2,length=L2,n=N)

    eqs='''
    Im=gL*(EL-v) : amp/meter**2
    I : amp (point current)
    '''

    neuron = SpatialNeuron(morphology=morpho, model=eqs, Cm=Cm, Ri=Ri)
    neuron.v = EL

    neuron.I[0]=0.02*nA # injecting at the left end
    run(100*ms)

    # Check space constant calculation
    assert_allclose(la, neuron.space_constant[0])
    assert_allclose(l1, neuron.L.space_constant[0])
    assert_allclose(l2, neuron.R.space_constant[0])

    # Theory
    x = neuron.main.distance
    ra = la*4*Ri/(pi*diameter**2)
    l = length/la + L1/l1
    theory = EL+ra*neuron.I[0]*cosh(l-x/la)/sinh(l)
    v = neuron.main.v
    assert_allclose(v-EL, theory-EL, rtol=0.001)
    x = neuron.L.distance
    theory = EL+ra*neuron.I[0]*cosh(l-neuron.main.distance[-1]/la-(x-neuron.main.distance[-1])/l1)/sinh(l)
    v = neuron.L.v
    assert_allclose(v-EL, theory-EL, rtol=0.001)
    x = neuron.R.distance
    theory = EL+ra*neuron.I[0]*cosh(l-neuron.main.distance[-1]/la-(x-neuron.main.distance[-1])/l2)/sinh(l)
    v = neuron.R.v
    assert_allclose(v-EL, theory-EL, rtol=0.001)

if __name__ == '__main__':
    test_construction()
    test_infinitecable()
    test_finitecable()
    test_rall()
