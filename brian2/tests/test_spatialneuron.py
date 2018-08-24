import os
import itertools

from numpy.testing.utils import assert_equal, assert_raises
from nose import with_setup, SkipTest
from nose.plugins.attrib import attr

from brian2 import *
from brian2.devices.device import reinit_devices
from brian2.tests.utils import assert_allclose


@attr('codegen-independent')
def test_custom_events():
    # Set (could be moved in a setup)
    EL = -65*mV
    gL = 0.0003*siemens/cm**2
    ev = '''
    Im = gL * (EL - v) : amp/meter**2
    event_time1 : second
    '''
    # Create a three compartments morphology
    morpho = Soma(diameter=10*um)
    morpho.dend1 = Cylinder(n=1, diameter=1*um, length=10*um )
    morpho.dend2 = Cylinder(n=1, diameter=1*um, length=10*um )
    G = SpatialNeuron(morphology=morpho,
                      model=ev,
                      events={'event1': 't>=i*ms and t<i*ms+dt'})
    G.run_on_event('event1', 'event_time1 = 0.1*ms')
    run(0.2*ms)
    # Event has size three now because there are three compartments
    assert_allclose(G.event_time1[:], [0.1, 0, 0]*ms)

@attr('codegen-independent')
@with_setup(teardown=reinit_devices)
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
    assert_raises(DimensionMismatchError, lambda: SpatialNeuron(morphology=morpho,
                                                                model=eqs))

    eqs='''
    Im=gL*(EL-v) : amp/meter**2
    '''
    neuron = SpatialNeuron(morphology=morpho, model=eqs, Cm=1 * uF / cm ** 2, Ri=100 * ohm * cm)
    # Test initialization of values
    neuron.LL.v = EL
    assert_allclose(neuron.L.main.v, 0*mV)
    assert_allclose(neuron.LL.v, EL)
    neuron.LL[1*um:3*um].v = 0*mV
    assert_allclose(neuron.LL.v, Quantity([EL, 0*mV, 0*mV, EL, EL]))
    assert_allclose(neuron.Cm, 1 * uF / cm ** 2)

    # Test morphological variables
    assert_allclose(neuron.L.main.distance, morpho.L.distance)
    assert_allclose(neuron.L.main.area, morpho.L.area)
    assert_allclose(neuron.L.main.length, morpho.L.length)

    # Check basic consistency of the flattened representation
    assert all(neuron.diffusion_state_updater._ends[:].flat >=
               neuron.diffusion_state_updater._starts[:].flat)

    # Check that length and distances make sense
    assert_allclose(sum(morpho.L.length), 10*um)
    assert_allclose(morpho.L.distance, (0.5 + np.arange(10))*um)
    assert_allclose(sum(morpho.LL.length), 5*um)
    assert_allclose(morpho.LL.distance, (10 + .5 + np.arange(5))*um)
    assert_allclose(sum(morpho.LR.length), 5*um)
    assert_allclose(morpho.LR.distance, (10 + 0.25 + np.arange(10)*0.5)*um)
    assert_allclose(sum(morpho.right.length), 3*um)
    assert_allclose(morpho.right.distance, (0.5 + np.arange(7))*3./7.*um)
    assert_allclose(sum(morpho.right.nextone.length), 2*um)
    assert_allclose(morpho.right.nextone.distance, 3*um + (0.5 + np.arange(3))*2./3.*um)


@attr('codegen-independent')
@with_setup(teardown=reinit_devices)
def test_construction_coordinates():
    # Same as test_construction, but uses coordinates instead of lengths to
    # set up everything
    # Note that all coordinates here are relative to the origin of the
    # respective cylinder
    BrianLogger.suppress_name('resolution_conflict')
    morpho = Soma(diameter=30*um)
    morpho.L = Cylinder(x=[0, 10]*um, diameter=1*um, n=10)
    morpho.LL = Cylinder(y=[0, 5]*um, diameter=2*um, n=5)
    morpho.LR = Cylinder(z=[0, 5]*um, diameter=2*um, n=10)
    morpho.right = Cylinder(x=[0, sqrt(2)*1.5]*um, y=[0, sqrt(2)*1.5]*um,
                            diameter=1*um, n=7)
    morpho.right.nextone = Cylinder(y=[0, sqrt(2)]*um, z=[0, sqrt(2)]*um,
                                    diameter=1*um, n=3)
    gL=1e-4*siemens/cm**2
    EL=-70*mV
    eqs='''
    Im=gL*(EL-v) : amp/meter**2
    I : meter (point current)
    '''

    # Check units of currents
    assert_raises(DimensionMismatchError, lambda: SpatialNeuron(morphology=morpho,
                                                                model=eqs))

    eqs='''
    Im=gL*(EL-v) : amp/meter**2
    '''
    neuron = SpatialNeuron(morphology=morpho, model=eqs, Cm=1 * uF / cm ** 2, Ri=100 * ohm * cm)

    # Test initialization of values
    neuron.LL.v = EL
    assert_allclose(neuron.L.main.v, 0*mV)
    assert_allclose(neuron.LL.v, EL)
    neuron.LL[1*um:3*um].v = 0*mV
    assert_allclose(neuron.LL.v, Quantity([EL, 0*mV, 0*mV, EL, EL]))
    assert_allclose(neuron.Cm, 1 * uF / cm ** 2)

    # Test morphological variables
    assert_allclose(neuron.L.main.x, morpho.L.x)
    assert_allclose(neuron.LL.main.x, morpho.LL.x)
    assert_allclose(neuron.right.main.x, morpho.right.x)
    assert_allclose(neuron.L.main.distance, morpho.L.distance)
    # assert_allclose(neuron.L.main.diameter, morpho.L.diameter)
    assert_allclose(neuron.L.main.area, morpho.L.area)
    assert_allclose(neuron.L.main.length, morpho.L.length)

    # Check basic consistency of the flattened representation
    assert all(neuron.diffusion_state_updater._ends[:].flat >=
               neuron.diffusion_state_updater._starts[:].flat)

    # Check that length and distances make sense
    assert_allclose(sum(morpho.L.length), 10*um)
    assert_allclose(morpho.L.distance, (0.5 + np.arange(10))*um)
    assert_allclose(sum(morpho.LL.length), 5*um)
    assert_allclose(morpho.LL.distance, (10 + .5 + np.arange(5))*um)
    assert_allclose(sum(morpho.LR.length), 5*um)
    assert_allclose(morpho.LR.distance, (10 + 0.25 + np.arange(10)*0.5)*um)
    assert_allclose(sum(morpho.right.length), 3*um)
    assert_allclose(morpho.right.distance, (0.5 + np.arange(7))*3./7.*um)
    assert_allclose(sum(morpho.right.nextone.length), 2*um)
    assert_allclose(morpho.right.nextone.distance, 3*um + (0.5 + np.arange(3))*2./3.*um)


@attr('long')
@with_setup(teardown=reinit_devices)
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

    neuron.I[len(neuron)//2]=1*nA # injecting in the middle
    run(0.02*ms)
    neuron.I=0*amp
    run(3*ms)
    t = mon.t
    v = mon[N//2-20].v
    # Theory (incorrect near cable ends)
    x = 20*morpho.length[0]
    la = neuron.space_constant[0]
    taum = Cm/gL # membrane time constant
    theory = 1./(la*Cm*pi*diameter)*sqrt(taum/(4*pi*(t+defaultclock.dt)))*\
                 exp(-(t+defaultclock.dt)/taum-taum/(4*(t+defaultclock.dt))*(x/la)**2)
    theory = theory*1*nA*0.02*ms
    assert_allclose(v[t>0.5*ms],theory[t>0.5*ms], rtol=1e14, atol=1e10) # high error tolerance (not exact because not infinite cable)

@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_finitecable():
    '''
    Test simulation of short cylinder vs. theory for constant current.
    '''
    if prefs.core.default_float_dtype is np.float32:
        raise SkipTest('Need double precision for this test')
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

    run(100*ms)

    # Theory
    x = neuron.distance
    v = neuron.v
    la = neuron.space_constant[0]
    ra = la*4*Ri/(pi*diameter**2)
    theory = EL+ra*neuron.I[0]*cosh((length-x)/la)/sinh(length/la)
    assert_allclose(v-EL, theory-EL, rtol=1e12, atol=1e8)

@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_rallpack1():
    '''
    Rallpack 1
    '''
    if prefs.core.default_float_dtype is np.float32:
        raise SkipTest('Need double precision for this test')
    defaultclock.dt = 0.05*ms

    # Morphology
    diameter = 1*um
    length = 1*mm
    Cm = 1 * uF / cm ** 2
    Ri = 100 * ohm * cm
    N = 1000
    morpho = Cylinder(diameter=diameter, length=length, n=N)

    # Passive channels
    gL = 1./(40000*ohm*cm**2)
    EL = -65*mV
    eqs = '''
    Im = gL*(EL - v) : amp/meter**2
    I : amp (point current, constant)
    '''
    neuron = SpatialNeuron(morphology=morpho, model=eqs, Cm=Cm, Ri=Ri)
    neuron.v = EL

    neuron.I[0] = 0.1*nA  # injecting at the left end

    #Record at the two ends
    mon = StateMonitor(neuron, 'v', record=[0, 999], when='start', dt=0.05*ms)

    run(250*ms + defaultclock.dt)

    # Load the theoretical results
    basedir = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                           'rallpack_data')
    data_0 = np.loadtxt(os.path.join(basedir, 'ref_cable.0'))
    data_x = np.loadtxt(os.path.join(basedir, 'ref_cable.x'))

    scale_0 = max(data_0[:, 1]*volt) - min(data_0[:, 1]*volt)
    scale_x = max(data_x[:, 1]*volt) - min(data_x[:, 1]*volt)
    squared_diff_0 = (data_0[:, 1] * volt - mon[0].v)**2
    squared_diff_x = (data_x[:, 1] * volt - mon[999].v)**2
    rel_RMS_0 = sqrt(mean(squared_diff_0))/scale_0
    rel_RMS_x = sqrt(mean(squared_diff_x))/scale_x
    max_rel_0 = sqrt(max(squared_diff_0))/scale_0
    max_rel_x = sqrt(max(squared_diff_x))/scale_x

    # sanity check: times are the same
    assert_allclose(mon.t/second, data_0[:, 0])
    assert_allclose(mon.t/second, data_x[:, 0])

    # RMS error should be < 0.1%, maximum error along the curve should be < 0.5%
    assert 100*rel_RMS_0 < 0.1
    assert 100*rel_RMS_x < 0.1
    assert 100*max_rel_0 < 0.5
    assert 100*max_rel_x < 0.5


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_rallpack2():
    '''
    Rallpack 2
    '''
    if prefs.core.default_float_dtype is np.float32:
        raise SkipTest('Need double precision for this test')
    defaultclock.dt = 0.1*ms

    # Morphology
    diameter = 32*um
    length = 16*um
    Cm = 1 * uF / cm ** 2
    Ri = 100 * ohm * cm

    # Construct binary tree according to Rall's formula
    morpho = Cylinder(n=1, diameter=diameter, y=[0, float(length)]*meter)
    endpoints = {morpho}
    for depth in xrange(1, 10):
        diameter /= 2.**(1./3.)
        length /= 2.**(2./3.)
        new_endpoints = set()
        for endpoint in endpoints:
            new_L = Cylinder(n=1, diameter=diameter, length=length)
            new_R = Cylinder(n=1, diameter=diameter, length=length)
            new_endpoints.add(new_L)
            new_endpoints.add(new_R)
            endpoint.L = new_L
            endpoint.R = new_R
        endpoints = new_endpoints

    # Passive channels
    gL = 1./(40000*ohm*cm**2)
    EL = -65*mV
    eqs = '''
    Im = gL*(EL - v) : amp/meter**2
    I : amp (point current, constant)
    '''
    neuron = SpatialNeuron(morphology=morpho, model=eqs, Cm=Cm, Ri=Ri,
                           method='rk4')
    neuron.v = EL

    neuron.I[0] = 0.1*nA  # injecting at the origin

    endpoint_indices = [endpoint.indices[0] for endpoint in endpoints]
    mon = StateMonitor(neuron, 'v', record=[0] + endpoint_indices,
                       when='start', dt=0.1*ms)

    run(250*ms + defaultclock.dt)

    # Load the theoretical results
    basedir = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                           'rallpack_data')
    # Only use very second time step, since we run with 0.1ms instead of 0.05ms
    data_0 = np.loadtxt(os.path.join(basedir, 'ref_branch.0'))[::2]
    data_x = np.loadtxt(os.path.join(basedir, 'ref_branch.x'))[::2]

    # sanity check: times are the same
    assert_allclose(mon.t/second, data_0[:, 0])
    assert_allclose(mon.t/second, data_x[:, 0])

    # Check that all endpoints are the same:
    for endpoint in endpoints:
        assert_allclose(mon[endpoint].v, mon[endpoint[0]].v)

    scale_0 = max(data_0[:, 1]*volt) - min(data_0[:, 1]*volt)
    scale_x = max(data_x[:, 1]*volt) - min(data_x[:, 1]*volt)
    squared_diff_0 = (data_0[:, 1] * volt - mon[0].v)**2

    # One endpoint
    squared_diff_x = (data_x[:, 1] * volt - mon[endpoint_indices[0]].v)**2
    rel_RMS_0 = sqrt(mean(squared_diff_0))/scale_0
    rel_RMS_x = sqrt(mean(squared_diff_x))/scale_x
    max_rel_0 = sqrt(max(squared_diff_0))/scale_0
    max_rel_x = sqrt(max(squared_diff_x))/scale_x

    # RMS error should be < 0.25%, maximum error along the curve should be < 0.5%
    assert 100*rel_RMS_0 < 0.25
    assert 100*rel_RMS_x < 0.25
    assert 100*max_rel_0 < 0.5
    assert 100*max_rel_x < 0.5


@attr('long', 'standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_rallpack3():
    '''
    Rallpack 3
    '''
    if prefs.core.default_float_dtype is np.float32:
        raise SkipTest('Need double precision for this test')
    defaultclock.dt = 1*usecond

    # Morphology
    diameter = 1*um
    length = 1*mm
    N = 1000
    morpho = Cylinder(diameter=diameter, length=length, n=N)
    # Passive properties
    gl = 1./(40000*ohm*cm**2)
    El = -65*mV
    Cm = 1 * uF / cm ** 2
    Ri = 100 * ohm * cm
    # Active properties
    ENa = 50*mV
    EK = -77*mV
    gNa = 120*msiemens/cm**2
    gK = 36*msiemens/cm**2
    eqs = '''
    Im = gl * (El-v) + gNa * m**3 * h * (ENa-v) + gK * n**4 * (EK-v) : amp/meter**2
    dm/dt = alpham * (1-m) - betam * m : 1
    dn/dt = alphan * (1-n) - betan * n : 1
    dh/dt = alphah * (1-h) - betah * h : 1
    v_shifted = v - El : volt
    alpham = (0.1/mV) * (-v_shifted+25*mV) / (exp((-v_shifted+25*mV) / (10*mV)) - 1)/ms : Hz
    betam = 4 * exp(-v_shifted/(18*mV))/ms : Hz
    alphah = 0.07 * exp(-v_shifted/(20*mV))/ms : Hz
    betah = 1/(exp((-v_shifted+30*mV) / (10*mV)) + 1)/ms : Hz
    alphan = (0.01/mV) * (-v_shifted+10*mV) / (exp((-v_shifted+10*mV) / (10*mV)) - 1)/ms : Hz
    betan = 0.125*exp(-v_shifted/(80*mV))/ms : Hz
    I : amp (point current, constant)
    '''
    axon = SpatialNeuron(morphology=morpho, model=eqs, Cm=Cm, Ri=Ri, method='exponential_euler')
    axon.v = El
    # Pre-calculated equilibrium values at v = El
    axon.m = 0.0529324852572
    axon.n = 0.317676914061
    axon.h = 0.596120753508
    axon.I[0] = 0.1*nA  # injecting at the left end

    #Record at the two ends
    mon = StateMonitor(axon, 'v', record=[0, 999], when='start', dt=0.05*ms)

    run(250*ms + defaultclock.dt)


    # Load the theoretical results
    basedir = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                           'rallpack_data')
    data_0 = np.loadtxt(os.path.join(basedir, 'ref_axon.0.neuron'))
    data_x = np.loadtxt(os.path.join(basedir, 'ref_axon.x.neuron'))

    # sanity check: times are the same
    assert_allclose(mon.t/second, data_0[:, 0])
    assert_allclose(mon.t/second, data_x[:, 0])

    scale_0 = max(data_0[:, 1]*volt) - min(data_0[:, 1]*volt)
    scale_x = max(data_x[:, 1]*volt) - min(data_x[:, 1]*volt)
    squared_diff_0 = (data_0[:, 1] * volt - mon[0].v)**2
    squared_diff_x = (data_x[:, 1] * volt - mon[999].v)**2

    rel_RMS_0 = sqrt(mean(squared_diff_0))/scale_0
    rel_RMS_x = sqrt(mean(squared_diff_x))/scale_x
    max_rel_0 = sqrt(max(squared_diff_0))/scale_0
    max_rel_x = sqrt(max(squared_diff_x))/scale_x

    # RMS error should be < 0.1%, maximum error along the curve should be < 0.5%
    # Note that this is much stricter than the original Rallpack evaluation, but
    # with the 1us time step, the voltage traces are extremely similar
    assert 100*rel_RMS_0 < 0.1
    assert 100*rel_RMS_x < 0.1
    assert 100*max_rel_0 < 0.5
    assert 100*max_rel_x < 0.5


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_rall():
    '''
    Test simulation of a cylinder plus two branches, with diameters according to Rall's formula
    '''
    if prefs.core.default_float_dtype is np.float32:
        raise SkipTest('Need double precision for this test')
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
    assert_allclose(v-EL, theory-EL, rtol=1e12, atol=1e8)
    x = neuron.L.distance
    theory = EL+ra*neuron.I[0]*cosh(l-neuron.main.distance[-1]/la-(x-neuron.main.distance[-1])/l1)/sinh(l)
    v = neuron.L.v
    assert_allclose(v-EL, theory-EL, rtol=1e12, atol=1e8)
    x = neuron.R.distance
    theory = EL+ra*neuron.I[0]*cosh(l-neuron.main.distance[-1]/la-(x-neuron.main.distance[-1])/l2)/sinh(l)
    v = neuron.R.v
    assert_allclose(v-EL, theory-EL, rtol=1e12, atol=1e8)

@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_basic_diffusion():
    # A very basic test that shows that propagation is working in a very basic
    # sense, testing all morphological classes

    defaultclock.dt = 0.01*ms

    EL = -70*mV
    gL = 1e-4*siemens/cm**2
    target = -10*mV
    eqs = '''
    Im = gL*(EL-v) + gClamp*(target-v): amp/meter**2
    gClamp : siemens/meter**2
    '''

    morph = Soma(diameter=30*um)
    morph.axon = Cylinder(n=10, diameter=10*um, length=100*um)
    morph.dend = Section(n=10, diameter=[10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0.1]*um,
                         length=np.ones(10)*10*um)

    neuron = SpatialNeuron(morph, eqs)
    neuron.v = EL
    neuron.axon.gClamp[0] = 100*siemens/cm**2

    mon = StateMonitor(neuron, 'v', record=True)

    run(0.25*ms)
    assert all(abs(mon.v[:, -1]/mV + 10) < 0.25), mon.v[:, -1]/mV


@attr('codegen-independent')
def test_allowed_integration():
    morph = Soma(diameter=30 * um)
    EL = -70 * mV
    gL = 1e-4 * siemens / cm ** 2
    ENa = 115 * mV
    gNa = 120 * msiemens / cm ** 2
    VT = -50.4 * mV
    DeltaT = 2 * mV
    ENMDA = 0. * mV

    @check_units(voltage=volt, result=volt)
    def user_fun(voltage):
        return voltage  # could be an arbitrary function and is therefore unsafe
    allowed_eqs = ['Im = gL*(EL-v) : amp/meter**2',
                   '''Im = gl * (El-v) + gNa * m**3 * h * (ENa-v) : amp/meter**2
                      dm/dt = alpham * (1-m) - betam * m : 1
                      dh/dt = alphah * (1-h) - betah * h : 1
                      alpham = (0.1/mV) * (-v+25*mV) / (exp((-v+25*mV) / (10*mV)) - 1)/ms : Hz
                      betam = 4 * exp(-v/(18*mV))/ms : Hz
                      alphah = 0.07 * exp(-v/(20*mV))/ms : Hz
                      betah = 1/(exp((-v+30*mV) / (10*mV)) + 1)/ms : Hz''',
                   '''Im = gl * (El-v) : amp/meter**2
                      I_ext = 1*nA + sin(2*pi*100*Hz*t)*nA : amp (point current)''',
                   '''Im = I_leak + I_spike : amp/meter**2
                      I_leak = gL*(EL - v) : amp/meter**2
                      I_spike = gL*DeltaT*exp((v - VT)/DeltaT): amp/meter**2 (constant over dt)
                   ''',
                   '''
                   Im = gL*(EL-v) : amp/meter**2
                   I_NMDA = gNMDA*(ENMDA-v)*Mgblock : amp (point current)
                   gNMDA : siemens
                   Mgblock = 1./(1. +  exp(-0.062*v/mV)/3.57) : 1 (constant over dt)
                   ''',
                   'Im = gL*(EL - v) + gL*DeltaT*exp((v - VT)/DeltaT) : amp/meter**2',
                   '''Im = I_leak + I_spike : amp/meter**2
                      I_leak = gL*(EL - v) : amp/meter**2
                      I_spike = gL*DeltaT*exp((v - VT)/DeltaT): amp/meter**2
                   ''',
                   '''
                   Im = gL*(EL-v) : amp/meter**2
                   I_NMDA = gNMDA*(ENMDA-v)*Mgblock : amp (point current)
                   gNMDA : siemens
                   Mgblock = 1./(1. +  exp(-0.062*v/mV)/3.57) : 1
                   ''',
                   ]
    forbidden_eqs = [
                    '''Im = gl * (El-v + user_fun(v)) : amp/meter**2''',
                    '''Im = gl * clip(El-v, -100*mV, 100*mV) : amp/meter**2''',
                    ]
    for eqs in allowed_eqs:
        # Should not raise an error
        neuron = SpatialNeuron(morph, eqs)

    for eqs in forbidden_eqs:
        # Should raise an error
        assert_raises(TypeError, SpatialNeuron, morph, eqs)

@attr('codegen-independent')
def test_spatialneuron_indexing():
    sec = Cylinder(length=50*um, diameter=10*um, n=1)
    sec.sec1 = Cylinder(length=50 * um, diameter=10 * um, n=2)
    sec.sec1.sec11 = Cylinder(length=50 * um, diameter=10 * um, n=4)
    sec.sec1.sec12 = Cylinder(length=50 * um, diameter=10 * um, n=8)
    sec.sec2 = Cylinder(length=50 * um, diameter=10 * um, n=16)
    sec.sec2.sec21 = Cylinder(length=50 * um, diameter=10 * um, n=32)
    neuron = SpatialNeuron(sec, 'Im = 0*amp/meter**2 : amp/meter**2')

    # Accessing indices/variables of a subtree refers to the full subtree
    assert len(neuron.indices[:]) == 1 + 2 + 4 + 8 + 16 + 32
    assert len(neuron.sec1.indices[:]) == 2 + 4 + 8
    assert len(neuron.sec1.sec11.indices[:]) == 4
    assert len(neuron.sec1.sec12.indices[:]) == 8
    assert len(neuron.sec2.indices[:]) == 16 + 32
    assert len(neuron.sec2.sec21.indices[:]) == 32
    assert len(neuron.v[:]) == 1 + 2 + 4 + 8 + 16 + 32
    assert len(neuron.sec1.v[:]) == 2 + 4 + 8
    assert len(neuron.sec1.sec11.v[:]) == 4
    assert len(neuron.sec1.sec12.v[:]) == 8
    assert len(neuron.sec2.v[:]) == 16 + 32
    assert len(neuron.sec2.sec21.v[:]) == 32
    # Accessing indices/variables with ".main" only refers to the section
    assert len(neuron.main.indices[:]) == 1
    assert len(neuron.sec1.main.indices[:]) == 2
    assert len(neuron.sec1.sec11.main.indices[:]) == 4
    assert len(neuron.sec1.sec12.main.indices[:]) == 8
    assert len(neuron.sec2.main.indices[:]) == 16
    assert len(neuron.sec2.sec21.main.indices[:]) == 32
    assert len(neuron.main.v[:]) == 1
    assert len(neuron.sec1.main.v[:]) == 2
    assert len(neuron.sec1.sec11.main.v[:]) == 4
    assert len(neuron.sec1.sec12.main.v[:]) == 8
    assert len(neuron.sec2.main.v[:]) == 16
    assert len(neuron.sec2.sec21.main.v[:]) == 32

@attr('codegen-independent')
def test_tree_index_consistency():
    # Test all possible trees with depth 3 and a maximum of 3 branches subtree
    # (a total of 84 trees)
    # This tests whether the indices (i.e. where the compartments are placed in
    # the overall flattened 1D structure) make sense: for the `SpatialSubgroup`
    # mechanism to work correctly, each subtree has to have contiguous indices.
    # Separate subtrees should of course have non-overlapping indices.
    for tree_description in itertools.product([1, 2, 3],  # children of root
                                              [0, 1, 2, 3], # children of first branch
                                              [0, 1, 2, 3], # children of second branch
                                              [0, 1, 2, 3]  # children of third branch
                                              ):
        sec = Cylinder(length=50 * um, diameter=10 * um, n=1)
        root_children = tree_description[0]
        if not all([tree_description[x] == 0 for x in xrange(root_children + 1, 4)]):
            # skip redundant descriptions (differing number of branches in a
            # subtree that does not exist)
            continue

        # Create a tree according to the description
        for idx in xrange(root_children):
            setattr(sec, 'sec%d' % (idx + 1),
                    Cylinder(length=50*um, diameter=10*um, n=2*(idx + 1)))
        for child in xrange(root_children):
            subsec = getattr(sec, 'sec%d' % (child + 1))
            subsec_children = tree_description[child + 1]
            for idx in xrange(subsec_children):
                setattr(subsec, 'sec%d%d' % (child + 1, idx + 1),
                        Cylinder(length=50 * um, diameter=10 * um, n=1 + (child + 1) * idx))

        neuron = SpatialNeuron(sec, 'Im = 0*amp/meter**2 : amp/meter**2')
        # Check the indicies for the full neuron:
        assert_equal(neuron.indices[:], np.arange(sec.total_compartments))

        all_subsec_indices = []
        for child in xrange(root_children):
            subsec = getattr(neuron, 'sec%d' % (child + 1))
            sub_indices = set(subsec.main.indices[:])
            subsec_children = tree_description[child + 1]
            for idx in xrange(subsec_children):
                subsubsec = getattr(subsec, 'sec%d%d' % (child + 1, idx + 1))
                sub_indices |= set(subsubsec.main.indices[:])
            # The indices for a full subtree should be the union of the indices
            # for all subsections within that subtree
            assert sub_indices == set(subsec.indices[:])
            all_subsec_indices.extend(subsec.indices[:])
        # Separate subtrees should not overlap
        assert len(all_subsec_indices) == len(set(all_subsec_indices))

@attr('codegen-independent')
def test_spatialneuron_subtree_assignment():
    sec = Cylinder(length=50 * um, diameter=10 * um, n=2)
    sec.sec1 = Cylinder(length=50 * um, diameter=10 * um, n=2)
    sec.sec1.sec11 = Cylinder(length=50 * um, diameter=10 * um, n=2)
    sec.sec1.sec12 = Cylinder(length=50 * um, diameter=10 * um, n=2)
    sec.sec2 = Cylinder(length=50 * um, diameter=10 * um, n=2)
    sec.sec2.sec21 = Cylinder(length=50 * um, diameter=10 * um, n=2)
    neuron = SpatialNeuron(sec, 'Im = 0*amp/meter**2 : amp/meter**2')

    neuron.v = 1*volt
    assert_allclose(neuron.v[:], np.ones(12)*volt)
    neuron.sec1.v += 1*volt
    assert_allclose(neuron.main.v[:], np.ones(2)*volt)
    assert_allclose(neuron.sec1.v[:], np.ones(6)*2*volt)
    assert_allclose(neuron.sec1.main.v[:], np.ones(2)*2*volt)
    assert_allclose(neuron.sec1.sec11.v[:], np.ones(2)*2*volt)
    assert_allclose(neuron.sec1.sec12.v[:], np.ones(2)*2*volt)
    assert_allclose(neuron.sec2.v[:], np.ones(4)*volt)
    neuron.sec2.v = 5*volt
    assert_allclose(neuron.sec2.v[:], np.ones(4)*5*volt)
    assert_allclose(neuron.sec2.main.v[:], np.ones(2)*5*volt)
    assert_allclose(neuron.sec2.sec21.v[:], np.ones(2)*5*volt)


@attr('codegen-independent')
def test_spatialneuron_morphology_assignment():
    sec = Cylinder(length=50 * um, diameter=10 * um, n=2)
    sec.sec1 = Cylinder(length=50 * um, diameter=10 * um, n=2)
    sec.sec1.sec11 = Cylinder(length=50 * um, diameter=10 * um, n=2)
    sec.sec1.sec12 = Cylinder(length=50 * um, diameter=10 * um, n=2)
    sec.sec2 = Cylinder(length=50 * um, diameter=10 * um, n=2)
    sec.sec2.sec21 = Cylinder(length=50 * um, diameter=10 * um, n=2)
    neuron = SpatialNeuron(sec, 'Im = 0*amp/meter**2 : amp/meter**2')

    neuron.v[sec.sec1.sec11] = 1*volt
    assert_allclose(neuron.sec1.sec11.v[:], np.ones(2)*volt)
    assert_allclose(neuron.sec1.sec12.v[:], np.zeros(2)*volt)
    assert_allclose(neuron.sec1.main.v[:], np.zeros(2)*volt)
    assert_allclose(neuron.main.v[:], np.zeros(2)*volt)
    assert_allclose(neuron.sec2.v[:], np.zeros(4)*volt)

    neuron.v[sec.sec2[25*um:]] = 2*volt
    neuron.v[sec.sec1[:25*um]] = 3 * volt
    assert_allclose(neuron.main.v[:], np.zeros(2)*volt)
    assert_allclose(neuron.sec2.main.v[:], [0, 2]*volt)
    assert_allclose(neuron.sec2.sec21.v[:], np.zeros(2)*volt)
    assert_allclose(neuron.sec1.main.v[:], [3, 0]*volt)
    assert_allclose(neuron.sec1.sec11.v[:], np.ones(2)*volt)
    assert_allclose(neuron.sec1.sec12.v[:], np.zeros(2)*volt)

@attr('standalone-compatible', 'multiple-runs')
@with_setup(teardown=reinit_devices)
def test_spatialneuron_capacitive_currents():
    if prefs.core.default_float_dtype is np.float32:
        raise SkipTest('Need double precision for this test')
    defaultclock.dt = 0.1*ms
    morpho = Cylinder(x=[0, 10]*cm, diameter=2*238*um, n=200, type='axon')

    El = 10.613* mV
    ENa = 115*mV
    EK = -12*mV
    gl = 0.3*msiemens/cm**2
    gNa0 = 120*msiemens/cm**2
    gK = 36*msiemens/cm**2

    # Typical equations
    eqs = '''
    # The same equations for the whole neuron, but possibly different parameter values
    # distributed transmembrane current
    Im = gl * (El-v) + gNa * m**3 * h * (ENa-v) + gK * n**4 * (EK-v) : amp/meter**2
    I : amp (point current) # applied current
    dm/dt = alpham * (1-m) - betam * m : 1
    dn/dt = alphan * (1-n) - betan * n : 1
    dh/dt = alphah * (1-h) - betah * h : 1
    alpham = (0.1/mV) * (-v+25*mV) / (exp((-v+25*mV) / (10*mV)) - 1)/ms : Hz
    betam = 4 * exp(-v/(18*mV))/ms : Hz
    alphah = 0.07 * exp(-v/(20*mV))/ms : Hz
    betah = 1/(exp((-v+30*mV) / (10*mV)) + 1)/ms : Hz
    alphan = (0.01/mV) * (-v+10*mV) / (exp((-v+10*mV) / (10*mV)) - 1)/ms : Hz
    betan = 0.125*exp(-v/(80*mV))/ms : Hz
    gNa : siemens/meter**2
    '''

    neuron = SpatialNeuron(morphology=morpho, model=eqs, Cm=1*uF/cm**2,
                           Ri=35.4*ohm*cm, method="exponential_euler")
    mon = StateMonitor(neuron, ['Im', 'Ic'], record=True, when='end')
    run(10*ms)
    neuron.I[0] = 1*uA  # current injection at one end
    run(3*ms)
    neuron.I = 0*amp
    run(10*ms)
    device.build(direct_call=False, **device.build_options)
    assert_allclose((mon.Im-mon.Ic).sum(axis=0)/(mA/cm**2), np.zeros(230),
                    atol=1e6)

if __name__ == '__main__':
    test_custom_events()
    test_construction()
    test_construction_coordinates()
    test_infinitecable()
    test_finitecable()
    test_rallpack1()
    test_rallpack2()
    test_rallpack3()
    test_rall()
    test_basic_diffusion()
    test_allowed_integration()
    test_spatialneuron_indexing()
    test_tree_index_consistency()
    test_spatialneuron_subtree_assignment()
    test_spatialneuron_morphology_assignment()
    test_spatialneuron_capacitive_currents()
