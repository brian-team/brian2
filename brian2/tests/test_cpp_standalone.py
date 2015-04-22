import tempfile
import os

from nose import with_setup
from nose.plugins.attrib import attr
import numpy
from numpy.testing.utils import assert_allclose, assert_equal

from brian2 import *
from brian2.devices.cpp_standalone import cpp_standalone_device
from brian2.devices.device import restore_device

@attr('cpp_standalone', 'standalone-only')
@with_setup(teardown=restore_device)
def test_cpp_standalone(with_output=False):
    previous_device = get_device()
    set_device('cpp_standalone')
    ##### Define the model
    tau = 1*ms
    eqs = '''
    dV/dt = (-40*mV-V)/tau : volt (unless refractory)
    '''
    threshold = 'V>-50*mV'
    reset = 'V=-60*mV'
    refractory = 5*ms
    N = 1000
    
    G = NeuronGroup(N, eqs,
                    reset=reset,
                    threshold=threshold,
                    refractory=refractory,
                    name='gp')
    G.V = '-i*mV'
    M = SpikeMonitor(G)
    S = Synapses(G, G, 'w : volt', pre='V += w')
    S.connect('abs(i-j)<5 and i!=j')
    S.w = 0.5*mV
    S.delay = '0*ms'

    net = Network(G, M, S)
    net.run(100*ms)
    tempdir = tempfile.mkdtemp()
    if with_output:
        print tempdir
    device.build(directory=tempdir, compile=True, run=True,
                 with_output=with_output)
    # we do an approximate equality here because depending on minor details of how it was compiled, the results
    # may be slightly different (if -ffast-math is on)
    assert len(M.i)>=17000 and len(M.i)<=18000
    assert len(M.t) == len(M.i)
    assert M.t[0] == 0.
    assert M.t[-1] == 100*ms - defaultclock.dt
    set_device(previous_device)

@attr('cpp_standalone', 'standalone-only')
@with_setup(teardown=restore_device)
def test_multiple_connects(with_output=False):
    previous_device = get_device()
    set_device('cpp_standalone')
    G = NeuronGroup(10, 'v:1')
    S = Synapses(G, G, 'w:1')
    S.connect([0], [0])
    S.connect([1], [1])
    tempdir = tempfile.mkdtemp()
    if with_output:
        print tempdir
    run(0*ms)
    device.build(directory=tempdir, compile=True, run=True,
                 with_output=True)
    assert len(S) == 2 and len(S.w[:]) == 2
    set_device(previous_device)

@attr('cpp_standalone', 'standalone-only')
@with_setup(teardown=restore_device)
def test_storing_loading(with_output=False):
    previous_device = get_device()
    set_device('cpp_standalone')
    G = NeuronGroup(10, '''v : volt
                           x : 1
                           n : integer
                           b : boolean''')
    v = np.arange(10)*volt
    x = np.arange(10, 20)
    n = np.arange(20, 30)
    b = np.array([True, False]).repeat(5)
    G.v = v
    G.x = x
    G.n = n
    G.b = b
    S = Synapses(G, G, '''v_syn : volt
                          x_syn : 1
                          n_syn : integer
                          b_syn : boolean''', connect='i==j')
    S.v_syn = v
    S.x_syn = x
    S.n_syn = n
    S.b_syn = b
    run(0*ms)
    tempdir = tempfile.mkdtemp()
    if with_output:
        print tempdir
    device.build(directory=tempdir, compile=True, run=True, with_output=True)
    assert_allclose(G.v[:], v)
    assert_allclose(S.v_syn[:], v)
    assert_allclose(G.x[:], x)
    assert_allclose(S.x_syn[:], x)
    assert_allclose(G.n[:], n)
    assert_allclose(S.n_syn[:], n)
    assert_allclose(G.b[:], b)
    assert_allclose(S.b_syn[:], b)
    set_device(previous_device)

@attr('cpp_standalone', 'standalone-only')
@with_setup(teardown=restore_device)
def test_openmp_consistency(with_output=False):
    previous_device = get_device()
    n_cells    = 100
    n_recorded = 10
    numpy.random.seed(42)
    taum       = 20 * ms
    taus       = 5 * ms
    Vt         = -50 * mV
    Vr         = -60 * mV
    El         = -49 * mV
    fac        = (60 * 0.27 / 10)
    gmax       = 20*fac
    dApre      = .01
    taupre     = 20 * ms
    taupost    = taupre
    dApost     = -dApre * taupre / taupost * 1.05
    dApost    *=  0.1*gmax
    dApre     *=  0.1*gmax

    connectivity = numpy.random.randn(n_cells, n_cells)
    sources      = numpy.random.random_integers(0, n_cells-1, 10*n_cells)
    # Only use one spike per time step (to rule out that a single source neuron
    # has more than one spike in a time step)
    times        = numpy.random.choice(numpy.arange(10*n_cells), 10*n_cells,
                                       replace=False)*ms
    v_init       = Vr + numpy.random.rand(n_cells) * (Vt - Vr)

    eqs  = Equations('''
    dv/dt = (g-(v-El))/taum : volt
    dg/dt = -g/taus         : volt
    ''')

    results = {}

    for (n_threads, devicename) in [(0, 'runtime'),
                                    (0, 'cpp_standalone'),
                                    (1, 'cpp_standalone'),
                                    (2, 'cpp_standalone'),
                                    (3, 'cpp_standalone'),
                                    (4, 'cpp_standalone')]:
        set_device(devicename)
        Synapses.__instances__().clear()
        if devicename=='cpp_standalone':
            device.reinit()
        prefs.devices.cpp_standalone.openmp_threads = n_threads
        P    = NeuronGroup(n_cells, model=eqs, threshold='v>Vt', reset='v=Vr', refractory=5 * ms)
        Q    = SpikeGeneratorGroup(n_cells, sources, times)
        P.v  = v_init
        P.g  = 0 * mV
        S    = Synapses(P, P, 
                            model = '''dApre/dt=-Apre/taupre    : 1 (event-driven)    
                                       dApost/dt=-Apost/taupost : 1 (event-driven)
                                       w                        : 1''', 
                            pre = '''g     += w*mV
                                     Apre  += dApre
                                     w      = w + Apost''',
                            post = '''Apost += dApost
                                      w      = w + Apre''',
                            connect=True)
        
        S.w       = fac*connectivity.flatten()

        T         = Synapses(Q, P, model = "w : 1", pre="g += w*mV", connect='i==j')
        T.w       = 10*fac

        spike_mon = SpikeMonitor(P)
        rate_mon  = PopulationRateMonitor(P)
        state_mon = StateMonitor(S, 'w', record=range(n_recorded), dt=0.1*second)
        v_mon     = StateMonitor(P, 'v', record=range(n_recorded))

        run(0.2 * second, report='text')

        if devicename=='cpp_standalone':
            tempdir = tempfile.mkdtemp()
            if with_output:
                print tempdir
            device.build(directory=tempdir, compile=True,
                         run=True, with_output=with_output)

        results[n_threads, devicename]      = {}
        results[n_threads, devicename]['w'] = state_mon.w
        results[n_threads, devicename]['v'] = v_mon.v
        results[n_threads, devicename]['s'] = spike_mon.num_spikes
        results[n_threads, devicename]['r'] = rate_mon.rate[:]

    for key1, key2 in [((0, 'runtime'), (0, 'cpp_standalone')),
                       ((1, 'cpp_standalone'), (0, 'cpp_standalone')),
                       ((2, 'cpp_standalone'), (0, 'cpp_standalone')),
                       ((3, 'cpp_standalone'), (0, 'cpp_standalone')),
                       ((4, 'cpp_standalone'), (0, 'cpp_standalone'))
                       ]:
        assert_allclose(results[key1]['w'], results[key2]['w'])
        assert_allclose(results[key1]['v'], results[key2]['v'])
        assert_allclose(results[key1]['r'], results[key2]['r'])
        assert_allclose(results[key1]['s'], results[key2]['s'])
    set_device(previous_device)

@attr('cpp_standalone', 'standalone-only')
@with_setup(teardown=restore_device)
def test_timedarray(with_output=True):
    previous_device = get_device()
    set_device('cpp_standalone')

    defaultclock.dt = 0.1*ms
    ta1d = TimedArray(np.arange(10)*volt, dt=1*ms)
    ta2d = TimedArray(np.arange(300).reshape(3, 100).T, dt=defaultclock.dt)
    G = NeuronGroup(4, '''x = ta1d(t) : volt
                          y = ta2d(t, i) : 1''')
    mon = StateMonitor(G, ['x', 'y'], record=True)
    run(11*ms)
    tempdir = tempfile.mkdtemp()
    if with_output:
        print tempdir
    device.build(directory=tempdir, compile=True,
                 run=True, with_output=with_output)

    for idx in xrange(4):
        # x variable should have neuron independent values
        assert_equal(mon[idx].x[:],
                     np.clip(np.arange(11).repeat(10), 0, 9)*volt)

    for idx in xrange(3):
        # y variable is neuron-specific
        assert_equal(mon[idx].y[:],
                     np.clip(np.arange(110), 0, 99) + idx*100)
    # the 2d array only has 3 columns, the last neuron should therefore contain
    # only NaN
    assert_equal(mon[3].y[:], np.nan)

    set_device(previous_device)


if __name__=='__main__':
    # Print the debug output when testing this file only but not when running
    # via nose test
    for t in [
             test_cpp_standalone,
             test_multiple_connects,
             test_storing_loading,
             test_openmp_consistency,
             test_timedarray
             ]:
        t(with_output=True)
        restore_device()
