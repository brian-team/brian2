import tempfile
import os

from nose import with_setup
from nose.plugins.attrib import attr
import numpy
from numpy.testing.utils import assert_allclose

from brian2 import *
from brian2.devices.cpp_standalone import cpp_standalone_device


def restore_device():
    cpp_standalone_device.reinit()
    set_device('runtime')
    restore_initial_state()


@attr('standalone')
@with_setup(teardown=restore_device)
def test_cpp_standalone(with_output=False):
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

@attr('standalone')
@with_setup(teardown=restore_device)
def test_multiple_connects(with_output=False):
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

@attr('standalone')
@with_setup(teardown=restore_device)
def test_storing_loading(with_output=False):
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
    S = Synapses(G, G, '''v : volt
                          x : 1
                          n : integer
                          b : boolean''', connect='i==j')
    S.v = v
    S.x = x
    S.n = n
    S.b = b
    run(0*ms)
    tempdir = tempfile.mkdtemp()
    if with_output:
        print tempdir
    device.build(directory=tempdir, compile=True, run=True, with_output=True)
    assert_allclose(G.v[:], v)
    assert_allclose(S.v[:], v)
    assert_allclose(G.x[:], x)
    assert_allclose(S.x[:], x)
    assert_allclose(G.n[:], n)
    assert_allclose(S.n[:], n)
    assert_allclose(G.b[:], b)
    assert_allclose(S.b[:], b)

@attr('standalone')
@with_setup(teardown=restore_device)
def test_openmp_consistency(with_output=False):

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
    times        = 1*second*numpy.sort(numpy.random.rand(10*n_cells))
    v_init       = Vr + numpy.random.rand(n_cells) * (Vt - Vr)

    eqs  = Equations('''
    dv/dt = (g-(v-El))/taum : volt
    dg/dt = -g/taus         : volt
    ''')

    results = {}

    for (n_threads, devicename) in [(0, 'runtime'),
                                    (0, 'cpp_standalone'),
                                    (1, 'cpp_standalone'),
                                    (2, 'cpp_standalone')]:
        set_device(devicename)
        Synapses.__instances__().clear()
        if devicename=='cpp_standalone':
            device.reinit()
        brian_prefs.codegen.cpp_standalone.openmp_threads = n_threads                
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
                       ]:
        assert_allclose(results[key1]['w'], results[key2]['w'])
        assert_allclose(results[key1]['v'], results[key2]['v'])
        assert_allclose(results[key1]['r'], results[key2]['r'])
        assert_allclose(results[key1]['s'], results[key2]['s'])


if __name__=='__main__':
    # Print the debug output when testing this file only but not when running
    # via nose test
    for t in [
             test_cpp_standalone,
             test_multiple_connects,
             test_storing_loading,
             test_openmp_consistency
             ]:
        t(with_output=True)
        restore_device()
