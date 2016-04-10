import tempfile

from nose import with_setup, SkipTest
from nose.plugins.attrib import attr
from numpy.testing.utils import assert_allclose, assert_equal, assert_raises

from brian2 import *
from brian2.devices.device import reinit_devices, set_device, reset_device

@attr('cpp_standalone', 'standalone-only')
@with_setup(teardown=reinit_devices)
def test_cpp_standalone(with_output=False):
    set_device('cpp_standalone', build_on_run=False)
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
    S = Synapses(G, G, 'w : volt', on_pre='V += w')
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
    reset_device()

@attr('cpp_standalone', 'standalone-only')
@with_setup(teardown=reinit_devices)
def test_dt_changes_between_runs_standalone(with_output=False):
    set_device('cpp_standalone', build_on_run=False)
    defaultclock.dt = 0.1*ms
    G = NeuronGroup(1, 'v:1')
    mon = StateMonitor(G, 'v', record=True)
    run(.5*ms)
    defaultclock.dt = .5*ms
    run(.5*ms)
    defaultclock.dt = 0.1*ms
    run(.5*ms)
    tempdir = tempfile.mkdtemp()
    if with_output:
        print tempdir
    device.build(directory=tempdir, compile=True, run=True,
                 with_output=True)
    assert len(mon.t[:]) == 5 + 1 + 5
    assert_allclose(mon.t[:],
                    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1., 1.1, 1.2, 1.3, 1.4]*ms)
    reset_device()

@attr('cpp_standalone', 'standalone-only')
@with_setup(teardown=reinit_devices)
def test_multiple_connects(with_output=False):
    set_device('cpp_standalone', build_on_run=False)
    G = NeuronGroup(10, 'v:1')
    S = Synapses(G, G, 'w:1')
    S.connect(i=[0], j=[0])
    S.connect(i=[1], j=[1])
    tempdir = tempfile.mkdtemp()
    if with_output:
        print tempdir
    run(0*ms)
    device.build(directory=tempdir, compile=True, run=True,
                 with_output=True)
    assert len(S) == 2 and len(S.w[:]) == 2
    reset_device()

@attr('cpp_standalone', 'standalone-only')
@with_setup(teardown=reinit_devices)
def test_storing_loading(with_output=False):
    set_device('cpp_standalone', build_on_run=False)
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
                          b_syn : boolean''')
    S.connect(j='i')
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
    reset_device()

@attr('cpp_standalone', 'standalone-only', 'openmp')
@with_setup(teardown=reinit_devices)
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
        set_device(devicename, build_on_run=False, with_output=False)
        Synapses.__instances__().clear()
        if devicename=='cpp_standalone':
            reinit_devices()
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
                                      w      = w + Apre''')
        S.connect()
        
        S.w       = fac*connectivity.flatten()

        T         = Synapses(Q, P, model = "w : 1", on_pre="g += w*mV")
        T.connect(j='i')
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
    reset_device(previous_device)

@attr('cpp_standalone', 'standalone-only')
@with_setup(teardown=reinit_devices)
def test_timedarray(with_output=True):
    set_device('cpp_standalone', build_on_run=False)

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

    reset_device()

@attr('cpp_standalone', 'standalone-only')
@with_setup(teardown=reinit_devices)
def test_duplicate_names_across_nets(with_output=True):
    set_device('cpp_standalone', build_on_run=False)
    # In standalone mode, names have to be globally unique, not just unique
    # per network
    obj1 = BrianObject(name='name1')
    obj2 = BrianObject(name='name2')
    obj3 = BrianObject(name='name3')
    obj4 = BrianObject(name='name1')
    net1 = Network(obj1, obj2)
    net2 = Network(obj3, obj4)
    net1.run(0*ms)
    net2.run(0*ms)
    assert_raises(ValueError, lambda: device.build())

    reset_device()

@attr('cpp_standalone', 'standalone-only', 'openmp')
@with_setup(teardown=reinit_devices)
def test_openmp_scalar_writes(with_output=False):
    # Test that writing to a scalar variable only is done once in an OpenMP
    # setting (see github issue #551)
    set_device('cpp_standalone', build_on_run=False)
    prefs.devices.cpp_standalone.openmp_threads = 4
    G = NeuronGroup(10, 's : 1 (shared)')
    G.run_regularly('s += 1')
    run(defaultclock.dt)
    tempdir = tempfile.mkdtemp()
    if with_output:
        print tempdir
    device.build(directory=tempdir, run=True, compile=True,
                 with_output=with_output)
    assert_equal(G.s[:], 1.0)

    reset_device()

@attr('cpp_standalone', 'standalone-only')
@with_setup(teardown=reinit_devices)
def test_time_after_run(with_output=False):
    set_device('cpp_standalone', build_on_run=False)
    # Check that the clock and network time after a run is correct, even if we
    # have not actually run the code yet (via build)
    G = NeuronGroup(10, 'dv/dt = -v/(10*ms) : 1')
    net = Network(G)
    assert_allclose(defaultclock.dt, 0.1*ms)
    assert_allclose(defaultclock.t, 0.*ms)
    assert_allclose(G.t, 0.*ms)
    assert_allclose(net.t, 0.*ms)
    net.run(10*ms)
    assert_allclose(defaultclock.t, 10.*ms)
    assert_allclose(G.t, 10.*ms)
    assert_allclose(net.t, 10.*ms)
    net.run(10*ms)
    assert_allclose(defaultclock.t, 20.*ms)
    assert_allclose(G.t, 20.*ms)
    assert_allclose(net.t, 20.*ms)
    tempdir = tempfile.mkdtemp()
    if with_output:
        print tempdir
    device.build(directory=tempdir, run=True, compile=True,
                 with_output=with_output)
    # Everything should of course still be accessible
    assert_allclose(defaultclock.t, 20.*ms)
    assert_allclose(G.t, 20.*ms)
    assert_allclose(net.t, 20.*ms)

    reset_device()

@attr('cpp_standalone', 'standalone-only')
@with_setup(teardown=reinit_devices)
def test_array_cache(with_output=False):
    # Check that variables are only accessible from Python when they should be
    set_device('cpp_standalone', build_on_run=False)
    G = NeuronGroup(10, '''dv/dt = -v / (10*ms) : 1
                           w : 1
                           x : 1
                           y : 1
                           z : 1 (shared)''',
                    threshold='v>1')
    S = Synapses(G, G, 'weight: 1', on_pre='w += weight')
    S.connect(p=0.2)
    S.weight = 7
    # All neurongroup values should be known
    assert_allclose(G.v, 0)
    assert_allclose(G.w, 0)
    assert_allclose(G.x, 0)
    assert_allclose(G.y, 0)
    assert_allclose(G.z, 0)
    assert_allclose(G.i, np.arange(10))

    # But the synaptic variable is not -- we don't know the number of synapses
    assert_raises(NotImplementedError, lambda: S.weight[:])

    # Setting variables with explicit values should not change anything
    G.v = np.arange(10)+1
    G.w = 2
    G.y = 5
    G.z = 7
    assert_allclose(G.v, np.arange(10)+1)
    assert_allclose(G.w, 2)
    assert_allclose(G.y, 5)
    assert_allclose(G.z, 7)

    # But setting with code should invalidate them
    G.x = 'i*2'
    assert_raises(NotImplementedError, lambda: G.x[:])

    # Make sure that the array cache does not allow to use incorrectly sized
    # values to pass
    assert_raises(ValueError, lambda: setattr(G, 'w', [0, 2]))
    assert_raises(ValueError, lambda: G.w.__setitem__(slice(0, 4), [0, 2]))

    run(10*ms)
    # v is now no longer known without running the network
    assert_raises(NotImplementedError, lambda: G.v[:])
    # Neither is w, it is updated in the synapse
    assert_raises(NotImplementedError, lambda: G.w[:])
    # However, no code touches y or z
    assert_allclose(G.y, 5)
    assert_allclose(G.z, 7)
    # i is read-only anyway
    assert_allclose(G.i, np.arange(10))

    # After actually running the network, everything should be accessible
    tempdir = tempfile.mkdtemp()
    if with_output:
        print tempdir
    device.build(directory=tempdir, run=True, compile=True,
                 with_output=with_output)
    assert all(G.v > 0)
    assert all(G.w > 0)
    assert_allclose(G.x, np.arange(10)*2)
    assert_allclose(G.y, 5)
    assert_allclose(G.z, 7)
    assert_allclose(G.i, np.arange(10))
    assert_allclose(S.weight, 7)

    reset_device()


if __name__=='__main__':
    # Print the debug output when testing this file only but not when running
    # via nose test
    for t in [
             test_cpp_standalone,
             test_dt_changes_between_runs_standalone,
             test_multiple_connects,
             test_storing_loading,
             test_openmp_consistency,
             test_timedarray,
             test_duplicate_names_across_nets,
             test_openmp_scalar_writes,
             test_time_after_run,
             test_array_cache
             ]:
        t(with_output=True)
        reinit_devices()
