from collections import Counter

from nose import SkipTest, with_setup
from nose.plugins.attrib import attr
from numpy.testing.utils import assert_equal, assert_raises

from brian2.core.functions import timestep
from brian2.utils.logger import catch_logs
from brian2 import *
from brian2.equations.refractory import add_refractoriness
from brian2.devices.device import reinit_devices
from brian2.tests.utils import assert_allclose

@attr('codegen-independent')
def test_add_refractoriness():
    eqs = Equations('''
    dv/dt = -x*v/second : volt (unless refractory)
    dw/dt = -w/second : amp
    x : 1
    ''')
    # only make sure it does not throw an error
    eqs = add_refractoriness(eqs)
    # Check that the parameters were added
    assert 'not_refractory' in eqs
    assert 'lastspike' in eqs


@attr('codegen-independent')
def test_missing_refractory_warning():
    # Forgotten refractory argument
    with catch_logs() as l:
        group = NeuronGroup(1, 'dv/dt = -v / (10*ms) : 1 (unless refractory)',
                            threshold='v > 1', reset='v = 0')
    assert len(l) == 1
    assert l[0][0] == 'WARNING' and l[0][1].endswith('no_refractory')


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_refractoriness_basic():
    G = NeuronGroup(1, '''
                       dv/dt = 99.999*Hz : 1 (unless refractory)
                       dw/dt = 99.999*Hz : 1
                       ''',
                    threshold='v>1', reset='v=0;w=0',
                    refractory=5*ms)
    # It should take 10ms to reach the threshold, then v should stay at 0
    # for 5ms, while w continues to increase
    mon = StateMonitor(G, ['v', 'w'], record=True, when='end')
    run(20*ms)
    # No difference before the spike
    assert_allclose(mon[0].v[:timestep(10*ms, defaultclock.dt)],
                    mon[0].w[:timestep(10*ms, defaultclock.dt)])
    # v is not updated during refractoriness
    in_refractoriness = mon[0].v[timestep(10*ms, defaultclock.dt):timestep(15*ms, defaultclock.dt)]
    assert_equal(in_refractoriness, np.zeros_like(in_refractoriness))
    # w should evolve as before
    assert_allclose(mon[0].w[:timestep(5*ms, defaultclock.dt)],
                    mon[0].w[timestep(10*ms, defaultclock.dt)+1:timestep(15*ms, defaultclock.dt)+1])
    assert np.all(mon[0].w[timestep(10*ms, defaultclock.dt)+1:timestep(15*ms, defaultclock.dt)+1] > 0)
    # After refractoriness, v should increase again
    assert np.all(mon[0].v[timestep(15*ms, defaultclock.dt):timestep(20*ms, defaultclock.dt)] > 0)


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_refractoriness_variables():
    # Try a string evaluating to a quantity an an explicit boolean
    # condition -- all should do the same thing
    for ref_time in ['5*ms', '(t-lastspike + 1e-3*dt) < 5*ms',
                     'time_since_spike + 1e-3*dt < 5*ms', 'ref_subexpression',
                     '(t-lastspike + 1e-3*dt) <= ref', 'ref', 'ref_no_unit*ms']:
        reinit_devices()
        G = NeuronGroup(1, '''
                        dv/dt = 99.999*Hz : 1 (unless refractory)
                        dw/dt = 99.999*Hz : 1
                        ref : second
                        ref_no_unit : 1
                        time_since_spike = (t - lastspike) +1e-3*dt : second
                        ref_subexpression = (t - lastspike + 1e-3*dt) < ref : boolean
                        ''',
                        threshold='v>1', reset='v=0;w=0',
                        refractory=ref_time,
                        dtype={'ref': defaultclock.variables['t'].dtype,
                               'ref_no_unit': defaultclock.variables['t'].dtype,
                               'lastspike': defaultclock.variables['t'].dtype,
                               'time_since_spike': defaultclock.variables['t'].dtype})
        G.ref = 5*ms
        G.ref_no_unit = 5
        # It should take 10ms to reach the threshold, then v should stay at 0
        # for 5ms, while w continues to increase
        mon = StateMonitor(G, ['v', 'w'], record=True, when='end')
        run(20*ms)
        try:
            # No difference before the spike
            assert_allclose(mon[0].v[:timestep(10*ms, defaultclock.dt)],
                            mon[0].w[:timestep(10*ms, defaultclock.dt)])
            # v is not updated during refractoriness
            in_refractoriness = mon[0].v[timestep(10*ms, defaultclock.dt):timestep(15*ms, defaultclock.dt)]
            assert_allclose(in_refractoriness, np.zeros_like(in_refractoriness))
            # w should evolve as before
            assert_allclose(mon[0].w[:timestep(5*ms, defaultclock.dt)],
                         mon[0].w[timestep(10*ms, defaultclock.dt)+1:timestep(15*ms, defaultclock.dt)+1])
            assert np.all(mon[0].w[timestep(10*ms, defaultclock.dt)+1:timestep(15*ms, defaultclock.dt)+1] > 0)
            # After refractoriness, v should increase again
            assert np.all(mon[0].v[timestep(15*ms, defaultclock.dt):timestep(20*ms, defaultclock.dt)] > 0)
        except AssertionError as ex:
            raise
            raise AssertionError('Assertion failed when using %r as refractory argument:\n%s' % (ref_time, ex))

@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_refractoriness_threshold_basic():
    G = NeuronGroup(1, '''
    dv/dt = 199.99*Hz : 1
    ''', threshold='v > 1', reset='v=0', refractory=10*ms)
    # The neuron should spike after 5ms but then not spike for the next
    # 10ms. The state variable should continue to integrate so there should
    # be a spike after 15ms
    spike_mon = SpikeMonitor(G)
    run(16*ms)
    assert_allclose(spike_mon.t, [5, 15] * ms)


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_refractoriness_repeated():
    # Create a group that spikes whenever it can
    group = NeuronGroup(1, '', threshold='True', refractory=10*defaultclock.dt)
    spike_mon = SpikeMonitor(group)
    run(10000*defaultclock.dt)
    assert spike_mon.t[0] == 0*ms
    assert_allclose(np.diff(spike_mon.t), 10*defaultclock.dt)


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_refractoriness_repeated_legacy():
    if prefs.core.default_float_dtype == np.float32:
        raise SkipTest('Not testing legacy refractory mechanism with single '
                       'precision floats.')
    # Switch on behaviour from versions <= 2.1.2
    prefs.legacy.refractory_timing = True
    # Create a group that spikes whenever it can
    group = NeuronGroup(1, '', threshold='True', refractory=10*defaultclock.dt)
    spike_mon = SpikeMonitor(group)
    run(10000*defaultclock.dt)
    assert spike_mon.t[0] == 0*ms

    # Empirical values from running with earlier Brian versions
    assert_allclose(np.diff(spike_mon.t)[:10],
                    [1.1, 1, 1.1, 1, 1.1, 1.1, 1.1, 1.1, 1, 1.1]*ms)
    steps = Counter(np.diff(np.int_(np.round(spike_mon.t / defaultclock.dt))))
    assert len(steps) == 2 and steps[10] == 899 and steps[11] == 91
    prefs.legacy.refractory_timing = False


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_refractoriness_threshold():
    # Try a quantity, a string evaluating to a quantity an an explicit boolean
    # condition -- all should do the same thing
    for ref_time in [10*ms, '10*ms', '(t-lastspike) <= 10*ms',
                     '(t-lastspike) <= ref', 'ref', 'ref_no_unit*ms']:
        reinit_devices()
        G = NeuronGroup(1, '''
                        dv/dt = 199.99*Hz : 1
                        ref : second
                        ref_no_unit : 1
                        ''', threshold='v > 1',
                        reset='v=0', refractory=ref_time,
                        dtype={'ref': defaultclock.variables['t'].dtype,
                               'ref_no_unit': defaultclock.variables['t'].dtype})
        G.ref = 10*ms
        G.ref_no_unit = 10
        # The neuron should spike after 5ms but then not spike for the next
        # 10ms. The state variable should continue to integrate so there should
        # be a spike after 15ms
        spike_mon = SpikeMonitor(G)
        run(16*ms)
        assert_allclose(spike_mon.t, [5, 15] * ms)


@attr('codegen-independent')
def test_refractoriness_types():
    # make sure that using a wrong type of refractoriness does not work
    group = NeuronGroup(1, '', refractory='3*Hz')
    assert_raises(TypeError, lambda: Network(group).run(0*ms))
    group = NeuronGroup(1, 'ref: Hz', refractory='ref')
    assert_raises(TypeError, lambda: Network(group).run(0*ms))
    group = NeuronGroup(1, '', refractory='3')
    assert_raises(TypeError, lambda: Network(group).run(0*ms))
    group = NeuronGroup(1, 'ref: 1', refractory='ref')
    assert_raises(TypeError, lambda: Network(group).run(0*ms))

@attr('codegen-independent')
def test_conditional_write_set():
    '''
    Test that the conditional_write attribute is set correctly
    '''
    G = NeuronGroup(1, '''dv/dt = 10*Hz : 1 (unless refractory)
                          dw/dt = 10*Hz : 1''', refractory=2*ms)
    assert G.variables['v'].conditional_write is G.variables['not_refractory']
    assert G.variables['w'].conditional_write is None

@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_conditional_write_behaviour():
    H = NeuronGroup(1, 'v:1', threshold='v>-1')

    tau = 1*ms
    eqs = '''
    dv/dt = (2-v)/tau : 1 (unless refractory)
    dx/dt = 0/tau : 1 (unless refractory)
    dy/dt = 0/tau : 1
    '''
    reset = '''
    v = 0
    x -= 0.05
    y -= 0.05
    '''
    G = NeuronGroup(1, eqs, threshold='v>1', reset=reset, refractory=1*ms)

    Sx = Synapses(H, G, on_pre='x += dt*100*Hz')
    Sx.connect(True)

    Sy = Synapses(H, G, on_pre='y += dt*100*Hz')
    Sy.connect(True)

    M = StateMonitor(G, variables=True, record=True)

    run(10*ms)

    assert G.x[0] < 0.2
    assert G.y[0] > 0.2
    assert G.v[0] < 1.1


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_conditional_write_automatic_and_manual():
    source = NeuronGroup(1, '', threshold='True')  # spiking all the time
    target = NeuronGroup(2, '''dv/dt = 0/ms : 1 (unless refractory)
                               dw/dt = 0/ms : 1''',
                         threshold='t == 0*ms',
                         refractory='False')  # only refractory for the first time step
    # Cell is spiking/refractory only in the first time step
    syn = Synapses(source, target, on_pre='''v += 1
                                             w += 1 * int(not_refractory_post)''')
    syn.connect()
    mon = StateMonitor(target, ['v', 'w'], record=True, when='end')
    run(2*defaultclock.dt)

    # Synapse should not have been effective in the first time step
    assert_allclose(mon.v[:, 0], 0)
    assert_allclose(mon.v[:, 1], 1)
    assert_allclose(mon.w[:, 0], 0)
    assert_allclose(mon.w[:, 1], 1)


if __name__ == '__main__':
    test_add_refractoriness()
    test_missing_refractory_warning()
    test_refractoriness_basic()
    test_refractoriness_variables()
    test_refractoriness_threshold()
    test_refractoriness_threshold_basic()
    test_refractoriness_repeated()
    test_refractoriness_repeated_legacy()
    test_refractoriness_types()
    test_conditional_write_set()
    test_conditional_write_behaviour()
    test_conditional_write_automatic_and_manual()
