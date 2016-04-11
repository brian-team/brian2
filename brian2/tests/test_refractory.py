from nose import with_setup
from nose.plugins.attrib import attr
from numpy.testing.utils import assert_equal, assert_allclose, assert_raises

from brian2 import *
from brian2.equations.refractory import add_refractoriness
from brian2.devices.device import reinit_devices


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

@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_refractoriness_basic():
    G = NeuronGroup(1, '''
                       dv/dt = 100*Hz : 1 (unless refractory)
                       dw/dt = 100*Hz : 1
                       ''',
                    threshold='v>1', reset='v=0;w=0',
                    refractory=5*ms)
    # It should take 10ms to reach the threshold, then v should stay at 0
    # for 5ms, while w continues to increase
    mon = StateMonitor(G, ['v', 'w'], record=True, when='end')
    run(20*ms)
    # No difference before the spike
    assert_equal(mon[0].v[mon.t < 10*ms], mon[0].w[mon.t < 10*ms])
    # v is not updated during refractoriness
    in_refractoriness = mon[0].v[(mon.t >= 10*ms) & (mon.t <15*ms)]
    assert_equal(in_refractoriness, np.zeros_like(in_refractoriness))
    # w should evolve as before
    assert_equal(mon[0].w[mon.t < 5*ms], mon[0].w[(mon.t >= 10*ms) & (mon.t <15*ms)])
    assert np.all(mon[0].w[(mon.t >= 10*ms) & (mon.t <15*ms)] > 0)
    # After refractoriness, v should increase again
    assert np.all(mon[0].v[(mon.t >= 15*ms) & (mon.t <20*ms)] > 0)


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_refractoriness_variables():
    # Try a string evaluating to a quantity an an explicit boolean
    # condition -- all should do the same thing
    for ref_time in ['5*ms', '(t-lastspike) <= 5*ms',
                     'time_since_spike <= 5*ms', 'ref_subexpression',
                     '(t-lastspike) <= ref', 'ref', 'ref_no_unit*ms']:
        reinit_devices()
        G = NeuronGroup(1, '''
                        dv/dt = 100*Hz : 1 (unless refractory)
                        dw/dt = 100*Hz : 1
                        ref : second
                        ref_no_unit : 1
                        time_since_spike = t - lastspike : second
                        ref_subexpression = (t - lastspike) <= ref : boolean
                        ''',
                        threshold='v>1', reset='v=0;w=0',
                        refractory=ref_time)
        G.ref = 5*ms
        G.ref_no_unit = 5
        # It should take 10ms to reach the threshold, then v should stay at 0
        # for 5ms, while w continues to increase
        mon = StateMonitor(G, ['v', 'w'], record=True, when='end')
        run(20*ms)
        try:
            # No difference before the spike
            assert_equal(mon[0].v[mon.t < 10*ms], mon[0].w[mon.t < 10*ms])
            # v is not updated during refractoriness
            in_refractoriness = mon[0].v[(mon.t >= 10*ms) & (mon.t <15*ms)]
            assert_equal(in_refractoriness, np.zeros_like(in_refractoriness))
            # w should evolve as before
            assert_equal(mon[0].w[mon.t < 5*ms], mon[0].w[(mon.t >= 10*ms) & (mon.t <15*ms)])
            assert np.all(mon[0].w[(mon.t >= 10*ms) & (mon.t <15*ms)] > 0)
            # After refractoriness, v should increase again
            assert np.all(mon[0].v[(mon.t >= 15*ms) & (mon.t <20*ms)] > 0)
        except AssertionError as ex:
            raise AssertionError('Assertion failed when using %r as refractory argument:\n%s' % (ref_time, ex))

@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_refractoriness_threshold_basic():
    G = NeuronGroup(1, '''
    dv/dt = 200*Hz : 1
    ''', threshold='v > 1', reset='v=0', refractory=10*ms)
    # The neuron should spike after 5ms but then not spike for the next
    # 10ms. The state variable should continue to integrate so there should
    # be a spike after 15ms
    spike_mon = SpikeMonitor(G)
    run(16*ms)
    assert_allclose(spike_mon.t, [4.9, 15] * ms)


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_refractoriness_threshold():
    # Try a quantity, a string evaluating to a quantity an an explicit boolean
    # condition -- all should do the same thing
    for ref_time in [10*ms, '10*ms', '(t-lastspike) <= 10*ms',
                     '(t-lastspike) <= ref', 'ref', 'ref_no_unit*ms']:
        reinit_devices()
        G = NeuronGroup(1, '''
                        dv/dt = 200*Hz : 1
                        ref : second
                        ref_no_unit : 1
                        ''', threshold='v > 1',
                        reset='v=0', refractory=ref_time)
        G.ref = 10*ms
        G.ref_no_unit = 10
        # The neuron should spike after 5ms but then not spike for the next
        # 10ms. The state variable should continue to integrate so there should
        # be a spike after 15ms
        spike_mon = SpikeMonitor(G)
        run(16*ms)
        assert_allclose(spike_mon.t, [4.9, 15] * ms)


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


if __name__ == '__main__':
    test_add_refractoriness()
    test_refractoriness_variables()
    test_refractoriness_threshold()
    test_refractoriness_types()
    test_conditional_write_set()
    test_conditional_write_behaviour()
