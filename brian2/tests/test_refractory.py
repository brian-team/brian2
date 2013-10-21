import numpy as np
from numpy.testing.utils import assert_equal, assert_allclose, assert_raises

from brian2 import *
from brian2.equations.refractory import add_refractoriness
    

# We can only test C++ if weave is availabe
try:
    import scipy.weave
    codeobj_classes = [NumpyCodeObject, WeaveCodeObject]
except ImportError:
    # Can't test C++
    codeobj_classes = [NumpyCodeObject]


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


def test_refractoriness_variables():
    # Try a quantity, a string evaluating to a quantity an an explicit boolean
    # condition -- all should do the same thing
    for codeobj_class in codeobj_classes:
        for ref_time in [5*ms, '5*ms', '(t-lastspike) < 5*ms',
                         'time_since_spike < 5*ms', 'ref_subexpression',
                         '(t-lastspike) < ref', 'ref', 'ref_no_unit*ms']:
            G = NeuronGroup(1, '''
            dv/dt = 100*Hz : 1 (unless refractory)
            dw/dt = 100*Hz : 1
            ref : second
            ref_no_unit : 1
            time_since_spike = t - lastspike : second
            ref_subexpression = (t - lastspike) < ref : bool
            ''',
                            threshold='v>1', reset='v=0;w=0',
                            refractory=ref_time, codeobj_class=codeobj_class)
            G.ref = 5*ms
            G.ref_no_unit = 5
            # It should take 10ms to reach the threshold, then v should stay at 0
            # for 5ms, while w continues to increase
            mon = StateMonitor(G, ['v', 'w'], record=True)
            net = Network(G, mon)
            net.run(20*ms)
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


def test_refractoriness_threshold():
    # Try a quantity, a string evaluating to a quantity an an explicit boolean
    # condition -- all should do the same thing
    for codeobj_class in codeobj_classes:
        for ref_time in [10*ms, '10*ms', '(t-lastspike) <= 10*ms',
                         '(t-lastspike) <= ref', 'ref', 'ref_no_unit*ms']:
            G = NeuronGroup(1, '''
            dv/dt = 200*Hz : 1
            ref : second
            ref_no_unit : 1
            ''', threshold='v > 1',
                            reset='v=0', refractory=ref_time,
                            codeobj_class=codeobj_class)
            G.ref = 10*ms
            G.ref_no_unit = 10
            # The neuron should spike after 5ms but then not spike for the next
            # 10ms. The state variable should continue to integrate so there should
            # be a spike after 15ms
            spike_mon = SpikeMonitor(G)
            net = Network(G, spike_mon)
            net.run(16*ms)
            assert_allclose(spike_mon.t, [4.9, 15] * ms)


def test_refractoriness_types():
    # make sure that using a wrong type of refractoriness does not work
    assert_raises(TypeError, lambda: NeuronGroup(1, '', refractory='3*Hz'))
    assert_raises(TypeError, lambda: NeuronGroup(1, 'ref: Hz',
                                                 refractory='ref'))
    assert_raises(TypeError, lambda: NeuronGroup(1, '', refractory='3'))
    assert_raises(TypeError, lambda: NeuronGroup(1, 'ref: 1',
                                                 refractory='ref'))


if __name__ == '__main__':
    test_add_refractoriness()
    test_refractoriness_variables()
    test_refractoriness_threshold()
    test_refractoriness_types()
