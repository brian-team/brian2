import numpy as np
from numpy.testing.utils import assert_equal

from brian2 import *
from brian2.equations.refractory import add_refractoriness
    

def test_add_refractoriness():
    eqs = Equations('''
    dv/dt = -x*v/second : volt (unless-refractory)
    dw/dt = -w/second : amp
    x : 1
    ''')
    # only make sure it does not throw an error
    eqs = add_refractoriness(eqs)
    # Check that the parameters were added
    assert 'not_refractory' in eqs
    assert 'lastspike' in eqs


def test_refractoriness_time():
    # Try a quantity, a string evaluating to a quantity an an explicit boolean
    # condition -- all should do the same thing
    for ref_time in [5*ms, '5*ms', '(t-lastspike) < 5*ms']:
        G = NeuronGroup(1, '''
        dv/dt = 100*Hz : 1 (unless-refractory)
        dw/dt = 100*Hz : 1
        ''', threshold='v>1', reset='v=0;w=0', refractory=ref_time)
        # It should take 10ms to reach the threshold, then v should stay at 0
        # for 5ms, while w continues to increase
        mon = StateMonitor(G, ['v', 'w'], record=True)
        net = Network(G, mon)
        net.run(20*ms)
        # No difference before the spike
        assert_equal(mon.v[mon.t < 10*ms], mon.w[mon.t < 10*ms])
        # v is not updated during refractoriness
        in_refractoriness = mon.v[(mon.t >= 10*ms) & (mon.t <15*ms)]
        assert_equal(in_refractoriness, np.zeros_like(in_refractoriness))
        # w should evolve as before
        assert_equal(mon.w[mon.t < 5*ms], mon.w[(mon.t >= 10*ms) & (mon.t <15*ms)])
        assert np.all(mon.w[(mon.t >= 10*ms) & (mon.t <15*ms)] > 0)
        # After refractoriness, v should increase again
        assert np.all(mon.v[(mon.t >= 15*ms) & (mon.t <20*ms)] > 0)


if __name__ == '__main__':
    test_add_refractoriness()
    test_refractoriness_time()