'''
Tests for `SpikeGeneratorGroup`
'''
import numpy as np
from numpy.testing.utils import assert_equal

from brian2 import *

# We can only test C++ if weave is availabe
try:
    import scipy.weave
    codeobj_classes = [NumpyCodeObject, WeaveCodeObject]
except ImportError:
    # Can't test C++
    codeobj_classes = [NumpyCodeObject]

def test_spikegenerator_basic():
    '''
    Basic test for `SpikeGeneratorGroup`.
    '''
    for codeobj_class in codeobj_classes:
        G = NeuronGroup(10, 'v:1', codeobj_class=codeobj_class)
        mon = StateMonitor(G, 'v', record=True,
                           codeobj_class=codeobj_class)
        indices = np.array([3, 2, 1, 1, 4, 5])
        times =   np.array([6, 5, 4, 3, 3, 1]) * ms
        SG = SpikeGeneratorGroup(10, indices, times,
                                 codeobj_class=codeobj_class)
        S = Synapses(SG, G, pre='v+=1', connect='i==j',
                     codeobj_class=codeobj_class)
        net = Network(G, SG, mon, S)
        net.run(7*ms)
        # The following neurons should not receive any spikes
        for idx in [0, 6, 7, 8, 9]:
            assert all(mon[idx].v == 0)
        # The following neurons should receive a single spike
        for idx, time in zip([2, 3, 4, 5], [5, 6, 3, 1]*ms):
            assert all(mon[idx].v[mon.t<time] == 0)
            assert all(mon[idx].v[mon.t>=time] == 1)
        # This neuron receives two spikes
        assert all(mon[1].v[mon.t<3*ms] == 0)
        assert all(mon[1].v[(mon.t>=3*ms) & (mon.t<4*ms)] == 1)
        assert all(mon[1].v[(mon.t>=4*ms)] == 2)


if __name__ == '__main__':
    test_spikegenerator_basic()
