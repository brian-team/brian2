'''
Tests for `SpikeGeneratorGroup`
'''
import os
import tempfile

from nose import with_setup
from nose.plugins.attrib import attr
import numpy as np
from numpy.testing.utils import assert_equal

from brian2 import *
from brian2.devices.cpp_standalone import cpp_standalone_device


def restore_device():
    cpp_standalone_device.reinit()
    set_device('runtime')
    restore_initial_state()


# We can only test C++ if weave is availabe
try:
    import scipy.weave
    codeobj_classes = [NumpyCodeObject, WeaveCodeObject]
except ImportError:
    # Can't test C++
    codeobj_classes = [NumpyCodeObject]


def test_spikegenerator_connected():
    '''
    Test that `SpikeGeneratorGroup` connects properly.
    '''
    G = NeuronGroup(10, 'v:1')
    mon = StateMonitor(G, 'v', record=True)
    indices = np.array([3, 2, 1, 1, 4, 5])
    times =   np.array([6, 5, 4, 3, 3, 1]) * ms
    SG = SpikeGeneratorGroup(10, indices, times)
    S = Synapses(SG, G, pre='v+=1', connect='i==j')
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


def test_spikegenerator_basic():
    '''
    Basic test for `SpikeGeneratorGroup`.
    '''
    for codeobj_class in codeobj_classes:
        indices = np.array([3, 2, 1, 1, 2, 3, 3, 2, 1])
        times   = np.array([1, 4, 4, 3, 2, 4, 2, 3, 2]) * ms
        SG = SpikeGeneratorGroup(5, indices, times)
        s_mon = SpikeMonitor(SG)
        net = Network(SG, s_mon)
        net.run(5*ms)
        for idx in xrange(5):
            generator_spikes = sorted([(idx, time) for time in times[indices==idx]])
            recorded_spikes = sorted([(idx, time) for time in s_mon.t['i==%d' % idx]])
            assert generator_spikes == recorded_spikes

@attr('standalone')
@with_setup(teardown=restore_device)
def test_spikegenerator_standalone():
    '''
    Basic test for `SpikeGeneratorGroup` in standalone.
    '''
    set_device('cpp_standalone')
    indices = np.array([3, 2, 1, 1, 2, 3, 3, 2, 1])
    times   = np.array([1, 4, 4, 3, 2, 4, 2, 3, 2]) * ms
    SG = SpikeGeneratorGroup(5, indices, times)
    s_mon = SpikeMonitor(SG)
    net = Network(SG, s_mon)
    net.run(5*ms)
    tempdir = tempfile.mkdtemp()
    device.build(project_dir=tempdir, compile_project=True, run_project=True,
                 with_output=False)
    for idx in xrange(5):
        generator_spikes = sorted([(idx, time) for time in times[indices==idx]])
        recorded_spikes = sorted([(idx, time)
                                  for time in s_mon.t[s_mon.i==idx]])
        assert generator_spikes == recorded_spikes


if __name__ == '__main__':
    test_spikegenerator_connected()
    test_spikegenerator_basic()
    test_spikegenerator_standalone()
