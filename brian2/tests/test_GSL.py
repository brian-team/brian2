from nose import with_setup
from nose.plugins.attrib import attr
from numpy.testing.utils import assert_allclose, assert_equal, assert_raises

from brian2 import *
from brian2.devices.device import reinit_devices
from brian2.core.preferences import PreferenceError
from brian2.stateupdaters.base import UnsupportedEquationsException

import re

max_difference = .1*mV

@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_GSL_stateupdater_basic():
    # just the adaptive_threshold example: run for exponential_euler and GSL and see
    # if results are comparable (same amount of spikes and spikes > 0)
    eqs = '''
    dv/dt = -v/(10*ms) : volt
    dvt/dt = (10*mV-vt)/(15*ms) : volt
    '''
    reset = '''
    v = 0*mV
    vt += 3*mV
    '''
    neurons_conventional = NeuronGroup(1, model=eqs, reset=reset,
                                       threshold='v>vt', method='exponential_euler')
    neurons_GSL = NeuronGroup(1, model=eqs, reset=reset,
                              threshold='v>vt', method='gsl')
    neurons_conventional.vt = 10*mV
    neurons_GSL.vt = 10*mV
    # 50 'different' neurons so no neuron spikes more than once per dt
    P = SpikeGeneratorGroup(1, [0]*50, array(range(50))/50.*100*ms)
    C_conventional = Synapses(P, neurons_conventional, on_pre='v += 3*mV')
    C_GSL = Synapses(P, neurons_GSL, on_pre='v += 3*mV')
    C_conventional.connect()
    C_GSL.connect()
    SM_conventional = SpikeMonitor(neurons_conventional, variables='v')
    SM_GSL = SpikeMonitor(neurons_GSL, variables='v')
    net = Network(neurons_conventional, neurons_GSL, P,
                  C_conventional, C_GSL, SM_conventional, SM_GSL)
    net.run(100*ms)
    assert SM_conventional.num_spikes > 0, 'simulation should produce spiking, but no spikes monitored'
    assert SM_conventional.num_spikes == SM_GSL.num_spikes, ('GSL_statupdater produced different number '
                                                    'of spikes thanintegration with ',
                                                    'exponential euler')

@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_GSL_different_clocks():
    vt = 10*mV
    eqs = 'dv/dt = -v/(10*ms) : volt'
    neurons = NeuronGroup(1, model=eqs, threshold='v>vt',
                          method='gsl', dt=.2*ms)
    # for this test just check if it compiles
    run(0*ms)

@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_GSL_default_function():
    # phase_locking example
    tau = 20*ms
    n = 100
    b = 1.2 # constant current mean, the modulation varies
    freq = 10*Hz
    eqs = '''
    dv/dt = (-v + a * sin(2 * pi * freq * t) + b) / tau : 1
    a : 1
    '''
    vrand = rand(n)
    neurons_conventional = NeuronGroup(n, model=eqs, threshold='v > 1',
                                       reset='v = 0', method='exponential_euler')
    neurons_GSL = NeuronGroup(n, model=eqs, threshold='v > 1',
                                       reset='v = 0', method='gsl')
    neurons_conventional.v = vrand
    neurons_GSL.v = vrand
    neurons_conventional.a = '0.05 + 0.7*i/n'
    neurons_GSL.a = '0.05 + 0.7*i/n'

    trace_conventional = StateMonitor(neurons_conventional, 'v', record=50)
    trace_GSL = StateMonitor(neurons_GSL, 'v', record=50)
    net = Network(neurons_conventional, neurons_GSL, trace_conventional, trace_GSL)
    net.run(10*ms)

    assert max(trace_conventional.v[0]-trace_GSL.v[0]) < max_difference/mV, \
            ('difference between conventional and GSL output is larger than max_difference')
    assert not all(trace_conventional.v[0]==trace_GSL.v[0]), \
            ('output of GSL stateupdater is exactly the same as Brians stateupdater (unlikely to be right)')

@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_GSL_user_defined_function():
    # phase_locking example with user_defined sin
    eqs = '''
    dv/dt = (-v + a * sin(2 * pi * freq * t) + b) / tau : 1
    a : 1
    '''
    @implementation('cpp', '''
    double user_sin(double phase)
    {
        return sin(phase);
    }''')
    @implementation('cython', '''
    cdef double user_sin(double phase):
        return sin(phase)''')
    @check_units(phase=1,result=1)
    def user_sin(phase):
        raise Exception
    tau = 20*ms
    n = 100
    b = 1.2 # constant current mean, the modulation varies
    freq = 10*Hz
    eqs = '''
    dv/dt = (-v + a * user_sin(2 * pi * freq * t) + b) / tau : 1
    a : 1
    '''
    vrand = rand(n)
    neurons_conventional = NeuronGroup(n, model=eqs, threshold='v > 1',
                                       reset='v = 0', method='exponential_euler')
    neurons_GSL = NeuronGroup(n, model=eqs, threshold='v > 1',
                                       reset='v = 0', method='gsl')
    neurons_conventional.v = vrand
    neurons_GSL.v = vrand
    neurons_conventional.a = '0.05 + 0.7*i/n'
    neurons_GSL.a = '0.05 + 0.7*i/n'

    trace_conventional = StateMonitor(neurons_conventional, 'v', record=50)
    trace_GSL = StateMonitor(neurons_GSL, 'v', record=50)
    net = Network(neurons_conventional, neurons_GSL, trace_conventional, trace_GSL)
    net.run(10*ms)

    assert max(trace_conventional.v[0]-trace_GSL.v[0]) < max_difference/mV, \
            ('difference between conventional and GSL output is larger than max_difference')
    assert not all(trace_conventional.v[0]==trace_GSL.v[0]), \
            ('output of GSL stateupdater is exactly the same as Brians stateupdater (unlikely to be right)')

@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_GSL_x_variable():
    neurons = NeuronGroup(2, 'dx/dt = 300*Hz : 1', threshold='x>1', reset='x=0',
                          method='gsl')
    # just testing compilation
    run(0*ms)

@attr('codegen-independent')
def test_GSL_failing_directory():
    def set_dir(arg):
        prefs.GSL.directory = arg
    assert_raises(PreferenceError, set_dir, 1)
    assert_raises(PreferenceError, set_dir, '/usr/')
    assert_raises(PreferenceError, set_dir, '/usr/blablabla/')

@attr('codegen-independent')
def test_GSL_stochastic():
    tau = 10*ms
    eqs = '''
    dv/dt = (v + xi)/tau : volt
    '''
    neuron = NeuronGroup(1, eqs, method='gsl')

if __name__ == '__main__':
    test_GSL_stateupdater_basic()
    test_GSL_different_clocks()
    test_GSL_default_function()
    test_GSL_user_defined_function()
    test_GSL_x_variable()
