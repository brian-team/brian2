from brian2 import *
from brian2.devices import reinit_devices, reset_device

max_difference = .1*mV

targets = ['brian2', 'weave', 'cython', 'cpp_standalone']

setting_dict = {'brian2' : {'device' : 'runtime', # do one without GSL so we can check against this!
                        'stateupdater' : 'exponential_euler',
                        'target' : 'weave'},
                'weave' : {'device' : 'runtime',
                           'stateupdater' : 'GSL_stateupdater',
                           'target' : 'weave'},
                'cython' : {'device' : 'runtime',
                           'stateupdater' : 'GSL_stateupdater',
                           'target' : 'cython'},
                'cpp_standalone' : {'device' : 'cpp_standalone',
                           'stateupdater' : 'GSL_stateupdater',
                           'target' : 'weave'}}

# basic test with each target language
def test_GSL_stateupdater_basic():
    # just the adaptive_threshold example
    eqs = '''
    dv/dt = -v/(10*ms) : volt
    dvt/dt = (10*mV-vt)/(15*ms) : volt
    '''
    reset = '''
    v = 0*mV
    vt += 3*mV
    '''
    spike_container = []
    spiketimes = rand(50)*100*ms # 50 spikes in 100 ms == 500 Hz
    for target in targets:
        set_device(setting_dict[target]['device'], build_on_run=False)
        device.reinit()
        device.activate()
        prefs.codegen.target = setting_dict[target]['target']
        method = setting_dict[target]['stateupdater']
        if target == 'brian2':
            method = 'linear'
        neurons = NeuronGroup(1, model=eqs, reset=reset, threshold='v>vt', method=method)
        neurons.vt = 10*mV
        # 50 'different' neurons so no neuron spikes more than once per dt
        P = SpikeGeneratorGroup(50, range(50), spiketimes)
        C = Synapses(P, neurons, on_pre='v += 3*mV')
        C.connect()
        SM = SpikeMonitor(neurons, variables='v')
        net = Network(neurons, P, C, SM)
        net.run(100*ms)
        #device.build(directory=None, with_output=False)
        spike_container += [SM.num_spikes]
        print('.'),
        #reset_device()
    assert all([x == spike_container[0] for x in spike_container[1:]]), \
           'GSL_statupdater produced varying results for different target languages'

def test_GSL_different_clocks():
    vt = 10*mV
    eqs = 'dv/dt = -v/(10*ms) : volt'
    for target in targets:
        if target is 'brian2':
            continue
        set_device(setting_dict[target]['device'], build_on_run=False)
        device.reinit()
        device.activate()
        prefs.codegen.target = setting_dict[target]['target']
        method = setting_dict[target]['stateupdater']
        neurons = NeuronGroup(1, model=eqs, threshold='v>vt', method=method, dt=.2*ms)
        # for this test just check if it compiles
        run(0*ms)
        print('.'),

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
    trace_holder = []
    for target in targets:
        set_device(setting_dict[target]['device'], build_on_run=False)
        device.reinit()
        device.activate()
        prefs.codegen.target = setting_dict[target]['target']
        method = setting_dict[target]['stateupdater']
        neurons = NeuronGroup(n, model=eqs, threshold='v > 1', reset='v = 0', method=method)
        neurons.v = vrand
        neurons.a = '0.05 + 0.7*i/n'
        trace = StateMonitor(neurons, 'v', record=50)
        net = Network(neurons, trace)
        net.run(100*ms)
        #device.build(directory=None, with_output=False)
        trace_holder += [trace.v[0]]
        print('.'),
        #reset_device()
    assert all(array([max(trace_holder[0]-trace_holder[i]) for i in range(1,len(targets))]) < (max_difference/mV)),\
            'difference between output of different targets is larger than max_difference'
    # if the output is exactly the same there is probably something wrong
    assert sum([all(trace_holder[0]==trace_holder[i]) for i in range(1, len(targets))]) == 0,\
            'output of GSL stateupdater is exactly the same as Brians stateupdater (unlikely to be right)'
    # for the different target languages using GSL, though, the output should be the same
    assert sum([all(trace_holder[1]==trace_holder[i]) for i in range(2, len(targets))]) == len(targets)-2,\
            'output of GSL stateupdater varies for target languages'


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
    trace_holder = []
    for target in targets:
        set_device(setting_dict[target]['device'], build_on_run=False)
        device.reinit()
        device.activate()
        prefs.codegen.target = setting_dict[target]['target']
        method = setting_dict[target]['stateupdater']
        neurons = NeuronGroup(n, model=eqs, threshold='v > 1', reset='v = 0', method=method)
        neurons.v = vrand
        neurons.a = '0.05 + 0.7*i/n'
        trace = StateMonitor(neurons, 'v', record=50)
        net = Network(neurons, trace)
        net.run(100*ms)
        #device.build(directory=None, with_output=False)
        trace_holder += [trace.v[0]]
        print('.'),
        #reset_device()
    assert all(array([max(trace_holder[0]-trace_holder[i]) for i in range(1,len(targets))]) < (max_difference/mV)),\
            'difference between output of different targets is larger than max_difference'
    # if the output is exactly the same there is probably something wrong
    assert sum([all(trace_holder[0]==trace_holder[i]) for i in range(1, len(targets))]) == 0,\
            'output of GSL stateupdater is exactly the same as Brians stateupdater (unlikely to be right)'
    # for the different target languages using GSL, though, the output should be the same
    assert sum([all(trace_holder[1]==trace_holder[i]) for i in range(2, len(targets))]) == len(targets)-2,\
            'output of GSL stateupdater varies for target languages'

if __name__=='__main__':
    test_GSL_different_clocks()
    test_GSL_stateupdater_basic()
    test_GSL_default_function()
    test_GSL_user_defined_function()
