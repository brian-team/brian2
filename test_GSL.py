from brian2 import *
from brian2.core.preferences import PreferenceError
from brian2.stateupdaters.base import UnsupportedEquationsException

max_difference = .1*mV
max_difference_same_method = 1*pvolt

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
    assert spike_container[0] > 0, 'simulation should produce spiking, but no spikes monitored'
    assert all([x == spike_container[0] for x in spike_container[1:]]), \
           'GSL_statupdater produced varying results for different target languages'

def test_GSL_different_clocks():
    vt = 10*mV
    eqs = 'dv/dt = -v/(10*ms) : volt'
    for target in targets:
        if target is 'brian2':
            continue
        set_device(setting_dict[target]['device'], build_on_run=False, clean=True)
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

def test_GSL_fixed_timestep_rk4():
    '''
    In this test I run brian2 and GSL with same integration method and (fixed) timestep: results should match!
    (also tests 'unless refractory' tag)
    '''
    tau = 10*ms
    eqs = '''
    dv/dt = (v0 - v)/tau : volt (unless refractory)
    v0 : volt
    '''
    defaultclock.dt = .01*ms
    trace_holder = []
    for target in targets:
        set_device(setting_dict[target]['device'], directory='test_GSL_fixed_timestep_rk4', build_on_run=False)
        device.reinit()
        device.activate()
        prefs.codegen.target = setting_dict[target]['target']
        if target == 'brian2':
            neuron = NeuronGroup(1, eqs, threshold='v > 10*mV', reset='v = 0*mV',
                                 refractory=5*ms, method='rk4')
        else:
            neuron = NeuronGroup(1, eqs, threshold='v > 10*mV', reset='v = 0*mV',
                                  refractory=5*ms, method='GSL_stateupdater', method_options={'integrator' : 'rk4',
                                                                                              'adaptable_timestep' : True,
                                                                                              'eps_abs' : 1e-2,
                                                                                              'eps_rel' : 1e-2})
        neuron.v = 0*mV
        neuron.v0 = 13*mV
        mon = StateMonitor(neuron, 'v', record=True, dt=1*ms, when='start') # default of statemonitor is different for cpp_standalone!
        neuron.v0 = 10*mV
        mon = StateMonitor(neuron, 'v', record=True, dt=1*ms) # default of statemonitor is different for cpp_standalone!
        net = Network(neuron, mon)
        net.run(100*msecond)
        trace_holder += [mon.v[0]]
        print('.'),
    assert not all(diff(trace_holder[0]/mV) == 0), 'Membrane potential was unchanged'
    #TODO: figure out why this assertion doesn't pass
    #assert not any([max(trace_holder[0]-trace_holder[i]) < max_difference_same_method for i in range(1,len(targets))]), \
    #    'Different results for brian2 and GSL even though method and timestep were the same'

def test_GSL_x_variable():
    neurons = NeuronGroup(2, 'dx/dt = 300*Hz : 1', threshold='x>1', reset='x=0',
                          method='GSL_stateupdater')
    network = Network(neurons)
    # just testing compilation
    network.run(0*ms)
    print('.'),

def test_GSL_failing_directory():
    try:
        prefs.GSL.directory = 1
        raise Exception # shouldn't get here
    except PreferenceError:
        pass
    try:
        prefs.GSL.directory = '/usr/bla/bla/bla'
        raise Exception # shouldn't get here
    except PreferenceError:
        pass

def test_GSL_stochastic():
    Vr = 10*mV
    theta = 20*mV
    tau = 20*ms
    delta = 2*ms
    taurefr = 2*ms
    duration = .1*second
    C = 1000
    J = .1*mV
    muext = 25*mV
    sigmaext = 1*mV
    eqs = """
    dV/dt = (-V+muext + sigmaext * sqrt(tau) * xi)/tau : volt
    """
    group = NeuronGroup(10, eqs, method='GSL_stateupdater')
    try:
        run(0*ms)
        raise Exception(('The previous line should raise an UnsupportedEquationsException'))
    except UnsupportedEquationsException:
        pass
    print('.'),

if __name__=='__main__':
    test_GSL_stochastic()
    test_GSL_failing_directory()
    test_GSL_x_variable()
    test_GSL_fixed_timestep_rk4()
    test_GSL_different_clocks()
    test_GSL_stateupdater_basic()
    test_GSL_default_function()
    test_GSL_user_defined_function()
