import functools

from nose import with_setup, SkipTest
from nose.plugins.attrib import attr
from numpy.testing.utils import assert_raises

from brian2 import *
from brian2.devices.device import reinit_devices
from brian2.core.preferences import PreferenceError

from brian2.codegen.runtime.GSLcython_rt import IntegrationError
from brian2.stateupdaters.base import UnsupportedEquationsException

max_difference = .1*mV


def skip_if_not_implemented(func):
    @functools.wraps(func)
    def wrapped():
        try:
            func()
        except NotImplementedError:
            raise SkipTest('GSL support for numpy has not been implemented yet')
    return wrapped


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
@skip_if_not_implemented
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
                                                             'of spikes than integration with ',
                                                             'exponential euler')


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
@skip_if_not_implemented
def test_GSL_different_clocks():
    vt = 10*mV
    eqs = 'dv/dt = -v/(10*ms) : volt'
    neurons = NeuronGroup(1, model=eqs, threshold='v>vt',
                          method='gsl', dt=.2*ms)
    # for this test just check if it compiles
    run(0*ms)


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
@skip_if_not_implemented
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


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
@skip_if_not_implemented
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
    # assert not all(trace_conventional.v[0]==trace_GSL.v[0]), \
    #         ('output of GSL stateupdater is exactly the same as Brians stateupdater (unlikely to be right)')


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
@skip_if_not_implemented
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
@skip_if_not_implemented
def test_GSL_stochastic():
    tau = 20*ms
    sigma = .015
    eqs = '''
    dx/dt = (1.1 - x) / tau + sigma * (2 / tau)**.5 * xi : 1
    '''
    neuron = NeuronGroup(1, eqs, method='gsl')
    net = Network(neuron)
    assert_raises(UnsupportedEquationsException,
                  net.run, 0*ms, namespace={'tau': tau,
                                            'sigma': sigma})

@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
@skip_if_not_implemented
def test_GSL_error_dimension_mismatch_unit():
    eqs = '''
    dv/dt = (v0 - v)/(10*ms) : volt
    v0 : volt
    '''
    options = {'absolute_error_per_variable' : {'v' : 1*nS}}
    neuron = NeuronGroup(1, eqs, threshold='v > 10*mV', reset='v = 0*mV',
                         method='gsl', method_options=options)
    net = Network(neuron)
    assert_raises(DimensionMismatchError, net.run, 0*ms, namespace={})


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
@skip_if_not_implemented
def test_GSL_error_dimension_mismatch_dimensionless1():
    eqs = '''
    dv/dt = (v0 - v)/(10*ms) : 1
    v0 : 1
    '''
    options = {'absolute_error_per_variable' : {'v' : 1*mV}}
    neuron = NeuronGroup(1, eqs, threshold='v > 10', reset='v = 0',
                         method='gsl', method_options=options)
    net = Network(neuron)
    assert_raises(DimensionMismatchError, net.run, 0*ms, namespace={})


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
@skip_if_not_implemented
def test_GSL_error_dimension_mismatch_dimensionless2():
    eqs = '''
    dv/dt = (v0 - v)/(10*ms) : volt
    v0 : volt
    '''
    options = {'absolute_error_per_variable': {'v': 1e-3}}
    neuron = NeuronGroup(1, eqs, threshold='v > 10*mV', reset='v = 0*mV',
                         method='gsl', method_options=options)
    net = Network(neuron)
    assert_raises(DimensionMismatchError, net.run, 0*ms, namespace={})


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
@skip_if_not_implemented
def test_GSL_error_nonexisting_variable():
    eqs = '''
    dv/dt = (v0 - v)/(10*ms) : volt
    v0 : volt
    '''
    options = {'absolute_error_per_variable' : {'dummy' : 1e-3*mV}}
    neuron = NeuronGroup(1, eqs, threshold='v > 10*mV', reset='v = 0*mV',
                         method='gsl', method_options=options)
    net = Network(neuron)
    assert_raises(KeyError, net.run, 0*ms, namespace={})


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
@skip_if_not_implemented
def test_GSL_error_incorrect_error_format():
    eqs = '''
    dv/dt = (v0 - v)/(10*ms) : volt
    v0 : volt
    '''
    options = {'absolute_error_per_variable': object()}
    neuron = NeuronGroup(1, eqs, threshold='v > 10*mV', reset='v = 0*mV',
                         method='gsl', method_options=options)
    net = Network(neuron)
    options2 = {'absolute_error': 'not a float'}
    neuron2 = NeuronGroup(1, eqs, threshold='v > 10*mV', reset='v = 0*mV',
                         method='gsl', method_options=options2)
    net2 = Network(neuron2)
    assert_raises(TypeError, net.run, 0*ms, namespace={})
    assert_raises(TypeError, net2.run, 0 * ms, namespace={})


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
@skip_if_not_implemented
def test_GSL_error_nonODE_variable():
    eqs = '''
    dv/dt = (v0 - v)/(10*ms) : volt
    v0 : volt
    '''
    options = {'absolute_error_per_variable': {'v0': 1e-3*mV}}
    neuron = NeuronGroup(1, eqs, threshold='v > 10*mV', reset='v = 0*mV',
                         method='gsl', method_options=options)
    net = Network(neuron)
    assert_raises(KeyError, net.run, 0*ms, namespace={})


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
@skip_if_not_implemented
def test_GSL_error_bounds():
    runtime = 50*ms
    error1 = 1e-2*volt
    error2 = 1e-4*volt
    error3 = 1e-6*volt  # default error
    eqs = '''
    dv/dt = (stimulus(t) + -v)/(.1*ms) : volt
    '''
    stimulus = TimedArray(rand(int(runtime/(10*ms)))*3*volt, dt=5*ms)
    neuron1 = NeuronGroup(1, model=eqs, reset='v=0*mV', threshold='v>10*volt',
                          method='gsl',
                          method_options={'absolute_error_per_variable': {'v': error1}}, dt=1*ms)
    neuron2 = NeuronGroup(1, model=eqs, reset='v=0*mV', threshold='v>10*volt',
                          method='gsl',
                          method_options={'absolute_error_per_variable': {'v': error2}}, dt=1*ms)
    neuron3 = NeuronGroup(1, model=eqs, reset='v=0*mV', threshold='v>10*volt',
                          method='gsl',
                          method_options={'absolute_error_per_variable': {}}, dt=1*ms)  # Uses default error
    neuron_control = NeuronGroup(1, model=eqs, method='linear', dt=1*ms)
    mon1 = StateMonitor(neuron1, 'v', record=True)
    mon2 = StateMonitor(neuron2, 'v', record=True)
    mon3 = StateMonitor(neuron3, 'v', record=True)
    mon_control = StateMonitor(neuron_control, 'v', record=True)
    run(runtime)
    err1 = abs(mon1.v[0] - mon_control.v[0])
    err2 = abs(mon2.v[0] - mon_control.v[0])
    err3 = abs(mon3.v[0] - mon_control.v[0])
    assert max(err1) < error1, ("Error bound exceeded, error bound: %e, obtained error: %e"%(error1, max(err1)))
    assert max(err2) < error2, ("Error bound exceeded")
    assert max(err3) < error3, ("Error bound exceeded")
    assert max(err1) > max(err2), ("The simulation with smaller error bound produced a bigger maximum error")
    assert max(err2) > max(err3), ("The simulation with smaller error bound produced a bigger maximum error")


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
@skip_if_not_implemented
def test_GSL_non_autonomous():
    eqs = '''dv/dt = sin(2*pi*freq*t)/ms : 1
             freq : Hz'''
    neuron = NeuronGroup(10, eqs, method='gsl')
    neuron.freq = 'i*10*Hz + 10*Hz'
    neuron2 = NeuronGroup(10, eqs, method='euler')
    neuron2.freq = 'i*10*Hz + 10*Hz'
    mon = StateMonitor(neuron, 'v', record=True)
    mon2 = StateMonitor(neuron2, 'v', record=True)
    run(20*ms)
    abs_err = np.abs(mon.v.T - mon2.v.T)
    max_allowed = 1000*np.finfo(prefs.core.default_float_dtype).eps
    assert np.max(abs_err) < max_allowed


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
@skip_if_not_implemented
def test_GSL_non_autonomous():
    eqs = '''dv/dt = sin(2*pi*freq*t)/ms : 1
             freq : Hz'''
    neuron = NeuronGroup(10, eqs, method='gsl')
    neuron.freq = 'i*10*Hz + 10*Hz'
    neuron2 = NeuronGroup(10, eqs, method='euler')
    neuron2.freq = 'i*10*Hz + 10*Hz'
    mon = StateMonitor(neuron, 'v', record=True)
    mon2 = StateMonitor(neuron2, 'v', record=True)
    run(20*ms)
    abs_err = np.abs(mon.v.T - mon2.v.T)
    max_allowed = 1000 * np.finfo(prefs.core.default_float_dtype).eps
    assert np.max(abs_err) < max_allowed

@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
@skip_if_not_implemented
def test_GSL_refractory():
    eqs = '''dv/dt = 99.99*Hz : 1 (unless refractory)'''
    neuron = NeuronGroup(1, eqs, method='gsl', threshold='v>1', reset='v=0', refractory=3*ms)
    neuron2 = NeuronGroup(1, eqs, method='euler', threshold='v>1', reset='v=0', refractory=3*ms)
    mon = SpikeMonitor(neuron, 'v')
    mon2 = SpikeMonitor(neuron2, 'v')
    run(20*ms)
    assert mon.count[0] == mon2.count[0]


@skip_if_not_implemented
def test_GSL_save_step_count():
    eqs = '''
    dv/dt = -v/(.1*ms) : volt
    '''
    neuron = NeuronGroup(1, model=eqs, method='gsl',
                         method_options={'save_step_count': True}, dt=1*ms)
    run(0*ms)
    mon = StateMonitor(neuron, '_step_count', record=True, when='end')
    run(10*ms)
    assert mon._step_count[0][0] > 0, "Monitor did not save GSL step count"


HH_namespace = {
    'Cm': 1*ufarad*cm**-2,
    'gl': 5e-5*siemens*cm**-2,
    'El': -65*mV,
    'EK': -90*mV,
    'ENa': 50*mV,
    'g_na': 100*msiemens*cm**-2,
    'g_kd': 30*msiemens*cm**-2,
    'VT': -63*mV
}

HH_eqs = Equations('''
dv/dt = (gl*(El-v) - g_na*(m*m*m)*h*(v-ENa) - g_kd*(n*n*n*n)*(v-EK) + I)/Cm : volt
dm/dt = 0.32*(mV**-1)*(13.*mV-v+VT)/
    (exp((13.*mV-v+VT)/(4.*mV))-1.)/ms*(1-m)-0.28*(mV**-1)*(v-VT-40.*mV)/
    (exp((v-VT-40.*mV)/(5.*mV))-1.)/ms*m : 1
dn/dt = 0.032*(mV**-1)*(15.*mV-v+VT)/
    (exp((15.*mV-v+VT)/(5.*mV))-1.)/ms*(1.-n)-.5*exp((10.*mV-v+VT)/(40.*mV))/ms*n : 1
dh/dt = 0.128*exp((17.*mV-v+VT)/(18.*mV))/ms*(1.-h)-4./(1+exp((40.*mV-v+VT)/(5.*mV)))/ms*h : 1
I : amp/metre**2
''')

@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
@skip_if_not_implemented
def test_GSL_fixed_timestep_big_dt_small_error():
    # should raise integration error
    neuron = NeuronGroup(1, model=HH_eqs, threshold='v > -40*mV',
                         refractory='v > -40*mV', method='gsl',
                         method_options={'adaptable_timestep': False,
                                         'absolute_error': 1e-12},
                         dt=.001*ms, namespace=HH_namespace)
    neuron.I = 0.7*nA/(20000*umetre**2)
    neuron.v = HH_namespace['El']
    net = Network(neuron)
    assert_raises((RuntimeError, IntegrationError), net.run, 10*ms)


@attr('codegen-independent')
@skip_if_not_implemented
def test_GSL_internal_variable():
    assert_raises(SyntaxError, Equations, 'd_p/dt = 300*Hz : 1')


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
@skip_if_not_implemented
def test_GSL_method_options_neurongroup():
    neuron1 = NeuronGroup(1, model='dp/dt = 300*Hz : 1', method='gsl',
                          method_options={'adaptable_timestep':True})
    neuron2 = NeuronGroup(1, model='dp/dt = 300*Hz : 1', method='gsl',
                          method_options={'adaptable_timestep':False})
    run(0*ms)
    assert 'if (gsl_odeiv2_driver_apply_fixed_step' not in str(neuron1.state_updater.codeobj.code), \
        'This neuron should not call gsl_odeiv2_driver_apply_fixed_step()'
    assert 'if (gsl_odeiv2_driver_apply_fixed_step' in str(neuron2.state_updater.codeobj.code), \
        'This neuron should call gsl_odeiv2_driver_apply_fixed_step()'


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
@skip_if_not_implemented
def test_GSL_method_options_spatialneuron():
    morpho = Soma(30*um)
    eqs = '''
    Im = g * v : amp/meter**2
    dg/dt = siemens/metre**2/second : siemens/metre**2
    '''
    neuron1 = SpatialNeuron(morphology=morpho, model=eqs, Cm=1*uF/cm**2, Ri=100*ohm*cm,
                            method='gsl_rkf45', method_options={'adaptable_timestep': True})
    neuron2 = SpatialNeuron(morphology=morpho, model=eqs, Cm=1*uF/cm**2, Ri=100*ohm*cm,
                            method='gsl_rkf45', method_options={'adaptable_timestep': False})
    run(0*ms)
    assert 'if (gsl_odeiv2_driver_apply_fixed_step' not in str(neuron1.state_updater.codeobj.code), \
        'This neuron should not call gsl_odeiv2_driver_apply_fixed_step()'
    assert 'if (gsl_odeiv2_driver_apply_fixed_step' in str(neuron2.state_updater.codeobj.code), \
        'This neuron should call gsl_odeiv2_driver_apply_fixed_step()'


@skip_if_not_implemented
def test_GSL_method_options_synapses():
    N = 1000
    taum = 10*ms
    taupre = 20*ms
    taupost = taupre
    Ee = 0*mV
    vt = -54*mV
    vr = -60*mV
    El = -74*mV
    taue = 5*ms
    F = 15*Hz
    gmax = .01
    dApre = .01
    dApost = -dApre * taupre / taupost * 1.05
    dApost *= gmax
    dApre *= gmax
    eqs_neurons = '''
    dv/dt = (ge * (Ee-vr) + El - v) / taum : volt
    dge/dt = -ge / taue : 1
    '''
    input = PoissonGroup(N, rates=F)
    neurons = NeuronGroup(1, eqs_neurons, threshold='v>vt', reset='v = vr',
                          method='gsl_rkf45')
    S1 = Synapses(input, neurons,
                  '''w : 1
                     dApre/dt = -Apre / taupre : 1 (clock-driven)
                     dApost/dt = -Apost / taupost : 1 (clock-driven)''',
                  method='gsl_rkf45',
                  method_options={'adaptable_timestep':True})
    S2 = Synapses(input, neurons,
                  '''w : 1
                     dApre/dt = -Apre / taupre : 1 (clock-driven)
                     dApost/dt = -Apost / taupost : 1 (clock-driven)''',
                  method='gsl_rkf45',
                  method_options={'adaptable_timestep':False})
    run(0*ms)
    assert 'if (gsl_odeiv2_driver_apply_fixed_step' not in str(S1.state_updater.codeobj.code), \
        'This state_updater should not call gsl_odeiv2_driver_apply_fixed_step()'
    assert 'if (gsl_odeiv2_driver_apply_fixed_step' in str(S2.state_updater.codeobj.code), \
        'This state_updater should call gsl_odeiv2_driver_apply_fixed_step()'


if __name__ == '__main__':
    for t in [test_GSL_stateupdater_basic,
              test_GSL_different_clocks,
              test_GSL_default_function,
              test_GSL_user_defined_function,
              test_GSL_x_variable,
              test_GSL_failing_directory,
              test_GSL_stochastic,
              test_GSL_error_dimension_mismatch_unit,
              test_GSL_error_dimension_mismatch_dimensionless1,
              test_GSL_error_dimension_mismatch_dimensionless2,
              test_GSL_error_nonexisting_variable,
              test_GSL_error_incorrect_error_format,
              test_GSL_error_nonODE_variable,
              test_GSL_error_bounds,
              test_GSL_non_autonomous,
              test_GSL_refractory,
              test_GSL_save_step_count,
              test_GSL_fixed_timestep_big_dt_small_error,
              test_GSL_method_options_neurongroup,
              test_GSL_method_options_spatialneuron,
              test_GSL_method_options_synapses
              ]:
        try:
            t()
        except SkipTest as ex:
            print('Skipped: {} ({})'.format(t.__name__, str(ex)))
