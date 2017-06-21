#!/usr/bin/env python
'''
Input-Frequency curve of a IF model.
Network: 1000 unconnected integrate-and-fire neurons (leaky IF)
with an input parameter v0.
The input is set differently for each neuron.
'''
from brian2 import *

prefs.codegen.target = 'cython'
prefs.codegen.loop_invariant_optimisations = True

prefs.codegen.cpp.libraries += ['gsl', 'gslcblas']
prefs.codegen.cpp.headers += ['gsl/gsl_odeiv2.h']
prefs.codegen.cpp.include_dirs += ['/home/charlee/softwarefolder/gsl-2.3/gsl/']

n = 10
duration = .1*second
HH = False

if HH:
    # Parameters
    area = 20000*umetre**2
    Cm = 1*ufarad*cm**-2 * area
    gl = 5e-5*siemens*cm**-2 * area
    El = -65*mV
    EK = -90*mV
    ENa = 50*mV
    g_na = 100*msiemens*cm**-2 * area
    g_kd = 30*msiemens*cm**-2 * area
    VT = -63*mV

    # The model
    eqs = Equations('''
    dv/dt = (gl*(El-v) - g_na*(m*m*m)*h*(v-ENa) - g_kd*(n*n*n*n)*(v-EK) + I)/Cm : volt
    dm/dt = 0.32*(mV**-1)*(13.*mV-v+VT)/
        (exp((13.*mV-v+VT)/(4.*mV))-1.)/ms*(1-m)-0.28*(mV**-1)*(v-VT-40.*mV)/
        (exp((v-VT-40.*mV)/(5.*mV))-1.)/ms*m : 1
    dn/dt = 0.032*(mV**-1)*(15.*mV-v+VT)/
        (exp((15.*mV-v+VT)/(5.*mV))-1.)/ms*(1.-n)-.5*exp((10.*mV-v+VT)/(40.*mV))/ms*n : 1
    dh/dt = 0.128*exp((17.*mV-v+VT)/(18.*mV))/ms*(1.-h)-4./(1+exp((40.*mV-v+VT)/(5.*mV)))/ms*h : 1
    I : amp
    ''')

    threshold = 'v > -40*mV'
    refractory = 'v > -40*mV'
    reset = None
else:
    tau = 10*ms
    eqs = '''
    dv/dt = (v0 - v) / tau : volt (unless refractory)
    v0 : volt
    '''

    threshold = 'v > 10*mV'
    refractory = 5*ms
    reset = 'v = 0*mV'


from brian2.units.stdunits import stdunits
class GSLStateUpdater(StateUpdateMethod):
    def __call__(self, equations, variables=None):
        # the approach is to write all variables so they can
        # be translated to GSL code (i.e. with indexing and pointers)
        diff_eqs = equations.get_substituted_expressions(variables)

        code_begin = []
        code_end = []
        count_statevariables = 0
        counter = {}

        for diff_name, expr in diff_eqs:
            counter[diff_name] = count_statevariables
            code_end += ['_gsl_{var}_f{count} = {expr}'.format(var=diff_name,
                                                               expr=expr,
                                                               count=counter[diff_name])]
            count_statevariables += 1

        print ('\n').join(code_begin+code_end)
        return ('\n').join(code_begin+code_end)

GSLgroup = NeuronGroup(n, eqs, threshold=threshold, reset=reset,
                    refractory=refractory, method=GSLStateUpdater())
GSLgroup.state_updater.codeobj_class = GSLCythonCodeObject

EEgroup = NeuronGroup(n, eqs, threshold=threshold, reset=reset,
                        refractory=refractory, method='rk4')

if HH:
    GSLgroup.v = El
    GSLgroup.I = [0.7*nA * i for i in range(n)]
    EEgroup.v = GSLgroup.v
    EEgroup.I = GSLgroup.I
else:
    GSLgroup.v = 0*mV
    GSLgroup.v0 = '20*mV * i / (n-1)'
    EEgroup.v = GSLgroup.v
    EEgroup.v0 = GSLgroup.v0

GSL_spikemon = SpikeMonitor(GSLgroup)
GSL_statemon = StateMonitor(GSLgroup, 'v', record=True)
EE_spikemon = SpikeMonitor(EEgroup)
EE_statemon = StateMonitor(EEgroup, 'v', record=True)

GSLnetwork = Network(GSLgroup, GSL_spikemon, GSL_statemon)
EEnetwork = Network(EEgroup, EE_spikemon, EE_statemon)

defaultclock.dt = .01*ms
GSLnetwork.run(duration)
EEnetwork.run(duration)

#print group.state_updater.abstract_code
print GSLgroup.state_updater.codeobj.code
print EEgroup.state_updater.codeobj.code

if HH:
    f, ax = subplots(1, 2, figsize=(10, 5))
    ax[0].plot(EEgroup.I/nA, EE_spikemon.count / duration, label='Brian rk4')
    ax[0].plot(GSLgroup.I/nA, GSL_spikemon.count / duration, '--', label='GSL rk4')
    ax[0].set_xlabel('I (nA)')
    ax[0].set_ylabel('Firing rate (sp/s)')
    ax[0].legend()
    ax[0].set_title('IF-plot')

    ax[1].plot(EE_statemon.t/ms, EE_statemon.v[8]/mV, label='Brian rk4')
    ax[1].plot(GSL_statemon.t/ms, GSL_statemon.v[8]/mV, '--', label='GSL rk4')
    ax[1].set_xlabel('Time (ms)')
    ax[1].set_ylabel('Membrane potential (mV')
    ax[1].legend()
    ax[1].set_title('Example trace')

    show()

else:
    f, ax = subplots(1, 2, figsize=(10, 5))
    ax[0].plot(EEgroup.v0/mV, EE_spikemon.count / duration, label='Brian rk4')
    ax[0].plot(GSLgroup.v0/mV, GSL_spikemon.count / duration, '--', label='GSL rk4')
    ax[0].set_xlabel('v0 (mV)')
    ax[0].set_ylabel('Firing rate (sp/s)')
    ax[0].legend()
    ax[0].set_title('v0-F plot')

    ax[1].plot(EE_statemon.t/ms, EE_statemon.v[8]/mV, label='Brian rk4')
    ax[1].plot(GSL_statemon.t/ms, GSL_statemon.v[8]/mV, '--', label='GSL rk4')
    ax[1].set_xlabel('Time (ms)')
    ax[1].set_ylabel('Membrane potential (mV')
    ax[1].legend()
    ax[1].set_title('Example trace')

    show()
