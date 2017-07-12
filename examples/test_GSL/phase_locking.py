#!/usr/bin/env python
'''
Phase locking of IF neurons to a periodic input.

23/06/2017 Edit: adapted to integrate with GSL code (output is the same as with normal cython code)
'''
from brian2 import *

GSLstandalone = True
GSL = True

if GSLstandalone:
    from brian2.devices.cpp_standalone import GSLCPPStandaloneCodeObject
    set_device('cpp_standalone', directory='phase_locking_cpp')
else:
    prefs.codegen.target = 'weave'

tau = 20*ms
n = 100
#b = 1.2 # constant current mean, the modulation varies
freq = 10*Hz

eqs = '''
dv/dt = (-v + a * sin(2 * pi * freq * t) + b) / tau : 1
a : 1
b : 1
'''

if GSL:
    neurons = NeuronGroup(n, model=eqs, threshold='v > 1', reset='v = 0',
                          method='GSL_stateupdater')
    if GSLstandalone:
        neurons.state_updater.codeobj_class = GSLCPPStandaloneCodeObject
    elif prefs.codegen.target == 'weave':
        neurons.state_updater.codeobj_class = GSLWeaveCodeObject
    else:
        neurons.state_updater.codeobj_class = GSLCythonCodeObject
else:
    neurons = NeuronGroup(n, model=eqs, threshold='v > 1', reset='v = 0',
                          method='exponential_euler')

neurons.v = 'rand()'
neurons.a = '0.05 + 0.7*i/n'
neurons.b = 1.2
S = SpikeMonitor(neurons)
trace = StateMonitor(neurons, 'v', record=50)

run(1*second)
print neurons.state_updater.codeobj.code

subplot(211)
plot(S.t/ms, S.i, '.k')
xlabel('Time (ms)')
ylabel('Neuron index')
subplot(212)
plot(trace.t/ms, trace.v.T)
xlabel('Time (ms)')
ylabel('v')
show()
