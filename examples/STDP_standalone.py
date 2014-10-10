#!/usr/bin/env python
'''
Spike-timing dependent plasticity
Adapted from Song, Miller and Abbott (2000) and Song and Abbott (2001)

This example is modified from ``synapses_STDP.py`` and writes a standalone C++ project in the directory
``STDP_standalone``.
'''
from time import time

from brian2 import *

set_device('cpp_standalone')

N = 1000
taum = 10 * ms
taupre = 20 * ms
taupost = taupre
Ee = 0 * mV
vt = -54 * mV
vr = -60 * mV
El = -74 * mV
taue = 5 * ms
F = 15 * Hz
gmax = .01
dApre = .01
dApost = -dApre * taupre / taupost * 1.05
dApost *= gmax
dApre *= gmax

eqs_neurons = '''
dv/dt=(ge*(Ee-vr)+El-v)/taum : volt   # the synaptic current is linearized
dge/dt=-ge/taue : 1
'''

input = PoissonGroup(N, rates=F)
neurons = NeuronGroup(1, eqs_neurons, threshold='v>vt', reset='v=vr')
myconst = np.inf
S = Synapses(input, neurons,
             '''w:1
                dApre/dt=-Apre/taupre : 1 (event-driven)
                dApost/dt=-Apost/taupost : 1 (event-driven)''',
             pre='''ge+=w
                    Apre+=dApre
                    w=clip(w+Apost,0,gmax) + 1.0/myconst''',
             post='''Apost+=dApost
                     w=clip(w+Apre,0,gmax)''',
             connect=True,
             )
S.w = 'rand()*gmax'
mon = StateMonitor(S, 'w', record=[0, 1])
s_mon = SpikeMonitor(input)
r_mon = PopulationRateMonitor(input)
start_time = time()
run(100 * second)
device.build(project_dir='STDP_standalone', compile_project=True,
             run_project=True, debug=True)

plt.subplot(311)
plt.plot(S.w / gmax, '.')
plt.subplot(312)
plt.hist(S.w / gmax, 20)
plt.subplot(313)
plt.plot(mon.t/second, mon.w.T/gmax)
# plt.figure()
# plt.subplot(211)
# plt.plot(s_mon.t/second, s_mon.i, ',k')
# plt.subplot(212)
# plt.plot(r_mon.t/second, r_mon.rate/Hz)
plt.show()
