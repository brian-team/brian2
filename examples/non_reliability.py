#!/usr/bin/env python
'''
Reliability of spike timing.
See e.g. Mainen & Sejnowski (1995) for experimental results in vitro.

Here: a constant current is injected in all trials.

R. Brette
'''
from brian2 import *

#brian_prefs.codegen.target = 'weave'

N = 25
tau = 20 * ms
sigma = .015
eqs_neurons = '''
dx/dt=(1.1-x)/tau+sigma*(2./tau)**.5*xi:1
'''
neurons = NeuronGroup(N, model=eqs_neurons, threshold='x>1', reset='x=0')
neurons.refractory = 5*ms
spikes = SpikeMonitor(neurons)

run(500 * ms)
plot(spikes.t / ms, spikes.i, '.')
show()