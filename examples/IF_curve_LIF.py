#!/usr/bin/env python
'''
Input-Frequency curve of a IF model
Network: 1000 unconnected integrate-and-fire neurons (leaky IF)
with an input parameter v0.
The input is set differently for each neuron.
'''
from brian2 import *

# brian_prefs.codegen.target = 'weave'

N = 1000
tau = 10 * ms
eqs = '''
dv/dt=(v0-v)/tau : volt (unless refractory)
v0 : volt
'''
group = NeuronGroup(N, model=eqs, threshold='v>10 * mV',
                    reset='v = 0 * mV', refractory=5*ms)
group.v = 0 * mV
group.v0 = linspace(0 * mV, 20 * mV, N)

monitor = SpikeMonitor(group)

duration = 1 * second
run(duration)
plot(group.v0 / mV, monitor.count / duration)
show()
