#!/usr/bin/env python
'''
Neurons with gap junctions
'''
from brian2 import *

#brian_prefs.codegen.target = 'weave'

N = 10
v0 = 1.05
tau = 10*ms

eqs = '''
dv/dt=(v0-v+Igap)/tau : 1
Igap : 1 # gap junction current
'''

neurons = NeuronGroup(N, eqs, threshold='v>1', reset='v=0')
neurons.v = linspace(0, 1, N)
trace = StateMonitor(neurons, 'v', record=[0, 5])

S = Synapses(neurons, neurons, '''w : 1 # gap junction conductance
                                  Igap_post = w*(v_pre-v_post): 1 (summed)''')
S.connect(True)
S.w = .02

run(500*ms)

plot(trace.t / ms, trace[0].v)
plot(trace.t / ms, trace[5].v)
show()
