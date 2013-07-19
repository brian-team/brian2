#!/usr/bin/env python
'''
Neurons with gap junctions
'''
import matplotlib.pyplot as plt
import numpy as np

from brian2 import *

N = 10
v0 = 1.05
tau = 10*ms

eqs = '''
dv/dt=(v0-v+Igap)/tau : 1
Igap : 1 # gap junction current
'''

neurons = NeuronGroup(N, eqs, threshold='v>1', reset='v=0')
neurons.v = np.linspace(0, 1, N)
trace = StateMonitor(neurons, 'v', record=[0, 5])

S = Synapses(neurons, neurons, '''w:1 # gap junction conductance
                                Igap=w*(v_pre-v_post): 1 (lumped)''',
            )
S.connect(True)
S.w = .02

run(500*ms)

plt.plot(trace.t / ms, trace.v[:, 0])
plt.plot(trace.t / ms, trace.v[:, 1])
plt.show()
