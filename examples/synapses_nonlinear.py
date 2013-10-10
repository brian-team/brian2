#!/usr/bin/env python
"""
NMDA synapses
"""
from brian2 import *

# brian_prefs.codegen.target = 'weave'

a=1/(10*ms)
b=1/(10*ms)
c=1/(10*ms)

input=NeuronGroup(2, 'dv/dt=1/(10*ms):1', threshold='v>1', reset='v=0')
neurons = NeuronGroup(1, """dv/dt=(g-v)/(10*ms) : 1
                            g : 1""")
S=Synapses(input,neurons,
           '''# This variable could also be called g_syn to avoid confusion
              dg/dt=-a*g+b*x*(1-g) : 1
              g_post = g : 1 (summed)
              dx/dt=-c*x : 1
              w : 1 # synaptic weight
           ''', pre='x+=w') # NMDA synapses

S.connect(True)
S.w = [1., 10.]
input.v = [0., 0.5]

M = StateMonitor(S, 'g', record=True)
Mn = StateMonitor(neurons, 'g', record=0)

run(1000*ms)

import matplotlib.pyplot as plt
plt.subplot(2, 1, 1)
plt.plot(M.t / ms, M.g.T)
plt.subplot(2, 1, 2)
plt.plot(Mn.t / ms, Mn[0].g)
plt.show()