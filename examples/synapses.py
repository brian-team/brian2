import numpy as np
import matplotlib.pyplot as plt

from brian2 import *

language = CPPLanguage()

G1 = NeuronGroup(10, 'dv/dt = -v / (10*ms) : 1',
                threshold='v > 1',
                reset='v=0.', language=language)
G1.v = 1.2
G2 = NeuronGroup(10, 'dv/dt = -v / (10*ms) : 1',
                 threshold='v > 1',
                 reset='v=0', language=language)
 
syn = Synapses(G1, G2, 'dw/dt = -w / (50*ms): 1', pre='v+=w',
               language=language)

syn.connect('(i == j) and (rand() < 0.5)')

# Set the delays
syn.delay[:] = '1*ms + i * ms + 0.25*randn()*ms'
print 'delays:', syn.delay[:]
# Set the initial values of the synaptic variable
syn.w[:] = 1

mon = StateMonitor(G2, 'v', record=True)
run(20*ms)
plt.plot(mon.t / ms, mon.v)
plt.show()