'''
August 1st, 2017: adapted to run with GSL by Charlee Fletterman.
Tried with: weave, cython and cpp_standalone. Figure looks the same as conventional brian (by eye)
======================================
A simple example of using `Synapses`.
'''
from brian2 import *

G1 = NeuronGroup(10, 'dv/dt = -v / (10*ms) : 1',
                 threshold='v > 1', reset='v=0.', method='linear')
G1.v = 1.2
G2 = NeuronGroup(10, 'dv/dt = -v / (10*ms) : 1',
                 threshold='v > 1', reset='v=0', method='linear')
 
syn = Synapses(G1, G2, 'dw/dt = -w / (50*ms): 1 (clock-driven)', on_pre='v += w', method='GSL_stateupdater')

syn.connect('i == j', p=0.75)

# Set the delays
syn.delay = '1*ms + i*ms + 0.25*ms * randn()'
# Set the initial values of the synaptic variable
syn.w = 1

mon = StateMonitor(G2, 'v', record=True)
run(20*ms)
print getattr(syn.state_updater.codeobj, 'code', None)

plot(mon.t/ms, mon.v.T)
xlabel('Time (ms)')
ylabel('v')
show()
