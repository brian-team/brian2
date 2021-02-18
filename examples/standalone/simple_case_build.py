#!/usr/bin/env python
'''
The most simple case how to use standalone mode with several `run` calls.
'''
from brian2 import *
set_device('cpp_standalone', build_on_run=False)

tau = 10*ms
I = 1  # input current
eqs = '''
dv/dt = (I-v)/tau : 1
'''
G = NeuronGroup(10, eqs, method='exact')
G.v = 'rand()'
mon = StateMonitor(G, 'v', record=True)
run(20*ms)
I = 0
run(80*ms)
# Actually generate/compile/run the code:
device.build()

plt.plot(mon.t/ms, mon.v.T)
plt.gca().set(xlabel='t (ms)', ylabel='v')
plt.show()
