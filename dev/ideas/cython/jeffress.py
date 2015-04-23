#!/usr/bin/env python
'''
Phase locking of IF neurons to a periodic input
'''
from pylab import *
from brian2 import *

import time

#brian_prefs.codegen.target = 'weave'
brian_prefs.codegen.target = 'cython'
#brian_prefs.codegen.target = 'numpy'
#BrianLogger.log_level_debug()

tau = 20 * ms
N = 10000
b = 1.2 # constant current mean, the modulation varies
freq = 10 * Hz

eqs = '''
dv/dt=(-v+a*sin(2*pi*freq*t)+b)/tau : 1
a : 1
'''
neurons = NeuronGroup(N, model=eqs,
                      threshold='v>1',
                      reset='v=0',
                      )
neurons.v = rand(N)
neurons.a = linspace(.05, 0.75, N)
S = SpikeMonitor(neurons)
#trace = StateMonitor(neurons, 'v', record=50)

run(1*ms)
start = time.time()
run(1000 * ms)
print time.time()-start
#subplot(211)
plot(S.t / ms, S.i, '.')
#subplot(212)
#plot(trace.t / ms, trace.v.T)
show()
