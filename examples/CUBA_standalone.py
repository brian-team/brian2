#!/usr/bin/env python
# coding: latin-1
"""
CUBA example with delays.

Connection (no delay): 3.5 s
DelayConnection: 5.7 s
Synapses (with precomputed offsets): 6.6 s # 6.9 s
Synapses with weave: 6.4 s
Synapses with zero delays: 5.2 s
"""
from brian2 import *

set_device('cpp_standalone')

import time

n_threads  = 1
n_cells    = 10000
p_conn     = 0.02
start_time = time.time()
taum       = 20 * ms
taue       = 5 * ms
taui       = 10 * ms
Vt         = -50 * mV
Vr         = -60 * mV
El         = -49 * mV

eqs  = Equations('''
dv/dt  = (ge+gi-(v-El))/taum : volt
dge/dt = -ge/taue : volt
dgi/dt = -gi/taui : volt
''')

P    = NeuronGroup(n_cells, model=eqs, threshold='v>Vt', reset='v=Vr', refractory=5 * ms)
P.v  = Vr + rand(len(P)) * (Vt - Vr)
P.ge = 0 * mV
P.gi = 0 * mV

Pe   = P[0:int(0.8*n_cells)]
Pi   = P[int(0.8*n_cells):]

we   = (60 * 0.27 / 10) * mV # excitatory synaptic weight (voltage)
wi   = (-20 * 4.5 / 10) * mV # inhibitory synaptic weight

Se   = Synapses(Pe, P, model = 'w : 1', pre = 'ge += we')
Se.connect('i != j', p=p_conn)
Si   = Synapses(Pi, P, model = 'w : 1', pre = 'gi += wi')
Si.connect('i != j', p=p_conn)
Se.delay='rand()*ms'
Si.delay='rand()*ms'



mon = SpikeMonitor(P)

print "Network construction time:", time.time() - start_time, "seconds"
print len(P), "neurons in the network"
print "Simulation running..."
run(1 * second)
build(project_dir='CUBA_%d' %n_threads, compile_project=True, run_project=True, debug=True, n_threads=n_threads)
