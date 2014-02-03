#!/usr/bin/env python
# coding: latin-1
"""
CUBA example with delays.
"""

import sys, time
from brian2 import *

standalone = int(sys.argv[-2])
n_threads  = int(sys.argv[-1])

if standalone == 1:
    set_device('cpp_standalone')

start      = time.time()
n_cells    = 10000
n_exc      = int(0.8*n_cells)
p_conn     = 0.1
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

Pe   = P[0:n_exc]
Pi   = P[n_exc:]

we   = (60 * 0.27 / 10)  # excitatory synaptic weight (voltage)
wi   = (-20 * 4.5 / 10)  # inhibitory synaptic weight

Se   = Synapses(Pe, P, model = 'w : 1', pre = 'ge += w*mV')
Se.connect('i != j', p=p_conn)
Se.w     = '%g' %(we)
Se.delay ='rand()*ms'

Si   = Synapses(Pi, P, model = 'w : 1', pre = 'gi += w*mV')
Si.connect('i != j', p=p_conn)
Si.w     = '%g' %(wi)
Si.delay ='rand()*ms'

spike_mon = SpikeMonitor(P)

net = Network(P, Se, Si, spike_mon, name='stdp_net')

net.run(1 * second)
if standalone == 1:
    device.build(project_dir='data_cuba_%d' %n_threads, compile_project=True, run_project=True, debug=False, n_threads=n_threads)
