#!/usr/bin/env python
'''
Spike-timing dependent plasticity
Adapted from Song, Miller and Abbott (2000) and Song and Abbott (2001)

This example is modified from ``synapses_STDP.py`` and writes a standalone C++ project in the directory
``STDP_standalone``.
'''
import sys, time
from brian2 import *

standalone = int(sys.argv[-2])
n_threads  = int(sys.argv[-1])

if standalone == 1:
    set_device('cpp_standalone')

start   = time.time()
N       = 1000
taum    = 10 * ms
taupre  = 20 * ms
taupost = taupre
Ee      = 0 * mV
vt      = -54 * mV
vr      = -60 * mV
El      = -74 * mV
taue    = 5 * ms
F       =  30 * Hz
gmax    = .01
dApre   = .01
dApost  = -dApre * taupre / taupost * 1.05
dApost *= gmax
dApre  *= gmax

eqs_neurons = '''
dv/dt=(ge*(Ee-vr)+El-v)/taum : volt   # the synaptic current is linearized
dge/dt=-ge/taue : 1
'''

input   = PoissonGroup(N, rates=F)
neurons = NeuronGroup(100, eqs_neurons, threshold='v>vt', reset='v=vr')
S = Synapses(input, neurons,
             '''w:1
                dApre/dt=-Apre/taupre : 1 (event-driven)    
                dApost/dt=-Apost/taupost : 1 (event-driven)''',
             pre='''ge+=w
                    Apre+=dApre
                    w=clip(w+Apost,0,gmax)''',
             post='''Apost+=dApost
                     w=clip(w+Apre,0,gmax)''',
             connect=True,
             )
S.w          = 'rand()*gmax'
state_mon    = StateMonitor(S, 'w', record=[0])
spike_mon_1  = SpikeMonitor(input)
spike_mon_2  = SpikeMonitor(neurons)
start_time   = time.time()

net = Network(input, neurons, S, state_mon, spike_mon_1, spike_mon_2, name='stdp_net')

net.run(10 * second)

if standalone == 1:
    device.build(project_dir='data_stdp_%d' %n_threads, compile_project=True, run_project=True, debug=False, n_threads=n_threads)

