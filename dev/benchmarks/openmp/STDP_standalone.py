#!/usr/bin/env python
'''
Spike-timing dependent plasticity
Adapted from Song, Miller and Abbott (2000) and Song and Abbott (2001)

This example is modified from ``synapses_STDP.py`` and writes a standalone C++ project in the directory
``STDP_standalone``.
'''
import sys, time, os
from brian2 import *

standalone = int(sys.argv[-2])
n_threads  = int(sys.argv[-1])
path       = 'data_stdp_%d' %n_threads

if standalone == 1:
    set_device('cpp_standalone')
    brian_prefs.codegen.cpp_standalone.openmp_threads = n_threads

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
neurons = NeuronGroup(500, eqs_neurons, threshold='v>vt', reset='v=vr')
S = Synapses(input, neurons,
             '''w:1
                dApre/dt=-Apre/taupre : 1 (event-driven)    
                dApost/dt=-Apost/taupost : 1 (event-driven)''',
             on_pre='''ge+=w
                    Apre+=dApre
                    w=clip(w+Apost,0,gmax)''',
             on_post='''Apost+=dApost
                     w=clip(w+Apre,0,gmax)''')
S.connect()
S.w          = 'rand()*gmax'
state_mon    = StateMonitor(S, 'w', record=[0])
spike_mon_1  = SpikeMonitor(input)
spike_mon_2  = SpikeMonitor(neurons)
start_time   = time.time()

net = Network(input, neurons, S, state_mon, spike_mon_1, spike_mon_2, name='stdp_net')

if standalone == 1:
    device.insert_code('main', 'std::clock_t start = std::clock();')

net.run(5 * second, report='text')

if standalone == 1:
    device.insert_code('main', '''
        std::ofstream myfile ("speed.txt");
        if (myfile.is_open())
        {
            double value = (double) (std::clock() - start)/(%d * CLOCKS_PER_SEC); 
            myfile << value << std::endl;
            myfile.close();
        }
        ''' %(max(1, n_threads)))

try:
    os.removedirs(path)
except Exception:
    pass

if standalone == 1:
    device.build(project_dir=path, compile_project=True, run_project=True, debug=False)

