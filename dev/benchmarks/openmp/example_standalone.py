#!/usr/bin/env python
# coding: latin-1
"""
CUBA example with delays.
"""

import sys, time, os
from brian2 import *

standalone = int(sys.argv[-2])
n_threads  = int(sys.argv[-1])
path       = 'data_example_%d' %n_threads

if standalone == 1:
    set_device('cpp_standalone')
    brian_prefs.codegen.cpp_standalone.openmp_threads = n_threads

start      = time.time()
n_cells    = 1000

numpy.random.seed(42)
connectivity = numpy.random.randn(n_cells, n_cells)

taum       = 20 * ms
taus       = 5 * ms
Vt         = -50 * mV
Vr         = -60 * mV
El         = -49 * mV

fac        = (60 * 0.27 / 10)  # excitatory synaptic weight (voltage)

gmax       = 20*fac
dApre      = .01
taupre     = 20 * ms
taupost    = taupre
dApost     = -dApre * taupre / taupost * 1.05
dApost    *=  0.1*gmax
dApre     *=  0.1*gmax


eqs  = Equations('''
dv/dt  = (g-(v-El))/taum : volt
g                        : volt
''')

P    = NeuronGroup(n_cells, model=eqs, threshold='v>Vt', reset='v=Vr', refractory=5 * ms)
P.v  = Vr + numpy.random.rand(len(P)) * (Vt - Vr)
P.g  = 0 * mV



S    = Synapses(P, P, 
                    model = '''dApre/dt=-Apre/taupre    : 1 (event-driven)    
                               dApost/dt=-Apost/taupost : 1 (event-driven)
                               w                        : 1
                               dg/dt = -g/taus          : volt
                               g_post = g               : volt (summed)''', 
                    pre = '''g     += w*mV
                             Apre  += dApre
                             w      = w + Apost''',
                    post = '''Apost += dApost
                              w      = w + Apre''')
S.connect()
S.w  = fac*connectivity.flatten()


spike_mon = SpikeMonitor(P)
state_mon = StateMonitor(S, 'w', record=range(10), when=Clock(dt=0.1*second))
v_mon     = StateMonitor(P, 'v', record=range(10))

if standalone == 1:
    device.insert_code('main', 'std::clock_t start = std::clock();')

run(5 * second, report='text')

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
    device.build(project_dir='data_example_%d' %n_threads, compile_project=True, run_project=True, debug=False)
