from brian2 import *

standalone_mode = True
plot_results = True
duration = 100*second

import matplotlib.pyplot as plt
from time import time
import shutil

if standalone_mode:
    from brian2.devices.cpp_standalone import *
    set_device('cpp_standalone')
else:
    brian_prefs['codegen.target'] = 'weave'
    #brian_prefs['codegen.target'] = 'numpy'

N = 1000
taum = 10 * ms
taupre = 20 * ms
taupost = taupre
Ee = 0 * mV
vt = -54 * mV
vr = -60 * mV
El = -74 * mV
taue = 5 * ms
F = 15 * Hz
gmax = .01
dApre = .01
dApost = -dApre * taupre / taupost * 1.05
dApost *= gmax
dApre *= gmax

eqs_neurons = '''
dv/dt=(ge*(Ee-vr)+El-v)/taum : volt   # the synaptic current is linearized
dge/dt=-ge/taue : 1
'''

input = PoissonGroup(N, rates=F)
neurons = NeuronGroup(1, eqs_neurons, threshold='v>vt', reset='v=vr')
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

S.w='rand()*gmax'


net = Network(input, neurons, S)

net.run(duration)

if standalone_mode:
    shutil.rmtree('output')
    build(project_dir='output', compile_project=True, run_project=True)
else:
    pass

if not standalone_mode and plot_results:
    plt.subplot(211)
    plt.plot(S.w[:] / gmax, '.')
    plt.subplot(212)
    plt.hist(S.w[:] / gmax, 20)
    plt.show()
