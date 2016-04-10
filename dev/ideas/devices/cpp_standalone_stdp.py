from brian2 import *

standalone_mode = True
plot_results = True
duration = 1*second

import matplotlib.pyplot as plt
from time import time
import shutil, os

if standalone_mode:
    set_device('cpp_standalone')
else:
    brian_prefs['codegen.target'] = 'weave'
    #brian_prefs['codegen.target'] = 'numpy'

start = time()

clock = Clock()

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

input = PoissonGroup(N, rates=F, clock=clock)
neurons = NeuronGroup(1, eqs_neurons, threshold='v>vt', reset='v=vr', clock=clock)
S = Synapses(input, neurons,
             '''w:1
                dApre/dt=-Apre/taupre : 1 (event-driven)
                dApost/dt=-Apost/taupost : 1 (event-driven)''',
             on_pre='''ge+=w
                    Apre+=dApre
                    w=clip(w+Apost,0,gmax)''',
             on_post='''Apost+=dApost
                     w=clip(w+Apre,0,gmax)''',
             clock=clock,
             )
S.connect()

S.w = 'rand()*gmax'
    
net = Network(input, neurons, S, name='stdp_net')

net.run(0*second)

device.insert_code('main', '''
    double duration = 1.0;
    if(argc>1)
    {
        duration = atof(argv[1]);
    }
    stdp_net.run(duration);
    ''')

if standalone_mode:
#    if os.path.exists('output'):
#        shutil.rmtree('output')
    device.build(project_dir='output', compile_project=True, run_project=True, debug=False,
                 run_args=[str(float(duration))])
    w = fromfile('output/results/_dynamic_array_synapses_w', dtype=float64)
else:
    print 'Simulation time:', time()-start
    w = S.w[:]

if plot_results:
    plt.subplot(211)
    plt.plot(w / gmax, '.')
    plt.subplot(212)
    plt.hist(w / gmax, 20)
    plt.show()
