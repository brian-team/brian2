standalone_mode = True
plot_results = False

from pylab import *
from numpy import *
from brian2 import *
import time

#BrianLogger.log_level_debug()

start = time.time()

if standalone_mode:
    from brian2.devices.cpp_standalone import *
    set_device('cpp_standalone')
else:
    brian_prefs['codegen.target'] = 'weave'
    #brian_prefs['codegen.target'] = 'numpy'

##### Define the model
tau = 1*ms
eqs = '''
dV/dt = (-40*mV-V)/tau : volt (unless refractory)
'''
threshold = 'V>-50*mV'
reset = 'V=-60*mV'
refractory = 5*ms
N = 1000

##### Generate C++ code

# Use a NeuronGroup to fake the whole process
G = NeuronGroup(N, eqs,
                reset=reset,
                threshold=threshold,
                refractory=refractory,
                name='gp')
G.V = '-i*mV'
M = SpikeMonitor(G)
S = Synapses(G, G, 'w : volt', pre='V += w')
S.connect('abs(i-j)<5 and i!=j')
S.w = 0.5*mV
S.delay = '0*ms'

net = Network(G,
              M,
              S,
              )

if not standalone_mode:
    net.run(0*ms)
    start_sim = time.time()

net.run(100*ms)

if standalone_mode:
    build(project_dir='output', compile_project=True, run_project=True)
    print 'Build time:', time.time()-start
    if plot_results:
        S = loadtxt('output/results/spikemonitor_codeobject.txt', delimiter=',',
                    dtype=[('i', int), ('t', float)])
        i = S['i']
        t = S['t']*second
        plot(t, i, '.k')
else:
    print 'Build time:', start_sim-start
    print 'Simulation time:', time.time()-start_sim
    print 'Num spikes:', sum(M.count)
    print 'Num synapses:', len(S)
    if plot_results:
        i, t = M.it
        plot(t, i, '.k')

if plot_results:
    show()
