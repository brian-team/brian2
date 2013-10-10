'''
This is the not stable development version of this script.
Thomas/Achilleas/Divya: Use cpp_standalone.py for the moment.
'''

standalone_mode = True

from numpy import *
from brian2 import *
import time

BrianLogger.log_level_debug()

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
dV/dt = (-40*mV-V)/tau : volt (unless-refractory)
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
G2 = NeuronGroup(1, eqs, reset=reset, threshold=threshold, refractory=refractory, name='gp2')

net = Network(G,
              M,
              G2,
              #S,
              )

if not standalone_mode:
    net.run(0*ms)
    start_sim = time.time()

net.run(100*ms)

if standalone_mode:
    build()
    print 'Build time:', time.time()-start
else:
    print 'Build time:', start_sim-start
    print 'Simulation time:', time.time()-start_sim
    print 'Num spikes:', sum(M.count)
