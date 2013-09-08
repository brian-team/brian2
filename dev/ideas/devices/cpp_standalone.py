'''
This is the not stable development version of this script.
Thomas/Achilleas/Divya: Use cpp_standalone.py for the moment.
'''

from numpy import *
from brian2 import *
from brian2.devices.cpp_standalone import *

set_device('cpp_standalone')

##### Define the model
tau = 10*ms
eqs = '''
dV/dt = -V/tau : volt (unless-refractory)
'''
threshold = 'V>-50*mV'
reset = 'V=-60*mV'
refractory = 5*ms
N = 1000

##### Generate C++ code

# Use a NeuronGroup to fake the whole process
G = NeuronGroup(N, eqs, reset=reset, threshold=threshold, refractory=refractory, name='gp')
M = SpikeMonitor(G)
G2 = NeuronGroup(1, eqs, reset=reset, threshold=threshold, refractory=refractory, name='gp2')
# Run the network for 0 seconds to generate the code
net = Network(G,
              M,
              G2,
              )

net.run(0*second)
build(net)
