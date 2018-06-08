from brian2 import *
from brian2.devices.cpp_standalone.device import Parameter
set_device('cpp_standalone', directory='encapsulation')

tau = Parameter(9*ms, name='tau')
G = NeuronGroup(1, 'dv/dt=(2-v)/tau:1', threshold='v>1', reset='v=0')
G2 = NeuronGroup(1, 'dv/dt=(3-v)/tau:1', threshold='v>1', reset='v=0')
M = SpikeMonitor(G)
M2 = SpikeMonitor(G2)

run(10*ms)

print M.t/ms
