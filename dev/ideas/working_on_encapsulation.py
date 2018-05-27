from brian2 import *
from brian2.devices.cpp_standalone import cpp_standalone_device
set_device('cpp_standalone', directory='encapsulation')

tau = 10*ms
G = NeuronGroup(1, 'dv/dt=-v/tau:1')
M = StateMonitor(G, 'v', record=True)
G.v = 1

device.set_variables_to_write([(M, 't'), (M, 'v')])

run(10*ms)

print G.v
plot(M.t/ms, M.v[0])
show()
