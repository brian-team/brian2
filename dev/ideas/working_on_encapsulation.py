from brian2 import *
import time
start = time.time()
set_device('cpp_standalone', directory='encapsulation')

spiketimes, = randint(2, size=100).nonzero()
spiketimes = spiketimes*defaultclock.dt
numspikes = len(spiketimes)
print numspikes

tau = 9*ms
G = NeuronGroup(1, 'dv/dt=-v/tau:1')
G2 = SpikeGeneratorGroup(1, zeros(numspikes, dtype=int), spiketimes)
#M = StateMonitor(G, 'v', record=True)
#G.v = 1
M = SpikeMonitor(G2)

#device.set_variables_to_write([(M, 't'), (M, 'v')])

run(10*ms)

print M.t/ms
print spiketimes/ms
print time.time()-start

#print G.v
#plot(M.t/ms, M.v[0])
#show()
