from brian2 import *
import time

set_device('cpp_standalone')

N = 40000
rate = 10*Hz
M = int(rate * N * defaultclock.dt)
if M <= 0:
    M = 1
G = NeuronGroup(M, 'v:1', threshold='True')
H = NeuronGroup(N, 'w:1')
S = Synapses(G, H, on_pre='w += 1.0')
S.connect(True, p=1.0)

start_time = time.time()
run(.1*second, report='text')
device.build(directory='debug_slow_standalone', compile=True,
             run=True, debug=False)
print time.time()-start_time

print device._last_run_time
print device._last_run_completed_fraction

print profiling_summary(show=10)
