"""
Example: Using `network_operation` to stop a simulation.
This script demonstrates how to use `network_operation` to stop a simulation
when a group of neurons reaches a target number of spikes.
"""

from brian2 import *

N = 10
tau = 10*ms
I = 1.5
target_spikes = 20

G = NeuronGroup(N, 'dv/dt = (I - v)/tau : 1', threshold='v>1', reset='v=0', method='exact')
G.v = 'rand()'

spike_mon = SpikeMonitor(G)

@network_operation(dt=1*ms)
def check_spikes():
    if spike_mon.num_spikes >= target_spikes:
        print(f"Stopped at t={defaultclock.t} with {spike_mon.num_spikes} spikes")
        stop()

run(100*ms)

plot(spike_mon.t/ms, spike_mon.i, '.k')
xlabel('Time (ms)')
ylabel('Neuron index')
title('Spike raster plot')
show()
