#!/usr/bin/env python

"""
This script implements a homeostatic STDP rule for inhibitory
synapses onto excitatory neurons presented in Zenke et al. (2015) and
explained in detail in the accompanying supplementary information.

In essence, the magnitude and sign of the STDP kernel at inhibitory
synapses targeting excitatory neurons is modulated by a global factor
G, which corresponds to the difference between the average firing rate
of the excitatory postsynaptic neurons and a target firing rate
:math:`\gamma`. If the average firing rate of excitatory neurons exceeds the
target firing rate, pre/post pairings (and post/pre pairings) at
inhibitory synapses connecting to excitatory neurons result in
potentiation; depression of the inhibitory synapses occurs when the
excitatory neurons are unable to match the desired target rate.

The script emulates classic STDP protocols, simulating a connected
presynaptic and postsynaptic neuron pair with precisely controlled
spike times, and a large, independently firing population of
excitatory neurons.

Paul Brodersen, 2025

Reference:
Zenke, Friedemann, Everton J. Agnes, and Wulfram Gerstner.
"Diverse Synaptic Plasticity Mechanisms Orchestrated to Form and
Retrieve Memories in Spiking Neural Networks." Nature Communications 6,
no. 1 (April 21, 2015): 6922. https://doi.org/10.1038/ncomms7922.
"""

import numpy as np
import matplotlib.pyplot as plt
import brian2 as b2

net = b2.Network()

# excitatory neuron population
epoch_length = 1 * b2.second
spike_rate = np.arange(0, 12.5, 2.5)
total_epochs = len(spike_rate)
population_spike_rate = b2.TimedArray(spike_rate * b2.Hz, dt=epoch_length)
population_size = 1000
spike_generators = b2.PoissonGroup(population_size, rates="population_spike_rate(t)")
net.add(spike_generators)

# global factor G
gamma = 5 * b2.Hz
tau_H = 0.1 * b2.second # NB: value in Zenke et al (2015) is 10 seconds; reduced here to allow for shorter simulations
global_factor_model = """
G = H - gamma    : Hz
dH/dt = -H/tau_H : Hz
"""
global_factor = b2.NeuronGroup(1, model=global_factor_model)
net.add(global_factor)
spike_aggregator = b2.Synapses(spike_generators, global_factor, on_pre="H_post += 1/(tau_H * population_size)")
spike_aggregator.connect()
net.add(spike_aggregator)

# pre-synaptic neuron
spike_times = epoch_length * (np.arange(total_epochs) + 0.5)
spike_indices = np.zeros_like(spike_times / b2.second)
presynaptic_neuron = b2.SpikeGeneratorGroup(1, spike_indices, spike_times)
net.add(presynaptic_neuron)

# postsynaptic neuron
delay = 1 * b2.msecond
postsynaptic_neuron = b2.SpikeGeneratorGroup(1, spike_indices, spike_times + delay)
net.add(postsynaptic_neuron)

# synapse
tau_stdp = 20. * b2.msecond
eta = 1. * b2.nsiemens * b2.second
synapse_model = """
wij : siemens
dzi/dt = -zi / tau_stdp : 1 (event-driven)
dzj/dt = -zj / tau_stdp : 1 (event-driven)
"""
on_pre = """
zi += 1.
wij += eta * G * zj
"""
on_post = """
zj += 1.
wij += eta * G * (zi + 1)
"""
synapse = b2.Synapses(
    presynaptic_neuron, postsynaptic_neuron,
    model=synapse_model, on_pre=on_pre, on_post=on_post
)
synapse.connect(i=0, j=0)
# Synapses currently don't support linked variables.
# The following syntax hence results in an error:
# synapses.G = b2.linked_var(global_factor, "G")
# Instead we add the reference G as shown below.
# See also:
# https://brian.discourse.group/t/valueerror-equations-of-type-parameter-cannot-have-a-flag-linked-only-the-following-flags-are-allowed-constant-shared/1373/2
synapse.variables.add_reference("G", group=global_factor, index='0')
synapse.wij = 0 * b2.nsiemens
net.add(synapse)

# monitor
monitor = b2.StateMonitor(synapse, ["G", "wij"], record=True, dt=1*b2.msecond)
net.add(monitor)

# run
net.run(total_epochs * epoch_length)

# analysis
G = np.squeeze(monitor.G / b2.Hz)
wij = np.squeeze(monitor.wij / b2.nsiemens)
time = np.squeeze(monitor.t / b2.ms)
spike_times = spike_times / b2.ms

# visualize experimental protocol
fig, axes = plt.subplots(2, 1, sharex=True)
fig.suptitle("Experimental protocol")
axes[0].plot(time, G)
axes[0].set_ylabel("G [Hz]")
axes[1].plot(time, wij)
axes[1].set_ylabel("Weight [nS]")
for ax in axes:
    ax.vlines(spike_times, *ax.get_ylim(), color="gray", linestyle=":", label="Pre/post spike pairings")
axes[0].legend(loc="upper left")
axes[-1].set_xlabel("Time [ms]")

# plot relationship between G and \Delta wij
g = np.zeros_like(spike_times)
dw = np.zeros_like(spike_times)
for ii, spike_time in enumerate(spike_times):
    g[ii] = G[np.argmin(np.abs(time - spike_time))]
    w0 = wij[np.argmin(np.abs(time - spike_time - 10))]
    w1 = wij[np.argmin(np.abs(time - spike_time + 10))]
    dw[ii] = w1 - w0

fig, ax = plt.subplots()
ax.plot(g, dw, 'o')
ax.set_xlabel("G [Hz]")
ax.set_ylabel(r"$\Delta$weight [nS]")
ax.set_title("Weight change modulation")

plt.show()
