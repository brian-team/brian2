#!/usr/bin/env python
'''
Spike-timing dependent plasticity
Adapted from Song, Miller and Abbott (2000) and Song and Abbott (2001)

This simulation takes a long time! About 5 minutes.
'''
from time import time

from brian2 import *

# brian_prefs.codegen.target = 'weave'

taum = 10 * ms
taupre = 20 * ms
taupost = taupre
Ee = 0 * mV
vt = -54 * mV
vr = -60 * mV
El = -74 * mV
taue = 5 * ms
gmax = .01
dApre = .01
dApost = -dApre * taupre / taupost * 1.05
dApost *= gmax
dApre *= gmax

eqs_neurons = '''
dv/dt = (ge*(Ee - vr) + El - v)/taum : volt   # the synaptic current is linearized
dge/dt = -ge/taue : 1
'''

# Many Poisson input stimulis.
input = PoissonGroup(1000, rates=15 * Hz)

# Single neuron.
neurons = NeuronGroup(1, eqs_neurons, threshold='v>vt', reset='v=vr')

# STDP synapses between 'input' and 'neurons', of initial random weights (w).
S = Synapses(input, neurons,
             '''w : 1
                dApre/dt = -Apre/taupre : 1 (event-driven)
                dApost/dt = -Apost/taupost : 1 (event-driven)''',
             pre='''ge += w
                    Apre += dApre
                    w=clip(w + Apost, 0, gmax)''',
             post='''Apost += dApost
                     w = clip(w+Apre, 0, gmax)''',
             connect=True)
S.w='rand()*gmax'

# Run the simulation.
start_time = time()
run(100 * second, report='text')
print("Simulation time: {0:.1f} sec".format(time() - start_time))

# Plot
figsize(20, 12)

subplot(211)
title('Weight of each synapse (between input and the neuron group)')
ylabel('Weight')
xlabel('Synapse')
plot(S.w[:] / gmax, '.')

subplot(212)
title('Weight distribution of each synapse (between input and the neuron group)')
ylabel('Occurences')
xlabel('Weight')
hist(S.w[:] / gmax, 20)

tight_layout()
show()
