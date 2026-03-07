"""
Example: CUBA benchmark
=======================

This is a Brian script implementing a benchmark described in the following review paper:

Simulation of networks of spiking neurons: A review of tools and strategies (2007). 
Brette, Rudolph, Carnevale, Hines, Beeman, Bower, Diesmann, Goodman, Harris, Zirpe, 
Natschlager, Pecevski, Ermentrout, Djurfeldt, Lansner, Rochel, Vibert, Alvarez, Muller, 
Davison, El Boustani and Destexhe. Journal of Computational Neuroscience 23(3):349-98

Benchmark 2: random network of integrate-and-fire neurons with exponential synaptic currents.

Clock-driven implementation with exact subthreshold integration (but spike times are aligned to the grid).
"""

from brian2 import *
import matplotlib.pyplot as plt
# %%
# Setting up the parameters
# -------------------------
# First, we define all the time constants and voltages needed for the simulation.

taum = 20*ms
taue = 5*ms
taui = 10*ms
Vt = -50*mV
Vr = -60*mV
El = -49*mV

# %%
# Defining the Equations
# ----------------------
# Next, we write the differential equations that control the integrate-and-fire neurons.

eqs = '''
dv/dt  = (ge+gi-(v-El))/taum : volt (unless refractory)
dge/dt = -ge/taue : volt
dgi/dt = -gi/taui : volt
'''

# %%
# Creating the Neuron Group
# -------------------------
# We create a population of 4000 neurons using the exact integration method.

P = NeuronGroup(4000, eqs, threshold='v>Vt', reset='v = Vr', refractory=5*ms,
                method='exact')
P.v = 'Vr + rand() * (Vt - Vr)'
P.ge = 0*mV
P.gi = 0*mV

# %%
# Setting up Synapses
# -------------------
# We split the network into excitatory (first 3200 neurons) and inhibitory (remaining 800) connections.

we = (60*0.27/10)*mV # excitatory synaptic weight (voltage)
wi = (-20*4.5/10)*mV # inhibitory synaptic weight
Ce = Synapses(P, P, on_pre='ge += we')
Ci = Synapses(P, P, on_pre='gi += wi')
Ce.connect('i<3200', p=0.02)
Ci.connect('i>=3200', p=0.02)

# %%
# Running the Simulation
# ----------------------
# We record the spikes and run the simulation for 1 second of biological time.

s_mon = SpikeMonitor(P)
run(1 * second)

# %%
# Plotting the Results
# --------------------
# Finally, we generate a raster plot to visualize the firing pattern of the network.
# In the Sphinx-Gallery documentation, this plot will automatically appear right below this text!

plt.figure() # This tells Sphinx-Gallery "Get the camera ready!"
plt.plot(s_mon.t/ms, s_mon.i, ',k')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')
plt.show() # This takes the picture