#!/usr/bin/env python3
"""
FORCE training of a Leaky IF model to mimic a sinusoid (5 Hz) oscillator

Nicola, W., Clopath, C.
Supervised learning in spiking neural networks with FORCE training
Nat Commun 8, 2208 (2017)

https://doi.org/10.1038/s41467-017-01827-3

Based on https://github.com/ModelDBRepository/190565/blob/master/CODE%20FOR%20FIGURE%202/LIFFORCESINE.m

Sebastian Schmitt, 2022
"""

from brian2 import NeuronGroup, Synapses, StateMonitor, SpikeMonitor
from brian2 import run, defaultclock, linked_var, Clock
from brian2 import ms, second, Hz
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

# set seed for reproducible figures
np.random.seed(1)

# decay time of synaptic kernal
td = 20*ms

# rise time of synaptic kernal
tr = 2*ms

# membrane time constant
tm = 10*ms

# refractory period
tref = 2*ms

# reset potential
vreset = -65

# peak/threshold potential
vpeak = -40

# bias
BIAS = vpeak

# integration time step
defaultclock.dt = 0.05*ms

# total duration of simulation
T = 15*second

# start of training
imin = 5*second

# end of training
icrit = 10*second

# interval of training
step = 2.5*ms

# feedback scale factor
Q = 10

# neuron-to-neuron connection scale factor
G = 0.04

# connection probability
p = 0.1

# number of neurons
N = 2000

# correlation weight matrix for RLMS
alpha = defaultclock.dt / second * 0.1

# Frequency of target sinusoid
freq = 5 * Hz

readout = NeuronGroup(
    1,
    """
    z : 1 (shared)
    r_cd : 1 (shared)
    """,
)
neurons = NeuronGroup(N,
                      """
                      dv/dt = (-v + BIAS + IPSC + E*z)/tm: 1 (unless refractory)
                      dIPSC/dt = -IPSC/tr + h : 1
                      dh/dt = -h/td : 1/second
                      dr/dt = -r/tr + hr : 1
                      dhr/dt = -hr/td : 1/second
                      z : 1 (linked)
                      r_cd : 1 (linked)
                      cd : 1
                      zx = sin(2 * pi * freq * t) : 1 (shared)
                      err = z - zx : 1 (shared)
                      BPhi : 1
                      E : 1
                      """,
                      method="euler",
                      threshold="v>=vpeak",
                      reset="v=vreset; hr += 1/(tr*td)*second",
                      refractory=tref)
neurons.z = linked_var(readout, "z")
neurons.r_cd = linked_var(readout, "r_cd")
BPhi_update = neurons.run_regularly("BPhi -= cd * err", dt=step, when="end", order=1)

# fixed feedback weights
neurons.E = (2*np.random.uniform(size=N)-1)*Q

# initial membrane voltage
neurons.v = vreset + np.random.uniform(size=N)*(30-vreset)

synapses = Synapses(neurons, neurons, "w : second", on_pre="h += w/(tr*td)")
synapses.connect()
omega = G*(np.random.normal(size=(N,N))*(np.random.uniform(size=(N,N))<p))/(np.sqrt(N)*p)
synapses.w = omega.flatten()*second

spikemon = SpikeMonitor(neurons[:20])
statemon_BPhi = StateMonitor(neurons, "BPhi", record=range(10))
statemon_z = StateMonitor(readout, "z", record=[0])


do_readout = Synapses(
    neurons,
    readout,
    """
    z_post = BPhi_pre*r_pre : 1 (summed)
    r_cd_post = r_pre * cd_pre : 1 (summed)
    """,
)
do_readout.summed_updaters["r_cd_post"]._clock = Clock(dt=step)
do_readout.summed_updaters["r_cd_post"].when = "end"
do_readout.summed_updaters["r_cd_post"].order = 3
do_readout.connect()

training_connections = Synapses(
    neurons,
    neurons,
    """
    Pinv : 1
    cd_post = Pinv * r_pre : 1 (summed)
    """,
)
training_connections.summed_updaters["cd_post"]._clock = Clock(dt=step)
training_connections.summed_updaters["cd_post"].when = "end"
training_connections.summed_updaters["cd_post"].order = 2
training_connections.connect()
training_connections.Pinv = (np.eye(N) * alpha).flatten()
training_connections.run_regularly(
    "Pinv -= (cd_pre * cd_post) / (1 + r_cd_post)", dt=step, when="end", order=4
)

training_connections.active = False
do_readout.summed_updaters["r_cd_post"].active = False
BPhi_update.active = False
run(imin, report="text")

training_connections.active = True
do_readout.summed_updaters["r_cd_post"].active = True
BPhi_update.active = True
run(icrit - imin, report="text")

training_connections.active = False
do_readout.summed_updaters["r_cd_post"].active = False
BPhi_update.active = False
run(T - icrit, report="text")

def zx(t):  # Only used for plotting
    freq = 5*Hz
    return np.sin(2*np.pi*freq*t)

fig, axes = plt.subplots(2,2, figsize=(10,10))
axes = axes.flatten()

axes[0].set_title("Spike raster")
axes[0].scatter(spikemon.t/second,spikemon.i, marker='|', linestyle="None", color="black", s=100)
axes[0].set_xlim((imin-2*second)/second, imin/second+2)
axes[0].set_ylim(0, len(spikemon.source))
axes[0].set_xlabel("t [s]")
axes[0].set_ylabel("Neuron")
axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))

axes[1].plot(statemon_z.t/second, np.sin(2*np.pi*freq*statemon_z.t), linestyle='--', color='k')
axes[1].plot(statemon_z.t/second,statemon_z.z[0])

axes[1].set_title("Target and readout")
axes[1].annotate('RLS ON', xy=(imin/second, -1.05), xytext=(imin/second, -1.35),
            arrowprops=dict(facecolor='black', shrink=1), ha="center")
axes[1].annotate('RLS OFF', xy=(icrit/second, -1.05), xytext=(icrit/second, -1.35),
            arrowprops=dict(facecolor='black', shrink=1), ha="center")
axes[1].set_xlabel("t [s]")
axes[1].set_xlim((imin-1*second)/second, T/second)
axes[1].set_ylim(-1.4,1.1)

axes[2].set_title("Error")
axes[2].plot(statemon_z.t/second, statemon_z.z[0] - np.sin(2*np.pi*freq*statemon_z.t))
axes[2].annotate('RLS ON', xy=(imin/second, -0.15), xytext=(imin/second, -0.4),
            arrowprops=dict(facecolor='black', shrink=1), ha="center")
axes[2].annotate('RLS OFF', xy=(icrit/second, -0.15), xytext=(icrit/second, -0.4),
            arrowprops=dict(facecolor='black', shrink=1), ha="center")
axes[2].set_xlabel("t [s]")
axes[2].set_xlim((imin-1*second)/second, T/second)
axes[2].set_ylim(-1,1)

axes[3].set_title("Decoders")
for j in range(len(statemon_BPhi.record)):
    axes[3].plot(statemon_BPhi.t/second,statemon_BPhi.BPhi[j])
axes[3].set_xlim((imin-1*second)/second, T/second)
axes[3].set_xlabel("t [s]")
axes[3].set_ylim(-0.00020, 0.00015)
axes[3].set_yticklabels([])
axes[3].annotate('RLS ON', xy=(imin/second, -0.0001455), xytext=(imin/second, -0.00019),
            arrowprops=dict(facecolor='black', shrink=1), ha="center")
axes[3].annotate('RLS OFF', xy=(icrit/second, -0.0001455), xytext=(icrit/second, -0.00019),
            arrowprops=dict(facecolor='black', shrink=1), ha="center")

fig.tight_layout()
plt.show()