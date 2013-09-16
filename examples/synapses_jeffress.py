#!/usr/bin/env python
'''
Jeffress model, adapted with spiking neuron models.
A sound source (white noise) is moving around the head.
Delay differences between the two ears are used to determine the azimuth of the source.
Delays are mapped to a neural place code using delay lines (each neuron receives input
from both ears, with different delays).

Romain Brette
'''
from time import time
import numpy as np
import matplotlib.pyplot as plt

from brian2 import *

# brian_prefs.codegen.target = 'weave'

defaultclock.dt = .02 * ms
dt = defaultclock.dt

# Sound
sound = TimedArray(10 * np.random.randn(50000), dt=dt) # white noise

# Ears and sound motion around the head (constant angular speed)
sound_speed = 300 * metre / second
interaural_distance = 20 * cm # big head!
max_delay = interaural_distance / sound_speed
print "Maximum interaural delay:", max_delay
angular_speed = 2 * np.pi / second # 1 turn/second
tau_ear = 1 * ms
sigma_ear = .1
eqs_ears = '''
dx/dt = (sound(t-delay)-x)/tau_ear+sigma_ear*(2./tau_ear)**.5*xi : 1 (unless-refractory)
delay = distance*sin(theta) : second
distance : second # distance to the centre of the head in time units
dtheta/dt = angular_speed : radian
'''
ears = NeuronGroup(2, model=eqs_ears, threshold='x>1',
                   reset='x=0', refractory=2.5 * ms)
ears.distance = [-.5 * max_delay, .5 * max_delay]
traces = StateMonitor(ears, 'delay', record=True)
# Coincidence detectors
N = 300
tau = 1 * ms
sigma = .1
eqs_neurons = '''
dv/dt=-v/tau+sigma*(2./tau)**.5*xi : 1
'''
neurons = NeuronGroup(N, model=eqs_neurons, threshold='v>1', reset='v=0')

synapses = Synapses(ears, neurons, model='w:1', pre='v+=w')
synapses.connect(True)
synapses.w=.5
synapses.delay[0, :] = np.linspace(0 * ms, 1.1 * max_delay, N)
synapses.delay[1, :] = np.linspace(0 * ms, 1.1 * max_delay, N)[::-1]

spikes = SpikeMonitor(neurons)

run(0*ms)
t1 = time()
run(1000 * ms)
t2 = time()
print "Simulation took",t2-t1,"s"

# Plot the results
i, t = spikes.it
plt.subplot(2, 1, 1)
plt.plot(t / ms, i, '.')
plt.xlabel('time (ms)')
plt.ylabel('neuron index')
plt.xlim(0, 1000)
plt.subplot(2, 1, 2)
plt.plot(traces.t / ms, traces.delay.T / ms)
plt.xlabel('time (ms)')
plt.ylabel('input delay (ms)')
plt.xlim(0, 1000)
plt.show()
