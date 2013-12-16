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

from brian2 import *

# brian_prefs.codegen.target = 'weave'

defaultclock.dt = .02 * ms

# Sound
sound = TimedArray(10 * np.random.randn(50000), dt=defaultclock.dt) # white noise

# Ears and sound motion around the head (constant angular speed)
sound_speed = 300 * metre / second
interaural_distance = 20 * cm # big head!
max_delay = interaural_distance / sound_speed
print "Maximum interaural delay:", max_delay
angular_speed = 2 * np.pi / second # 1 turn/second
tau_ear = 1 * ms
sigma_ear = .1
eqs_ears = '''
dx/dt = (sound(t-delay)-x)/tau_ear+sigma_ear*(2./tau_ear)**.5*xi : 1 (unless refractory)
delay = distance*sin(theta) : second
distance : second # distance to the centre of the head in time units
dtheta/dt = angular_speed : radian
'''
ears = NeuronGroup(2, model=eqs_ears, threshold='x>1',
                   reset='x=0', refractory=2.5 * ms, name='ears')
ears.distance = [-.5 * max_delay, .5 * max_delay]
traces = StateMonitor(ears, 'delay', record=True)
# Coincidence detectors
num_neurons = 30
tau = 1 * ms
sigma = .1
eqs_neurons = '''
dv/dt=-v/tau+sigma*(2./tau)**.5*xi : 1
'''
neurons = NeuronGroup(num_neurons, model=eqs_neurons, threshold='v>1',
                      reset='v=0', name='neurons')

synapses = Synapses(ears, neurons, model='w:1', pre='v+=w')
synapses.connect(True)
synapses.w=.5

synapses.delay['i==0'] = '(1.0*j)/(num_neurons-1)*1.1*max_delay'
synapses.delay['i==1'] = '(1.0*(num_neurons-j-1))/(num_neurons-1)*1.1*max_delay'
# This could be also formulated like this (but then it wouldn't work with
# standalone at the moment)
#synapses.delay[0, :] = np.linspace(0 * ms, 1.1 * max_delay, num_neurons)
#synapses.delay[1, :] = np.linspace(0 * ms, 1.1 * max_delay, num_neurons)[::-1]

spikes = SpikeMonitor(neurons)

t1 = time()
run(1000 * ms)
t2 = time()
print "Simulation took",t2-t1,"s"

# Plot the results
i, t = spikes.it
subplot(2, 1, 1)
plot(t / ms, i, '.')
xlabel('time (ms)')
ylabel('neuron index')
xlim(0, 1000)
subplot(2, 1, 2)
plot(traces.t / ms, traces.delay.T / ms)
xlabel('time (ms)')
ylabel('input delay (ms)')
xlim(0, 1000)
show()
