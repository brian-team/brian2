'''
Spike-based adaptation of Licklider's model of pitch processing (autocorrelation with
delay lines) with phase locking.

Romain Brette
'''
from pylab import *
from brian2 import *

defaultclock.dt = .02 * ms

# Ear and sound
max_delay = 20 * ms # 50 Hz
tau_ear = 1 * ms
sigma_ear = .1
eqs_ear = '''
dx/dt=(sound-x)/tau_ear+sigma_ear*(2./tau_ear)**.5*xi : 1 (unless-refractory)
sound=5*sin(2*pi*frequency*t)**3 : 1 # nonlinear distorsion
#sound=5*(sin(4*pi*frequency*t)+.5*sin(6*pi*frequency*t)) : 1 # missing fundamental
frequency=(200+200*t*Hz)*Hz : Hz # increasing pitch
'''
receptors = NeuronGroup(2, eqs_ear, threshold='x>1', reset='x=0',
                        refractory=2*ms)
# Coincidence detectors
min_freq = 50 * Hz
max_freq = 1000 * Hz
N = 300
tau = 1 * ms
sigma = .1
eqs_neurons = '''
dv/dt=-v/tau+sigma*(2./tau)**.5*xi : 1
'''

neurons = NeuronGroup(N, eqs_neurons, threshold='v>1', reset='v=0')

synapses = Synapses(receptors, neurons, 'w : 1', pre='v+=w', connect=True)
synapses.w=0.5
synapses.delay[1, :] = 1. / exp(linspace(log(min_freq / Hz), log(max_freq / Hz), N)) * second

spikes = SpikeMonitor(neurons)

run(500 * ms)
plot(spikes.t, spikes.i, '.', mew=0)
ylabel('Frequency')
yticks([0, 99, 199, 299], array(1. / synapses.delay[1, [0, 99, 199, 299]], dtype=int))
show()
