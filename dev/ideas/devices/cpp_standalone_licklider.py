standalone_mode = True
plot_results = True

from pylab import *
from brian2 import *
import shutil

if standalone_mode:
    from brian2.devices.cpp_standalone import *
    set_device('cpp_standalone')
else:
    brian_prefs['codegen.target'] = 'weave'
    #brian_prefs['codegen.target'] = 'numpy'

defaultclock.dt = .02 * ms

# Ear and sound
max_delay = 20 * ms # 50 Hz
tau_ear = 1 * ms
sigma_ear = .0
eqs_ear = '''
dx/dt=(sound-x)/tau_ear+0.1*(2./tau_ear)**.5*xi : 1 (unless refractory)
sound=5*sin(2*pi*frequency*t)**3 : 1 # nonlinear distorsion
#sound=5*(sin(4*pi*frequency*t)+.5*sin(6*pi*frequency*t)) : 1 # missing fundamental
frequency=(200+200*t*Hz)*Hz : Hz # increasing pitch
'''
receptors = NeuronGroup(2, eqs_ear, threshold='x>1', reset='x=0',
                        refractory=2*ms)
# Coincidence detectors
min_freq = 50 * Hz
max_freq = 1000 * Hz
num_neurons = 300
tau = 1 * ms
sigma = .1
eqs_neurons = '''
dv/dt=-v/tau+sigma*(2./tau)**.5*xi : 1
'''

neurons = NeuronGroup(num_neurons, eqs_neurons, threshold='v>1', reset='v=0')

synapses = Synapses(receptors, neurons, 'w : 1', pre='v+=w', connect=True)
synapses.w = 0.5
# This should work but doesn't for standalone
#synapses.delay = 'i*1.0/exp(log(min_freq/Hz)+(j*1.0/(num_neurons-1))*log(max_freq/min_freq))*second'
synapses.delay = 'i*1.0/exp(log(50.0)+(j*1.0/299)*log(1000.0/50.0))*second'

spikes = SpikeMonitor(neurons)

net = Network(receptors, neurons, synapses, spikes)

net.run(500 * ms)

if standalone_mode:
    shutil.rmtree('output')
    build(project_dir='output', compile_project=True, run_project=True)
    S = loadtxt('output/results/spikemonitor_codeobject.txt', delimiter=',',
                dtype=[('i', int), ('t', float)])
    i = S['i']
    t = S['t']*second
else:
    i, t = spikes.it

if plot_results:
    plot(t, i, '.', mew=0)
    ylim(0, num_neurons)
    show()
