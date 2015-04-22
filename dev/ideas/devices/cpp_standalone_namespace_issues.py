from brian2 import *
import os, shutil

set_device('cpp_standalone')

clock = Clock(name='clock')

N = 25
tau = 20 * ms
sigma = .015
eqs_neurons = '''
dx/dt=(1.1-x)/tau+sigma*(2./tau)**.5*xi:1
'''
neurons = NeuronGroup(N, model=eqs_neurons, threshold='x>1', reset='x=0', clock=clock)
neurons.refractory = 5*ms
spikes = SpikeMonitor(neurons)

run(500 * ms)

if os.path.exists('output'):
    shutil.rmtree('output')

device.build(project_dir='output', compile_project=True, run_project=True)

#i = fromfile('output/results/spikemonitor_codeobject_i', dtype=int32)
#t = fromfile('output/results/spikemonitor_codeobject_t', dtype=float64) * second
#plot(t, i, '.k')
#show()
