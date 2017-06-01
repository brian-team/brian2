#!/usr/bin/env python
'''
Input-Frequency curve of a IF model.
Network: 1000 unconnected integrate-and-fire neurons (leaky IF)
with an input parameter v0.
The input is set differently for each neuron.
'''
from brian2 import *

prefs.codegen.target = 'cython'

prefs.codegen.cpp.libraries += ['gsl', 'gslcblas']
prefs.codegen.cpp.extra_compile_args_gcc = ['-lgsl', '-lgslcblas']
prefs.codegen.cpp.extra_link_args = ['-lgsl', '-lgslcblas']
prefs.codegen.cpp.library_dirs += ['/home/charlee/softwarefolder/gsl-2.3/gsl/']
# for some reason adding the above to library_dirs made the loading of the module hang. I googled this
# and saw somebody mentioning hanging on multiprocessing, if I turn it of it doens't hang..
prefs['codegen.runtime.cython.multiprocess_safe'] = False

n = 1000
duration = 1*second
tau = 10*ms
eqs = '''
dv/dt = (v0 - v) / tau : volt (unless refractory)
v0 : volt
'''
group = NeuronGroup(n, eqs, threshold='v > 10*mV', reset='v = 0*mV',
                    refractory=5*ms, method='euler')
group.v = 0*mV
group.v0 = '20*mV * i / (n-1)'

monitor = SpikeMonitor(group)

run(duration)
plot(group.v0/mV, monitor.count / duration)
xlabel('v0 (mV)')
ylabel('Firing rate (sp/s)')
show()
