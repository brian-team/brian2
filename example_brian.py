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
prefs.codegen.cpp.headers += ['gsl/gsl_odeiv2.h']
prefs.codegen.cpp.include_dirs += ['/home/charlee/softwarefolder/gsl-2.3/gsl/']

n = 10
duration = .1*second
tau = 10*ms
eqs = '''
dv/dt = (v0 - v) / tau : volt (unless refractory)
v0 : volt
'''

#cdef int dimension = 2
#cdef double y[2]
#cdef gsl_odeiv2_system sysmbo
#cdef gsl_odeiv2_driver * d

external_library_functions = {'gsl_odeiv_alloc_y_new' : 'gsl_odeiv_alloc_y_new',
                              'gsl_odeiv2_step_rk2' : 'gsl_odeiv2_step_rk2'}

class GSLStateUpdater(StateUpdateMethod):
    def __call__(self, equations, variables=None):
        return '''
sys.function = func
sys.dimension = 2
sys.params = &tau
d = gsl_odeiv2_driver_alloc_y_new(&sys, gsl_odeiv2_step_rk2,1e-6, 1e-6, 0.0)
not_refractory = (t - lastspike) > .005
y[0] = v
y[1] = v0
t1 = t + dt
status = gsl_odeiv2_driver_apply(d, &t, t1, y)
_v = y[0]
'''

from brian2.core.functions import SymbolicConstant
DEFAULT_CONSTANTS['gsl_odeiv2_step_rk2'] = SymbolicConstant('gsl_odeiv2_step_rk2','gsl_odeiv2_step_rk2', value='gsl_odeiv2_step_rk2')
DEFAULT_CONSTANTS['gsl_odeiv2_driver_apply'] = SymbolicConstant('gsl_odeiv2_driver_apply','gsl_odeiv2_driver_apply', value='gsl_odeiv2_driver_apply')
DEFAULT_CONSTANTS['gsl_odeiv2_driver_alloc_y_new'] = SymbolicConstant('gsl_odeiv2_driver_alloc_y_new','gsl_odeiv2_driver_alloc_y_new', value='gsl_odeiv2_driver_alloc_y_new')
DEFAULT_CONSTANTS['sys'] = SymbolicConstant('sys','sys', value='sys')
DEFAULT_CONSTANTS['func'] = SymbolicConstant('func','func', value='func')
DEFAULT_CONSTANTS['y'] = SymbolicConstant('y','y', value='y')
prefs.codegen.loop_invariant_optimisations = False

group = NeuronGroup(n, eqs, threshold='v > 10*mV', reset='v = 0*mV',
                    refractory=5*ms, method=GSLStateUpdater(), namespace=external_library_functions)

#group = NeuronGroup(n, eqs, threshold='v > 10*mV', reset='v = 0*mV',
#                    refractory=5*ms, method='euler')



group.v = 0*mV
group.v0 = '20*mV * i / (n-1)'

print group.v0

monitor = SpikeMonitor(group)
mon2 = StateMonitor(group, 'v', record=True)

run(duration)
plot(group.v0/mV, monitor.count / duration)
xlabel('v0 (mV)')
ylabel('Firing rate (sp/s)')
show()
