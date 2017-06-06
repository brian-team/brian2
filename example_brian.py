#!/usr/bin/env python
'''
Input-Frequency curve of a IF model.
Network: 1000 unconnected integrate-and-fire neurons (leaky IF)
with an input parameter v0.
The input is set differently for each neuron.
'''
from brian2 import *

tau = 10*ms

@implementation('cython', code='''
from libc.stdlib cimport malloc, free

cdef extern from "gsl/gsl_odeiv2.h":
    # gsl_odeiv2_system
    ctypedef struct gsl_odeiv2_system:
        int (* function) (double t,  double y[], double dydt[], void * params) nogil
        int (* jacobian) (double t,  double y[], double * dfdy, double dfdt[], void * params) nogil
        size_t dimension
        void * params

    ctypedef struct gsl_odeiv2_step
    ctypedef struct gsl_odeiv2_control
    ctypedef struct gsl_odeiv2_evolve
    ctypedef struct gsl_odeiv2_driver

    ctypedef struct gsl_odeiv2_step_type

    gsl_odeiv2_step_type *gsl_odeiv2_step_rk2

    int gsl_odeiv2_driver_apply(
        gsl_odeiv2_driver *d, double *t, double t1, double y[]) nogil

    gsl_odeiv2_driver *gsl_odeiv2_driver_alloc_y_new(
        gsl_odeiv2_system *sys, gsl_odeiv2_step_type *T,
        double hstart, double epsabs, double epsrel) nogil

    int gsl_odeiv2_driver_free(gsl_odeiv2_driver *d) nogil

cdef struct parameters:
    double tau

cdef struct statevar_container:
    double* v
    double* v0

cdef double* assign_memory_y():
	return <double *>malloc(2 * sizeof(double))

cdef int assign_statevariable_arrays(_namespace, statevar_container* statevariables):
    cdef _numpy.ndarray[double, ndim=1, mode='c'] _buf__array_neurongroup_v = _namespace['_array_neurongroup_v']
    cdef double * _array_neurongroup_v = <double *> _buf__array_neurongroup_v.data
    cdef _numpy.ndarray[double, ndim=1, mode='c'] _buf__array_neurongroup_v0 = _namespace['_array_neurongroup_v0']
    cdef double * _array_neurongroup_v0 = <double *> _buf__array_neurongroup_v0.data

    statevariables.v = _array_neurongroup_v
    statevariables.v0 = _array_neurongroup_v0

    return 0

cdef int empty_statevariable_arrays(_namespace, statevar_container* statevariables):
    cdef _numpy.ndarray[double, ndim=1, mode='c'] _buf__array_neurongroup_v = _namespace['_array_neurongroup_v']
    cdef double * _array_neurongroup_v = <double *> _buf__array_neurongroup_v.data
    cdef _numpy.ndarray[double, ndim=1, mode='c'] _buf__array_neurongroup_v0 = _namespace['_array_neurongroup_v0']
    cdef double * _array_neurongroup_v0 = <double *> _buf__array_neurongroup_v0.data

    _array_neurongroup_v = statevariables.v
    _array_neurongroup_v0 = statevariables.v0

    return 0

cdef fill_odeiv2_system(_namespace, gsl_odeiv2_system *sys):
    cdef double tau = _namespace["tau"]
    sys.dimension = 2
    sys.params = &tau

cdef int fill_y_vector(statevar_container* statevariables, double * y, int _idx):
    cdef double v
    v = statevariables.v[_idx]
    cdef double v0
    v0 =  statevariables.v0[_idx]
    y[0] = v
    y[1] = v0
    return 0

cdef int empty_y_vector(statevar_container* statevariables, double * y, int _idx) nogil:
    statevariables.v[_idx] = y[0]
    statevariables.v0[_idx] = y[1]
    return 0

cdef int func (double t, const double y[], double f[], void *params) nogil:
    cdef double v, v0, tau
    tau = (<double *> params)[0]
    v = y[0]
    v0 = y[1]
    f[0] = (v0 - v)/10e-3
    f[1] = 0
''')
@check_units(result=1)
def GSL_functions():
    raise NotImplementedError
    return 0

prefs.codegen.target = 'cython'

prefs.codegen.cpp.libraries += ['gsl', 'gslcblas']
prefs.codegen.cpp.headers += ['gsl/gsl_odeiv2.h']
prefs.codegen.cpp.include_dirs += ['/home/charlee/so   ftwarefolder/gsl-2.3/gsl/']

n = 10
duration = .1*second

eqs = '''
dv/dt = (v0 - v) / tau : volt (unless refractory)
v0 : volt
'''

class GSLStateUpdater(StateUpdateMethod):
    def __call__(self, equations, variables=None):
        return '''not_refractory = (t - lastspike) > .005'''

group = NeuronGroup(n, eqs, threshold='v > 10*mV', reset='v = 0*mV',
                    refractory=5*ms, method=GSLStateUpdater())

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
