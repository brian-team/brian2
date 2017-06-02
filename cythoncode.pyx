#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: infer_types=True

import numpy as _numpy
cimport numpy as _numpy
from libc.math cimport sin, cos, tan, sinh, cosh, tanh, exp, log, log10, sqrt, asin, acos, atan, fmod, floor, ceil
cdef extern from "math.h":
    double M_PI
# Import the two versions of std::abs
from libc.stdlib cimport abs  # For integers
from libc.math cimport abs  # For floating point values
from libcpp cimport bool

_numpy.import_array()
cdef extern from "numpy/ndarraytypes.h":
    void PyArray_CLEARFLAGS(_numpy.PyArrayObject *arr, int flags)
from libc.stdlib cimport free

cdef extern from "stdint_compat.h":
    # Longness only used for type promotion
    # Actual compile time size used for conversion
    ctypedef signed int int32_t
    ctypedef signed long int64_t
    ctypedef unsigned long uint64_t
    # It seems we cannot used a fused type here
    cdef int int_(bool)
    cdef int int_(char)
    cdef int int_(short)
    cdef int int_(int)
    cdef int int_(long)
    cdef int int_(float)
    cdef int int_(double)
    cdef int int_(long double)


# support code
#### ADDED MANUALLY
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
    
    int gsl_odeiv2_driver_apply(gsl_odeiv2_driver *d, double *t, double t1, double y[]) nogil


    gsl_odeiv2_driver *gsl_odeiv2_driver_alloc_y_new(
        gsl_odeiv2_system *sys, gsl_odeiv2_step_type *T,
        double hstart, double epsabs, double epsrel) nogil

cdef struct parameters:
    double tau

cdef int func (double t, const double y[], double f[], void *params) nogil:
    cdef double v, v0, tau

    tau = (<double *> params)[0]

    v = y[0]
    v0 = y[1]   

    f[0] = (v0 - v)/10e-3
    f[1] = 0
#### END ADDED MANUALLY


# template-specific support code

def main(_namespace):
    cdef int _idx
    cdef int _vectorisation_idx
        
    _var_tau = _namespace["_var_tau"]
    cdef double tau = _namespace["tau"]
    _var_N = _namespace["_var_N"]
    cdef int64_t N = _namespace["N"]
    _var_v0 = _namespace["_var_v0"]
    cdef _numpy.ndarray[double, ndim=1, mode='c'] _buf__array_neurongroup_v0 = _namespace['_array_neurongroup_v0']
    cdef double * _array_neurongroup_v0 = <double *> _buf__array_neurongroup_v0.data
    cdef int _num_array_neurongroup_v0 = len(_namespace['_array_neurongroup_v0'])
    cdef double v0
    _var_t = _namespace["_var_t"]
    cdef _numpy.ndarray[double, ndim=1, mode='c'] _buf__array_defaultclock_t = _namespace['_array_defaultclock_t']
    cdef double * _array_defaultclock_t = <double *> _buf__array_defaultclock_t.data
    cdef double t
    _var_v = _namespace["_var_v"]
    cdef _numpy.ndarray[double, ndim=1, mode='c'] _buf__array_neurongroup_v = _namespace['_array_neurongroup_v']
    cdef double * _array_neurongroup_v = <double *> _buf__array_neurongroup_v.data
    cdef int _num_array_neurongroup_v = len(_namespace['_array_neurongroup_v'])
    cdef double v
    _var_dt = _namespace["_var_dt"]
    cdef _numpy.ndarray[double, ndim=1, mode='c'] _buf__array_defaultclock_dt = _namespace['_array_defaultclock_dt']
    cdef double * _array_defaultclock_dt = <double *> _buf__array_defaultclock_dt.data
    cdef double dt = _namespace["dt"]
    _var_lastspike = _namespace["_var_lastspike"]
    cdef _numpy.ndarray[double, ndim=1, mode='c'] _buf__array_neurongroup_lastspike = _namespace['_array_neurongroup_lastspike']
    cdef double * _array_neurongroup_lastspike = <double *> _buf__array_neurongroup_lastspike.data
    cdef int _num_array_neurongroup_lastspike = len(_namespace['_array_neurongroup_lastspike'])
    cdef double lastspike
    _var_not_refractory = _namespace["_var_not_refractory"]
    cdef _numpy.ndarray[char, ndim=1, mode='c', cast=True] _buf__array_neurongroup_not_refractory = _namespace['_array_neurongroup_not_refractory']
    cdef bool * _array_neurongroup_not_refractory = <bool *> _buf__array_neurongroup_not_refractory.data
    cdef int _num_array_neurongroup_not_refractory = len(_namespace['_array_neurongroup_not_refractory'])
    cdef bool not_refractory

    if '_owner' in _namespace:
        _owner = _namespace['_owner']

    # scalar code
    _vectorisation_idx = 1
        
    dt = _array_defaultclock_dt[0]
    _lio_1 = dt / tau

    #### ADDED MANUALLY
    cdef int dimension = 2
    cdef double y[2]

    cdef gsl_odeiv2_system sys
    sys.function = func
    sys.dimension = dimension
    sys.params = &tau

    cdef gsl_odeiv2_driver * d
    d = gsl_odeiv2_driver_alloc_y_new(
        &sys, gsl_odeiv2_step_rk2,
        1e-6, 1e-6, 0.0)   
    #### END ADDED MANUALLY
    
    # vector code
    for _idx in range(N):
        _vectorisation_idx = _idx
                
        v0 = _array_neurongroup_v0[_idx]
        t = _array_defaultclock_t[0]
        v = _array_neurongroup_v[_idx]
        lastspike = _array_neurongroup_lastspike[_idx]
        not_refractory = _array_neurongroup_not_refractory[_idx]
        not_refractory = (t - lastspike) > 0.005
        if not_refractory:
            #### ADDED MANUALLY
            y[0] = v
            y[1]= v0
            t1 = t + dt
            status = gsl_odeiv2_driver_apply(d, &t, t1, y)
            _v = y[0]
            #### END ADDED MANUALLY
            #_v = (_lio_1 * ((- v) + v0)) + v
        else:
            _v = v
        if not_refractory:
            v = _v
        _array_neurongroup_not_refractory[_idx] = not_refractory
        _array_neurongroup_v[_idx] = v


