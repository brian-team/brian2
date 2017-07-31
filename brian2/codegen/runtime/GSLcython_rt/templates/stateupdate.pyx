{% extends 'common.pyx' %}

{% block template_support_code %}

from brian2.codegen.runtime.GSLcython_rt import IntegrationError
from libc.stdlib cimport malloc, free

cdef enum:
    GSL_SUCCESS = 0

cdef extern from "gsl/gsl_odeiv2.h":
    # gsl_odeiv2_system
    ctypedef struct gsl_odeiv2_system:
        int (* function) (double t,  double y[], double dydt[], void * params)
        int (* jacobian) (double t,  double y[], double * dfdy, double dfdt[], void * params)
        size_t dimension
        void * params

    ctypedef struct gsl_odeiv2_step
    ctypedef struct gsl_odeiv2_control
    ctypedef struct gsl_odeiv2_evolve
    ctypedef struct gsl_odeiv2_driver

    ctypedef struct gsl_odeiv2_step_type

    gsl_odeiv2_step_type *gsl_odeiv2_step_{{GSL_settings['integrator']}}

    int gsl_odeiv2_driver_apply(
        gsl_odeiv2_driver *d, double *t, double t1, double y[])
    int gsl_odeiv2_driver_apply_fixed_step(
        gsl_odeiv2_driver *d, double *t, const double h, const unsigned long int n, double y[])
    int gsl_odeiv2_driver_reset(
        gsl_odeiv2_driver *d)

    gsl_odeiv2_driver *gsl_odeiv2_driver_alloc_y_new(
        gsl_odeiv2_system *sys, gsl_odeiv2_step_type *T,
        double hstart, double epsabs, double epsrel)

    int gsl_odeiv2_driver_free(gsl_odeiv2_driver *d)
{% endblock %}

{% block maincode %}
    {# ITERATE_ALL { _idx } #}
    {# USES_VARIABLES { N } #}
    {# ALLOWS_SCALAR_WRITE #}

    # scalar code
    _vectorisation_idx = 1
    dt = {{dt_array}}

    cdef double t1
    cdef dataholder * p = <dataholder *>malloc(sizeof(dataholder))

    {{scalar_code['GSL']|autoindent}}

    cdef double * y = assign_memory_y()
    
    cdef gsl_odeiv2_system sys
    cdef gsl_odeiv2_driver * d
    sys.function = func
    set_dimension(&sys.dimension)
    sys.params = p
    
    d = gsl_odeiv2_driver_alloc_y_new(&sys,
                                      gsl_odeiv2_step_{{GSL_settings['integrator']}},
                                      {{GSL_settings['h_start']}},
                                      {{GSL_settings['eps_abs']}},
                                      {{GSL_settings['eps_rel']}})

    # vector code
    for _idx in range(N):
        _vectorisation_idx = _idx

        t = {{t_array}}
        t1 = t + dt

        p._idx = _idx
        fill_y_vector(p, y, _idx)
        if not {{'gsl_odeiv2_driver_apply(d, &t, t1, y)' if GSL_settings['adaptable_timestep']
                    else 'gsl_odeiv2_driver_apply_fixed_step(d, &t, dt, 1, y)'}} == GSL_SUCCESS:
            raise IntegrationError
        empty_y_vector(p, y, _idx)
        gsl_odeiv2_driver_reset(d)

{% endblock %}