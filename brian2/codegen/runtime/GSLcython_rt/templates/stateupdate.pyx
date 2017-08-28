{% extends 'common.pyx' %}

{% block template_support_code %}

from brian2.codegen.runtime.GSLcython_rt import IntegrationError
from libc.stdlib cimport malloc, free

cdef enum:
    GSL_SUCCESS = 0

cdef extern from "gsl/gsl_odeiv2.h":
    # gsl_odeiv2_system
    ctypedef struct gsl_odeiv2_system:
        int (* function) (double t,  double _GSL_y[], double dydt[], void * params)
        int (* jacobian) (double t,  double _GSL_y[], double * dfdy, double dfdt[], void * params)
        size_t dimension
        void * params

    ctypedef struct gsl_odeiv2_driver:
        gsl_odeiv2_system * sys
        gsl_odeiv2_step *s
        gsl_odeiv2_control *c
        gsl_odeiv2_evolve *e
        double h
        double hmin
        double hmax
        unsigned long int n
        unsigned long int nmax

    ctypedef struct gsl_odeiv2_evolve:
        size_t dimension
        double *y0
        double *yerr
        double *dydt_in
        double *dydt_out
        double last_step
        unsigned long int count
        unsigned long int failed_steps
        const gsl_odeiv2_driver *driver

    ctypedef struct gsl_odeiv2_step
    ctypedef struct gsl_odeiv2_control

    ctypedef struct gsl_odeiv2_step_type

    gsl_odeiv2_step_type *gsl_odeiv2_step_{{GSL_settings['integrator']}}

    int gsl_odeiv2_driver_apply(
        gsl_odeiv2_driver *_GSL_driver, double *t, double t1, double _GSL_y[])
    int gsl_odeiv2_driver_apply_fixed_step(
        gsl_odeiv2_driver *_GSL_driver, double *t, const double h, const unsigned long int n, double _GSL_y[])
    int gsl_odeiv2_driver_reset_hstart(
        gsl_odeiv2_driver *_GSL_driver, const double hstart)
    int gsl_odeiv2_driver_reset(
        gsl_odeiv2_driver *GSL_driver)

    gsl_odeiv2_driver *gsl_odeiv2_driver_alloc_scaled_new(
        gsl_odeiv2_system *_sys, gsl_odeiv2_step_type *T,
        double hstart, double epsabs, double epsrel,
        double a_y, double a_dydt, double scale[])

    int gsl_odeiv2_driver_free(gsl_odeiv2_driver *_GSL_driver)
{% endblock %}

{% block maincode %}
    {# ITERATE_ALL { _idx } #}
    {# USES_VARIABLES { N } #}
    {# ALLOWS_SCALAR_WRITE #}

    # scalar code
    _vectorisation_idx = 1
    dt = {{dt_array}}

    cdef double t1
    cdef _dataholder * _GSL_dataholder = <_dataholder *>malloc(sizeof(_dataholder))

    {{scalar_code['GSL']|autoindent}}

    cdef double * _GSL_y = _assign_memory_y()
    cdef double * _GSL_scale_array = _get_GSL_scale_array()
    
    cdef gsl_odeiv2_system _sys
    _sys.function = _GSL_func
    set_dimension(&_sys.dimension)
    _sys.params = _GSL_dataholder

    cdef gsl_odeiv2_driver * _GSL_driver = gsl_odeiv2_driver_alloc_scaled_new(&_sys,gsl_odeiv2_step_{{GSL_settings['integrator']}},
                                              {{GSL_settings['dt_start']}},1,0,0,0,_GSL_scale_array);

    # vector code
    for _idx in range(N):
        _vectorisation_idx = _idx

        t = {{t_array}}
        t1 = t + dt

        _GSL_dataholder._idx = _idx
        _fill_y_vector(_GSL_dataholder, _GSL_y, _idx)
        {%if GSL_settings['use_last_timestep']%}
        gsl_odeiv2_driver_reset_hstart(_GSL_driver, {{pointer_last_timestep}})
        {% else %}
        gsl_odeiv2_driver_reset(_GSL_driver)
        {% endif %}
        if ({{'gsl_odeiv2_driver_apply(_GSL_driver, &t, t1, _GSL_y)' if GSL_settings['adaptable_timestep']
                    else 'gsl_odeiv2_driver_apply_fixed_step(_GSL_driver, &t, dt, 1, _GSL_y)'}} != GSL_SUCCESS):
            raise IntegrationError(("GSL integrator returned integration error. Reasons for this (amongst others) could be:"
                                    "\nIf adaptable_timestep is set to False: "
                                    "\n   the size of the timestep results in an error larger than that set by absolute_error."
                                    "\nIf adaptable_timestep is set to True:"
                                    "\n   the desired absolute_error cannot be achieved with current settings."))
        _empty_y_vector(_GSL_dataholder, _GSL_y, _idx)
        {%if GSL_settings['use_last_timestep']%}
        {{pointer_last_timestep}} = _GSL_driver.h
        {% endif %}
        {%if GSL_settings['save_failed_steps']%}
        {{pointer_failed_steps}} = _GSL_driver.e.failed_steps
        {% endif %}
        {%if GSL_settings['save_step_count']%}
        {{pointer_step_count}} = _GSL_driver.n
        {% endif %}

{% endblock %}
