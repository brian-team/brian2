{% extends 'common.pyx' %}

{% block maincode %}
    {# ITERATE_ALL { _idx } #}
    {# USES_VARIABLES { N } #}
    {# ALLOWS_SCALAR_WRITE #}

    # scalar code
    _vectorisation_idx = 1

    dt = _array_defaultclock_dt[0]
    {{scalar_code|autoindent}}

    cdef double * y = assign_memory_y()
    cdef statevar_container * statevariables = <statevar_container *>malloc(sizeof(statevar_container))
    assign_statevariable_arrays(_namespace, statevariables)
    cdef gsl_odeiv2_system sys
    cdef gsl_odeiv2_driver * d
    sys.function = func
    fill_odeiv2_system(_namespace, &sys)
    
    d = gsl_odeiv2_driver_alloc_y_new(
        &sys, gsl_odeiv2_step_rk2, # can also make this a pointer to chosen integrator
        1e-6, 1e-6, 0.0)       

    # vector code
    for _idx in range(N):
        _vectorisation_idx = _idx
        {{vector_code|autoindent}}

        t = _array_defaultclock_t[0]
        t1 = t + dt
        if not_refractory:
            fill_y_vector(statevariables, y, _idx)
            gsl_odeiv2_driver_apply(d, &t, t1, y)
            empty_y_vector(statevariables, y, _idx)

    empty_statevariable_arrays(_namespace, statevariables)


{% endblock %}
    