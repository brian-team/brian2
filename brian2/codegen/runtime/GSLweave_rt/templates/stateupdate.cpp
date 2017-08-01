{% extends 'common_group.cpp' %}
{# USES_VARIABLES { N } #}
{# ALLOWS_SCALAR_WRITE #}

{% block maincode %}
//// MAIN CODE ////////////
struct _dataholder _p;
double * _y = _assign_memory_y();

gsl_odeiv2_system _sys;
_sys.function = _func;
set_dimension(&_sys.dimension);
_sys.params = &_p;

gsl_odeiv2_driver * _d = gsl_odeiv2_driver_alloc_y_new(&_sys,
                                  gsl_odeiv2_step_{{GSL_settings['integrator']}},
                                  {{GSL_settings['h_start']}},
                                  {{GSL_settings['eps_abs']}},
                                  {{GSL_settings['eps_rel']}});

// This allows everything to work correctly for synapses where N is not a
// constant
const int _N = {{constant_or_scalar('N', variables['N'])}};
// scalar code
const int _vectorisation_idx = 1;
const double dt = {{dt_array}};

{{scalar_code['GSL']|autoindent}}

for(int _idx=0; _idx<_N; _idx++)
{
    // vector code
    const int _vectorisation_idx = _idx;
    double t = {{t_array}};
    double t1 = t + dt;
    _fill_y_vector(&_p, _y, _idx);
    _p._idx = _idx;
    if ({{'gsl_odeiv2_driver_apply(_d, &t, t1, _y)' if GSL_settings['adaptable_timestep']
                else 'gsl_odeiv2_driver_apply_fixed_step(_d, &t, dt, 1, _y)'}} != GSL_SUCCESS)
    {
        printf("Integration error running stateupdate with GSL\n");
        exit(-1);
    }
    gsl_odeiv2_driver_reset(_d);
    _empty_y_vector(&_p, _y, _idx);
}
{% endblock %}
