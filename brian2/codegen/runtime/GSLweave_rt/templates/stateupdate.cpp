{% extends 'common_group.cpp' %}
{# USES_VARIABLES { N } #}
{# ALLOWS_SCALAR_WRITE #}

{% block maincode %}
//// MAIN CODE ////////////
struct _dataholder _GSL_dataholder;
double * _GSL_y = _assign_memory_y();

gsl_odeiv2_system _sys;
_sys.function = _GSL_func;
set_dimension(&_sys.dimension);
_sys.params = &_GSL_dataholder;

gsl_odeiv2_driver * _GSL_driver = gsl_odeiv2_driver_alloc_y_new(&_sys,
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
    _fill_y_vector(&_GSL_dataholder, _GSL_y, _idx);
    _GSL_dataholder._idx = _idx;
    if ({{'gsl_odeiv2_driver_apply(_GSL_driver, &t, t1, _GSL_y)' if GSL_settings['adaptable_timestep']
                else 'gsl_odeiv2_driver_apply_fixed_step(_GSL_driver, &t, dt, 1, _GSL_y)'}} != GSL_SUCCESS)
    {
        printf("Integration error running stateupdate with GSL\n");
        exit(-1);
    }
    gsl_odeiv2_driver_reset(_GSL_driver);
    _empty_y_vector(&_GSL_dataholder, _GSL_y, _idx);
}
{% endblock %}
