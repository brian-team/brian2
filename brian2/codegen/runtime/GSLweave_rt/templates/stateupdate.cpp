{% extends 'common_group.cpp' %}
{# USES_VARIABLES { N } #}
{# ALLOWS_SCALAR_WRITE #}

{% block maincode %}
//// MAIN CODE ////////////
struct dataholder p;
double * y = assign_memory_y();

gsl_odeiv2_system sys;
sys.function = func;
set_dimension(&sys.dimension);
sys.params = &p;

gsl_odeiv2_driver * d = gsl_odeiv2_driver_alloc_y_new(&sys,
                                  gsl_odeiv2_step_{{GSL_settings['integrator']}},
                                  {{GSL_settings['h_start']}},
                                  {{GSL_settings['eps_abs']}},
                                  {{GSL_settings['eps_rel']}});

// This allows everything to work correctly for synapses where N is not a
// constant
const int _N = {{constant_or_scalar('N', variables['N'])}};
// scalar code
const int _vectorisation_idx = 1;
const double dt = _array_defaultclock_dt[0];

{{scalar_code['GSL']|autoindent}}

for(int _idx=0; _idx<_N; _idx++)
{
    // vector code
    const int _vectorisation_idx = _idx;
    double t = _array_defaultclock_t[0];
    double t1 = t + dt;
    fill_y_vector(&p, y, _idx);
    p._idx = _idx;
    if ({{'gsl_odeiv2_driver_apply(d, &t, t1, y)' if GSL_settings['adaptable_timestep']
                else 'gsl_odeiv2_driver_apply_fixed_step(d, &t, dt, 1, y)'}} != GSL_SUCCESS)
    {
        printf("Integration error running stateupdate with GSL\n");
        exit(-1);
    }
    gsl_odeiv2_driver_reset(d);
    empty_y_vector(&p, y, _idx);
}
{% endblock %}
