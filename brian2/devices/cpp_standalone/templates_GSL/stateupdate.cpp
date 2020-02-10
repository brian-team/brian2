{# USES_VARIABLES { N } #}
{# ALLOWS_SCALAR_WRITE #}
{% extends 'common_group.cpp' %}

{% block maincode %}
//// MAIN CODE ////////////
struct _dataholder _GSL_dataholder;
double _GSL_y[{{n_diff_vars}}];
{{define_GSL_scale_array}}

gsl_odeiv2_system _sys;
_sys.function = _GSL_func;
set_dimension(&_sys.dimension);
_sys.params = &_GSL_dataholder;

gsl_odeiv2_driver * _GSL_driver =
        gsl_odeiv2_driver_alloc_scaled_new(&_sys,gsl_odeiv2_step_{{GSL_settings['integrator']}},
                                          {{GSL_settings['dt_start']}}, 1, 0, 0, 0, _GSL_scale_array);
gsl_odeiv2_driver_set_nmax(_GSL_driver, {{GSL_settings['max_steps']}});
gsl_odeiv2_driver_set_hmax(_GSL_driver, {{GSL_settings['dt_start']}});
// This allows everything to work correctly for synapses where N is not a
// constant
const int _N = {{constant_or_scalar('N', variables['N'])}};
// scalar code
const int _vectorisation_idx = 1;
{% if define_dt %}
const double dt = {{dt_array}};
{% endif %}
{{scalar_code['GSL']|autoindent}}

for(int _idx=0; _idx<_N; _idx++)
{
    // vector code
    const int _vectorisation_idx = _idx;
    double t = {{t_array}};
    double t1 = t + dt;
    _fill_y_vector(&_GSL_dataholder, _GSL_y, _idx);
    _GSL_dataholder._idx = _idx;
    {%if GSL_settings['use_last_timestep']%}
    gsl_odeiv2_driver_reset_hstart(_GSL_driver, {{pointer_last_timestep}});
    {% else %}
    gsl_odeiv2_driver_reset(_GSL_driver);
    {% endif %}
    if ({{'gsl_odeiv2_driver_apply(_GSL_driver, &t, t1, _GSL_y)' if GSL_settings['adaptable_timestep']
                else 'gsl_odeiv2_driver_apply_fixed_step(_GSL_driver, &t, dt, 1, _GSL_y)'}} != GSL_SUCCESS)
    {
        {% if cpp_standalone %}
        exit(1);
        {% else %}
        PyErr_SetString(PyExc_RuntimeError, ("GSL integrator failed to integrate the equations."
            {% if GSL_settings['adaptable_timestep'] %}
                                           "\nThis means that the desired error cannot be achieved with the given maximum number of steps. "
                                           "Try using a larger error or a larger number of steps."
            {% else %}
                                           "\n This means that the size of the timestep results in an error larger than that set by absolute_error."
            {% endif %}
                                           ));
        throw 1;
        {% endif %}
    }
    {%if GSL_settings['use_last_timestep']%}
    {{pointer_last_timestep}} = _GSL_driver->h;
    {% endif %}
    {%if GSL_settings['save_failed_steps']%}
    {{pointer_failed_steps}} = _GSL_driver->e->failed_steps;
    {% endif %}
    {%if GSL_settings['save_step_count']%}
    {{pointer_step_count}} = _GSL_driver->n;
    {% endif %}
    _empty_y_vector(&_GSL_dataholder, _GSL_y, _idx);
}
gsl_odeiv2_driver_free(_GSL_driver);
{% endblock %}
