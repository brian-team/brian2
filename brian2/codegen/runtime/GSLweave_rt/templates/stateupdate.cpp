{% extends 'common_group.cpp' %}
{# USES_VARIABLES { N } #}
{# ALLOWS_SCALAR_WRITE #}

{% block support_code_block %}
    {{ common.support_code() }}
    {{ vector_code|write_GSL_support_code(variables, extra_information)|autoindent }}
{% endblock %}

{% block maincode %}
	//// MAIN CODE ////////////
struct parameters p;
double * y = assign_memory_y();

gsl_odeiv2_system sys;
sys.function = func;
set_dimension(&sys.dimension);
sys.params = &p;

double h = 1e-6;
double eps_abs = 1e-8;
double eps_rel = 1e-10;

gsl_odeiv2_driver * d =
    gsl_odeiv2_driver_alloc_y_new(&sys, gsl_odeiv2_step_rk8pd, h, eps_abs, eps_rel);

    // This allows everything to work correctly for synapses where N is not a
    // constant
    const int _N = {{constant_or_scalar('N', variables['N'])}};
	// scalar code
	const int _vectorisation_idx = 1;
	const double dt = _array_defaultclock_dt[0];

    {{vector_code|add_GSL_declarations(variables, extra_information)|autoindent}}
    {{scalar_code|add_GSL_scalar_code(extra_information)|autoindent}}

	for(int _idx=0; _idx<_N; _idx++)
	{
	    // vector code
	    const int _vectorisation_idx = _idx;
		double t = _array_defaultclock_t[0];
		double t1 = t + dt;
		fill_y_vector(&p, y, _idx);
		p._idx = _idx;
        //temp_func_a(&p, _idx);
		if (gsl_odeiv2_driver_apply(d, &t, t1, y) != GSL_SUCCESS)
		{
		    printf("Integration error running stateupdate with GSL\n");
		    exit(-1);
		}
		gsl_odeiv2_driver_reset(d);
		empty_y_vector(&p, y, _idx);
	}
{% endblock %}
