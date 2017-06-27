{% extends 'common_group.cpp' %}
{# USES_VARIABLES { N } #}
{# ALLOWS_SCALAR_WRITE #}

{% block support_code_block %}
    {{ common.support_code() }}
    {{ vector_code|write_GSL_support_code(variables, other_variables, variables_in_vector_statements)|autoindent }}
{% endblock %}

{% block maincode %}
	//// MAIN CODE ////////////

    // This allows everything to work correctly for synapses where N is not a
    // constant
    const int _N = {{constant_or_scalar('N', variables['N'])}};
	// scalar code
	const int _vectorisation_idx = 1;
	{{scalar_code|autoindent}}

	for(int _idx=0; _idx<_N; _idx++)
	{
	    // vector code
		const int _vectorisation_idx = _idx;
	}
{% endblock %}
