{% extends 'common_group.cpp' %}
{# USES_VARIABLES { N } #}
{# ALLOWS_SCALAR_WRITE #}

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
		{{ super() }}
	}
{% endblock %}
