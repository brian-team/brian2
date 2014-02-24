{% extends 'common_group.cpp' %}
{# USES_VARIABLES { N } #}
{# ALLOWS_SCALAR_WRITE #}

{% block maincode %}
	//// MAIN CODE ////////////

	// scalar code
	const int _vectorisation_idx = 1;
	{{scalar_code|autoindent}}

	for(int _idx=0; _idx<N; _idx++)
	{
	    // vector code
		const int _vectorisation_idx = _idx;
		{{ super() }}
	}
{% endblock %}
