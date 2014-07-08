{% extends 'common_group.cpp' %}
{# USES_VARIABLES { N } #}

{% block maincode %}
	{# USES_VARIABLES {_spikespace } #}
	// t, not_refractory and lastspike are added as needed_variables in the
	// Thresholder class, we cannot use the USES_VARIABLE mechanism
	// conditionally

	//// MAIN CODE ////////////
	// scalar code
	const int _vectorisation_idx = 1;
	{{scalar_code|autoindent}}

	long _cpp_numspikes = 0;
	for(int _idx=0; _idx<N; _idx++)
	{
	    // vector code
	    const int _vectorisation_idx = _idx;
		{{ super() }}
		if(_cond) {
			{{_spikespace}}[_cpp_numspikes++] = _idx;
			{% if _uses_refractory %}
			{{not_refractory}}[_idx] = false;
			{{lastspike}}[_idx] = t;
			{% endif %}
		}
	}
	{{_spikespace}}[N] = _cpp_numspikes;
{% endblock %}
