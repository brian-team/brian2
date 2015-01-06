{# IS_OPENMP_COMPATIBLE #}
{% extends 'common_group.cpp' %}
{% block maincode %}
	{# USES_VARIABLES { t, _spikespace, N } #}
	// not_refractory and lastspike are added as needed_variables in the
	// Thresholder class, we cannot use the USES_VARIABLE mechanism
	// conditionally

	//// MAIN CODE ////////////
	// scalar code
	const int _vectorisation_idx = -1;
	{{scalar_code|autoindent}}

	
	{{ openmp_pragma('single') }}
    {
        long _count = 0;
        for(int _idx=0; _idx<N; _idx++)
        {
            const int _vectorisation_idx = _idx;
            {{vector_code|autoindent}}
            if(_cond) {
                {{_spikespace}}[_count++] = _idx;
                {% if _uses_refractory %}
                // We have to use the pointer names directly here: The condition
                // might contain references to not_refractory or lastspike and in
                // that case the names will refer to a single entry.
                {{not_refractory}}[_idx] = false;
                {{lastspike}}[_idx] = t;
                {% endif %}
            }
        }
        {{_spikespace}}[N] = _count;
    }
{% endblock %}
