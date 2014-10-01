{# IS_OPENMP_COMPATIBLE #}
{% extends 'common_group.cpp' %}
{% block maincode %}
	{# USES_VARIABLES { _spikespace, N } #}

	const int32_t *_spikes = {{_spikespace}};
	const int32_t _num_spikes = {{_spikespace}}[N];

	//// MAIN CODE ////////////	
	// scalar code
	const int _vectorisation_idx = -1;
	{{scalar_code|autoindent}}
    
	{{ openmp_pragma('static') }}
	for(int _index_spikes=0; _index_spikes<_num_spikes; _index_spikes++)
	{
	    // vector code
		const int _idx = _spikes[_index_spikes];
		const int _vectorisation_idx = _idx;
        {{vector_code|autoindent}}
	}
{% endblock %}
