{% extends 'common_group.cpp' %}
{% block maincode %}
	{# USES_VARIABLES { _spikespace, N } #}

	const int *_spikes = {{_spikespace}};
	const int _num_spikes = {{_spikespace}}[N];

	//// MAIN CODE ////////////
	// scalar code
	const int _vectorisation_idx = -1;
	{{scalar_code|autoindent}}

	for(int _index_spikes=0; _index_spikes<_num_spikes; _index_spikes++)
	{
	    // vector code
		const int _idx = _spikes[_index_spikes];
		const int _vectorisation_idx = _idx;
        {{vector_code|autoindent}}
	}
{% endblock %}
