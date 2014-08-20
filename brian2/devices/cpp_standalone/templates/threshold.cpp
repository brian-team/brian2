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

	const int _padding = {{ openmp_pragma('get_thread_num') }}*(N/{{ openmp_pragma('get_num_threads') }});
    int         _count = 0;
	
	{{ openmp_pragma('static') }}		
	for(int _idx=0; _idx<N; _idx++)
	{
	    const int _vectorisation_idx = _idx;
		{{vector_code|autoindent}}
		if(_cond) {
			{{_spikespace}}[_padding + _count] = _idx;
			_count++;
			{% if _uses_refractory %}
			// We have to use the pointer names directly here: The condition
			// might contain references to not_refractory or lastspike and in
			// that case the names will refer to a single entry.
			{{not_refractory}}[_idx] = false;
			{{lastspike}}[_idx] = t;
			{% endif %}
		}
	}

	{{ openmp_pragma('static-ordered') }}
	for(int _thread=0; _thread < {{ openmp_pragma('get_num_threads') }}; _thread++)
	{
		{{ openmp_pragma('ordered') }}
		{
			// First we ask node 0 to set the total number of spikes to 0
			if (_thread == 0)
				{{_spikespace}}[N] = _count;
			if (_thread > 0)
			{
				for(int _idx=0; _idx<_count; _idx++)
					{{_spikespace}}[{{_spikespace}}[N]++] = {{_spikespace}}[_padding + _idx];
			}
		}
	}

{% endblock %}
