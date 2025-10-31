{# USES_VARIABLES { N } #}
{% extends 'common_group.cpp' %}
{% block maincode %}

    {#  Get the name of the array that stores these events (e.g. the spikespace array) #}
    {% set _eventspace = get_array_name(eventspace_variable) %}
	const int32_t *_events = {{_eventspace}};
	const int32_t _num_events = {{_eventspace}}[N];

	//// MAIN CODE ////////////
	// scalar code
	const size_t _vectorisation_idx = -1;
	{{scalar_code|autoindent}}

	{{ openmp_pragma('parallel-static') }}
	for(int32_t _index_events=0; _index_events<_num_events; _index_events++)
	{
	    // vector code
		const size_t _idx = _events[_index_events];
		const size_t _vectorisation_idx = _idx;
        {{vector_code|autoindent}}
	}
{% endblock %}
