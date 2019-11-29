{% extends 'common_group.cpp' %}

{% block maincode %}
	{# USES_VARIABLES { _group_idx } #}
	//// MAIN CODE ////////////
	// scalar code
    const size_t _vectorisation_idx = -1;
    {{scalar_code|autoindent}}

    {{ openmp_pragma('parallel-static') }}
	for(size_t _idx_group_idx=0; _idx_group_idx<_num_group_idx; _idx_group_idx++)
	{
	    // vector code
		const size_t _idx = {{_group_idx}}[_idx_group_idx];
		const size_t _vectorisation_idx = _idx;
        {{vector_code|autoindent}}
	}
{% endblock %}
