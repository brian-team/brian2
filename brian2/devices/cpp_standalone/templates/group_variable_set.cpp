{# USES_VARIABLES { _group_idx } #}
{% extends 'common_group.cpp' %}

{% block maincode %}
    //// MAIN CODE ////////////
    // scalar code
    const size_t _vectorisation_idx = -1;
    {{scalar_code|autoindent}}

    {{ openmp_pragma('parallel-static') }}
    for(int _idx_group_idx=0; _idx_group_idx<(int)_num_group_idx; _idx_group_idx++)
    {
        // vector code
        const size_t _idx = {{_group_idx}}[_idx_group_idx];
        const size_t _vectorisation_idx = _idx;
        {{vector_code|autoindent}}
    }
{% endblock %}
