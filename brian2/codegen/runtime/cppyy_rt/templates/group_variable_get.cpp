{# Get variable values template for cppyy backend #}
{# USES_VARIABLES { _group_idx } #}
{% extends 'common_group.cpp' %}

{% block maincode %}
    //// MAIN CODE ////////////
    {% set c_type = cpp_dtype(variables['_variable'].dtype) %}

    const size_t _vectorisation_idx = 1;
    const int _num_indices = _num{{ _group_idx }};

    // Allocate output array (returned via pointer parameter)
    {{ scalar_code | autoindent }}

    for (int _idx_group_idx = 0; _idx_group_idx < _num_indices; _idx_group_idx++) {
        const int _idx = {{ _group_idx }}[_idx_group_idx];
        const size_t _vectorisation_idx = _idx;

        {{ vector_code | autoindent }}

        _output[_idx_group_idx] = _variable;
    }
{% endblock %}
