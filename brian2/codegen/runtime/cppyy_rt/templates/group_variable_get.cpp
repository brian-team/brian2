{# Note: used only for subexpressions -- for normal arrays the device accesses
   data directly (see variableview_get_with_index_array) #}
{# USES_VARIABLES { _group_idx } #}
{% extends 'common_group.cpp' %}

{% block maincode %}
    const size_t _vectorisation_idx = 1;
    const int _num_indices = _num_group_idx;

    {{ scalar_code | autoindent }}

    for (int _idx_group_idx = 0; _idx_group_idx < _num_indices; _idx_group_idx++) {
        const int _idx = {{ _group_idx }}[_idx_group_idx];
        const size_t _vectorisation_idx = _idx;

        {{ vector_code | autoindent }}

        _output_buf[_idx_group_idx] = _variable;
    }
{% endblock %}
