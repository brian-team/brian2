{# USES_VARIABLES { _group_idx } #}
{% extends 'common_group.cpp' %}

{% block maincode %}
    const size_t _vectorisation_idx = -1;
    {{ scalar_code | autoindent }}

    // Note: for cppyy runtime, _return_values handling needs special care.
    // The numpy code object uses exec() which can set variables in a namespace.
    // For cppyy we compute into an output array parameter instead.
    for (int _idx_group_idx = 0; _idx_group_idx < (int)_num_group_idx; _idx_group_idx++) {
        const size_t _idx = {{ _group_idx }}[_idx_group_idx];
        const size_t _vectorisation_idx = _idx;
        {{ vector_code | autoindent }}
    }
{% endblock %}
