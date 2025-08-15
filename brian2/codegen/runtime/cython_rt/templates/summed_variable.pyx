{# USES_VARIABLES { N } #}
{% extends 'common_group.pyx' %}

{% block maincode %}
    {% set _target_var_array = get_array_name(_target_var) %}
    {% set _index_array = get_array_name(_index_var) %}
    cdef int _target_idx

    # Set all the target variable values to zero
    for _target_idx in range({{_target_size_name}}):
        {% if _target_contiguous %}
        {{_target_var_array}}[_target_idx + {{_target_start}}] = 0
        {% else %}
        {{_target_var_array}}[{{_target_indices}}[_target_idx]] = 0
        {% endif %}

    # scalar code
    _vectorisation_idx = 1
    {{scalar_code|autoindent}}

    for _idx in range({{N}}):
        # vector_code
        vectorisation_idx = _idx
        {{ vector_code | autoindent }}
        {{_target_var_array}}[{{_index_array}}[_idx]] += _synaptic_var

{% endblock %}
