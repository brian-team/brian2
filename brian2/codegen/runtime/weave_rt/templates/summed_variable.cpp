{% extends 'common_group.cpp' %}
{% block maincode %}
    {# USES_VARIABLES { N } #}
    //// MAIN CODE ////////////
    {% set _target_var_array = get_array_name(_target_var) %}
    {% set _index_array = get_array_name(_index_var) %}

    {# This enables summed variables for connections to a synapse #}
    const size_t _target_size = {{constant_or_scalar(_target_size_name, variables[_target_size_name])}};

    // Set all the target variable values to zero
    for (size_t _target_idx=0; _target_idx<_target_size; _target_idx++)
        {{_target_var_array}}[_target_idx + {{_target_start}}] = 0;

    // scalar code
    const size_t _vectorisation_idx = 1;
    {{scalar_code|autoindent}}

    for(size_t _idx=0; _idx<{{N}}; _idx++)
    {
        // vector_code
        const size_t vectorisation_idx = _idx;
        {{ super() }}
        {{_target_var_array}}[{{_index_array}}[_idx]] += _synaptic_var;
    }
{% endblock %}
