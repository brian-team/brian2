{% extends 'common_group.cpp' %}

{% block maincode %}
    {# USES_VARIABLES { N } #}
    {% set _target_var_array = get_array_name(_target_var) %}
    {% set _index_array = get_array_name(_index_var) %}
    //// MAIN CODE ////////////
    {# This enables summed variables for connections to a synapse #}
    const int _target_size = {{constant_or_scalar(_target_size_name, variables[_target_size_name])}};

    // Set all the target variable values to zero
    {{ openmp_pragma('parallel-static') }}
    for (int _target_idx=0; _target_idx<_target_size; _target_idx++)
    {
        {{_target_var_array}}[_target_idx] = 0;
    }

    // scalar code
    const int _vectorisation_idx = -1;
    {{scalar_code|autoindent}}

    for(int _idx=0; _idx<{{N}}; _idx++)
    {
        // vector code
        const int _vectorisation_idx = _idx;
        {{vector_code|autoindent}}
        {{_target_var_array}}[{{_index_array}}[_idx]] += _synaptic_var;
    }
{% endblock %}
