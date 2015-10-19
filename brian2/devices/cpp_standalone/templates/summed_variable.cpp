{% extends 'common_group.cpp' %}

{% block maincode %}
    {# USES_VARIABLES { _synaptic_post, _synaptic_pre, N_post } #}
    {% set _target_var_array = get_array_name(_target_var) %}
    //// MAIN CODE ////////////
    {# This enables summed variables for connections to a synapse #}
    const int _N_post = {{constant_or_scalar('N_post', variables['N_post'])}};

    // Set all the target variable values to zero
    {{ openmp_pragma('parallel-static') }}
    for (int _target_idx=0; _target_idx<_N_post; _target_idx++)
    {
        {{_target_var_array}}[_target_idx] = 0;
    }

    // scalar code
    const int _vectorisation_idx = -1;
    {{scalar_code|autoindent}}

    {# The following could be parallelized with OpenMP, but this needs some care
       to not write concurrently to the same post-synaptic variable. A general
       critical block around the write slows things down too much, though. #}
    for(int _idx=0; _idx<_num_synaptic_post; _idx++)
    {
        // vector code
        const int _vectorisation_idx = _idx;
        {{vector_code|autoindent}}
        {{_target_var_array}}[{{_synaptic_post}}[_idx]] += _synaptic_var;
    }
{% endblock %}
