{# IS_OPENMP_COMPATIBLE #}
{% extends 'common_group.cpp' %}

{% block maincode %}
    {# USES_VARIABLES { _synaptic_post, _synaptic_pre, N_post } #}
    {% set _target_var_array = get_array_name(_target_var) %}
    //// MAIN CODE ////////////
    // Set all the target variable values to zero
    std::vector<{{c_data_type(_target_var.dtype)}}> _local_sum;
    {# This enables summed variables for connections to a synapse #}
    const int _N_post = {{constant_or_scalar('N_post', variables['N_post'])}};
    _local_sum.resize(_N_post, 0);

    {{ openmp_pragma('parallel-static') }}
    for (int _target_idx=0; _target_idx<_N_post; _target_idx++)
    {
        {{_target_var_array}}[_target_idx] = 0;
    }

    // scalar code
    const int _vectorisation_idx = -1;
    {{scalar_code|autoindent}}

    {{ openmp_pragma('parallel-static') }}
    for(int _idx=0; _idx<_num_synaptic_post; _idx++)
    {
        // vector code
        const int _vectorisation_idx = _idx;
        {{vector_code|autoindent}}
        _local_sum[{{_synaptic_post}}[_idx]] += _synaptic_var;
    }

    {{ openmp_pragma('parallel-static') }}
    for (int _target_idx=0; _target_idx<_N_post; _target_idx++)
    {
        {{ openmp_pragma('atomic') }}
        {{_target_var_array}}[_target_idx] += _local_sum[_target_idx];
    }
{% endblock %}
