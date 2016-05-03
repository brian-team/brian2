{% extends 'common.pyx' %}

{% block maincode %}

    {# USES_VARIABLES { _synaptic_post, N_post } #}
    {% set _target_var_array = get_array_name(_target_var) %}

    cdef int _target_idx

    # Set all the target variable values to zero
    for _target_idx in range(N_post):
        {{_target_var_array}}[_target_idx] = 0

    # scalar code
    _vectorisation_idx = 1
    {{scalar_code|autoindent}}

    for _idx in range(_num{{_synaptic_post}}):
        # vector_code
        vectorisation_idx = _idx
        {{ vector_code | autoindent }}
        {{_target_var_array}}[{{_synaptic_post}}[_idx]] += _synaptic_var

{% endblock %}
