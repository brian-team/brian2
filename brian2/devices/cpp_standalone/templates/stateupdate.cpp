{# USES_VARIABLES { N } #}
{# ALLOWS_SCALAR_WRITE #}
{% extends 'common_group.cpp' %}

{% block maincode %}
    //// MAIN CODE ////////////
    // scalar code
    const size_t _vectorisation_idx = -1;
    {{scalar_code|autoindent}}

    {# N is a constant in most cases (NeuronGroup, etc.), but a scalar array for
       synapses, we therefore have to take care to get its value in the right
       way. #}
    const int _N = {{constant_or_scalar('N', variables['N'])}};
    {{openmp_pragma('parallel-static')}}
    for(int _idx=0; _idx<_N; _idx++)
    {
        // vector code
        const size_t _vectorisation_idx = _idx;
        {{vector_code|autoindent}}
    }
{% endblock %}
