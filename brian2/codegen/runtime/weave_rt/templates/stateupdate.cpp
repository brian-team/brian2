{% extends 'common_group.cpp' %}
{# USES_VARIABLES { N } #}
{# ALLOWS_SCALAR_WRITE #}

{% block maincode %}
    //// MAIN CODE ////////////

    // This allows everything to work correctly for synapses where N is not a
    // constant
    const size_t _N = {{constant_or_scalar('N', variables['N'])}};
    // scalar code
    const size_t _vectorisation_idx = 1;
    {{scalar_code|autoindent}}

    for(size_t _idx=0; _idx<_N; _idx++)
    {
        // vector code
        const size_t _vectorisation_idx = _idx;
        {{ super() }}
    }
{% endblock %}
