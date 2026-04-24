{# ITERATE_ALL { _idx } #}
{# USES_VARIABLES { N } #}
{# ALLOWS_SCALAR_WRITE #}
{% extends 'common_group.cpp' %}

{% block maincode %}
    // scalar code (runs once, outside the loop)
    const size_t _vectorisation_idx = -1;
    {{ scalar_code | autoindent }}

    const int _N = {{ constant_or_scalar('N', variables['N']) }};

    // vector code (runs per neuron)
    for (int _idx = 0; _idx < _N; _idx++) {
        const size_t _vectorisation_idx = _idx;
        {{ vector_code | autoindent }}
    }
{% endblock %}
