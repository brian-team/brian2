{# USES_VARIABLES { N } #}
{% extends 'common_group.cpp' %}

{% block maincode %}
    const size_t _vectorisation_idx = -1;
    {{ scalar_code['condition'] | autoindent }}
    {{ scalar_code['statement'] | autoindent }}
    const int _N = {{ constant_or_scalar('N', variables['N']) }};
    for (int _idx = 0; _idx < _N; _idx++) {
        const size_t _vectorisation_idx = _idx;
        {{ vector_code['condition'] | autoindent }}
        if (_cond) {
            {{ vector_code['statement'] | autoindent }}
        }
    }
{% endblock %}
