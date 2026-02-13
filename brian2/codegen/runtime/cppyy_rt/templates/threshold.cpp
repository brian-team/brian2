{# USES_VARIABLES { N, _spikespace } #}
{% extends 'common_group.cpp' %}

{% block maincode %}
    const size_t _vectorisation_idx = -1;
    {{ scalar_code | autoindent }}
    const int _N = {{ constant_or_scalar('N', variables['N']) }};
    long _count = 0;
    for (int _idx = 0; _idx < _N; _idx++) {
        const size_t _vectorisation_idx = _idx;
        {{ vector_code | autoindent }}
        if (_cond) {
            {{ _spikespace }}[_count++] = _idx;
        }
    }
    {{ _spikespace }}[_N] = _count;
{% endblock %}
