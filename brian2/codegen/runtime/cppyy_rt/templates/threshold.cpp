{# USES_VARIABLES { N } #}
{% extends 'common_group.cpp' %}

{% block maincode %}
    {% set _eventspace = get_array_name(eventspace_variable) %}
    const size_t _vectorisation_idx = -1;
    {{ scalar_code | autoindent }}
    const int _N = {{ constant_or_scalar('N', variables['N']) }};
    long _count = 0;
    for (int _idx = 0; _idx < _N; _idx++) {
        const size_t _vectorisation_idx = _idx;
        {{ vector_code | autoindent }}
        if (_cond) {
            {{ _eventspace }}[_count++] = _idx;
            {% if _uses_refractory %}
            {{ not_refractory }}[_idx] = false;
            {{ lastspike }}[_idx] = {{ t }};
            {% endif %}
        }
    }
    {{ _eventspace }}[_N] = _count;
{% endblock %}

{% block after_code %}
    {% set _eventspace = get_array_name(eventspace_variable) %}
    {{ _eventspace }}[{{ constant_or_scalar('N', variables['N']) }}] = 0;
{% endblock %}
