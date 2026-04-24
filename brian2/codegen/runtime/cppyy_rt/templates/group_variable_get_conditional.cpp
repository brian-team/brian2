{# USES_VARIABLES { N } #}
{% extends 'common_group.cpp' %}

{% block maincode %}
    const size_t _vectorisation_idx = -1;
    {{ scalar_code | autoindent }}

    const int _N = {{ constant_or_scalar('N', variables['N']) }};
    int _n_out = 0;
    for (int _idx = 0; _idx < _N; _idx++) {
        const size_t _vectorisation_idx = _idx;
        {{ vector_code | autoindent }}
        if (_cond) {
            _output_buf[_n_out++] = _variable;
        }
    }
    _output_n[0] = _n_out;
{% endblock %}
