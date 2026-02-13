{# State monitor template for cppyy backend #}
{# USES_VARIABLES { _clock_t, _indices, N } #}
{% extends 'common_group.cpp' %}

{% block maincode %}
    //// MAIN CODE ////////////
    const double _current_t = {{ _clock_t }};
    const int _num_indices = _num{{ _indices }};

    // Record time
    {{ _dynamic_t }}.push_back(_current_t);

    // Record state variables for each monitored index
    for (int _i = 0; _i < _num_indices; _i++) {
        const int _idx = {{ _indices }}[_i];
        const size_t _vectorisation_idx = _idx;

        {% for varname in record_variables %}
        // Record {{ varname }}
        {{ vector_code[varname] | autoindent }}
        {{ _dynamic_ ~ varname }}.push_back(_to_record_{{ varname }});
        {% endfor %}
    }
{% endblock %}
