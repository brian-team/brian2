{# Rate monitor template for cppyy backend #}
{# USES_VARIABLES { _clock_t, _spikespace, _rate, _t, _num_source_neurons } #}
{% extends 'common_group.cpp' %}

{% block maincode %}
    //// MAIN CODE ////////////
    {% set _eventspace = get_array_name(eventspace_variable) %}

    const double _current_t = {{ _clock_t }};
    const int _num_spikes = {{ _eventspace }}[_num{{ _eventspace }} - 1];
    const double _dt = {{ dt }};
    const int _source_neurons = {{ _num_source_neurons }};

    // Calculate instantaneous firing rate
    const double _current_rate = (double)_num_spikes / (_source_neurons * _dt);

    // Append to dynamic arrays
    {{ _dynamic_t }}.push_back(_current_t);
    {{ _dynamic_rate }}.push_back(_current_rate);
{% endblock %}
