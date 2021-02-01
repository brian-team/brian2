{# USES_VARIABLES { _n_sources, _n_targets, delay, _source_dt} #}

{% extends 'common_group.cpp' %}

{% block before_code %}
    {% set scalar = c_data_type(variables['delay'].dtype) %}
    std::vector<{{scalar}}> &real_delays = {{get_array_name(variables['delay'], access_data=False)}};
    {{scalar}}* real_delays_data = real_delays.empty() ? 0 : &(real_delays[0]);
    int32_t* sources = {{owner.name}}.sources.empty() ? 0 : &({{owner.name}}.sources[0]);
    const size_t n_delays = real_delays.size();
    const size_t n_synapses = {{owner.name}}.sources.size();
    {{owner.name}}.prepare({{constant_or_scalar('_n_sources', variables['_n_sources'])}},
                           {{constant_or_scalar('_n_targets', variables['_n_targets'])}},
                           real_delays_data, n_delays, sources,
                           n_synapses,
                           {{_source_dt}});
{% endblock %}

{% block maincode %}
    // we do advance at the beginning rather than at the end because it saves us making
    // a copy of the current spiking synapses
    {#  Get the name of the array that stores these events (e.g. the spikespace array) #}
    {% set _eventspace = get_array_name(eventspace_variable) %}
    {{openmp_pragma('parallel')}}
    {
        {{owner.name}}.advance();
        {{owner.name}}.push({{_eventspace}}, {{_eventspace}}[_num{{eventspace_variable.name}}-1]);
    }
{% endblock %}
