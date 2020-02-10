{% extends 'common_group.cpp' %}

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

    {% if profiled %}
    // Profiling
    {% if openmp_pragma('with_openmp') %}
    const double _run_time = omp_get_wtime() -_start_time;
    {% else %}
    const double _run_time = (double)(std::clock() -_start_time)/CLOCKS_PER_SEC;
    {% endif %}
    {{codeobj_name}}_profiling_info += _run_time;
    {% endif %}
{% endblock %}
