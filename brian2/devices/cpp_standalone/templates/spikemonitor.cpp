{% extends 'common_group.cpp' %}

{% block maincode %}
	//// MAIN CODE ////////////
    {# USES_VARIABLES { N, _clock_t, count,
                        _source_start, _source_stop} #}
    {#  Get the name of the array that stores these events (e.g. the spikespace array) #}
    {% set _eventspace = get_array_name(eventspace_variable) %}

	int32_t _num_events = {{_eventspace}}[_num{{eventspace_variable.name}}-1];

    if (_num_events > 0)
    {
        int _start_idx = _num_events;
        int _end_idx = _num_events;
        for(int _j=0; _j<_num_events; _j++)
        {
            const int _idx = {{_eventspace}}[_j];
            if (_idx >= _source_start) {
                _start_idx = _j;
                break;
            }
        }
        for(int _j=_start_idx; _j<_num_events; _j++)
        {
            const int _idx = {{_eventspace}}[_j];
            if (_idx >= _source_stop) {
                _end_idx = _j;
                break;
            }
        }
        _num_events = _end_idx - _start_idx;
        if (_num_events > 0) {
             const int _vectorisation_idx = 1;
            {{scalar_code|autoindent}}
            for(int _j=_start_idx; _j<_end_idx; _j++)
            {
                const int _idx = {{_eventspace}}[_j];
                const int _vectorisation_idx = _idx;
                {{vector_code|autoindent}}
                {% for varname, var in record_variables.items() %}
                {{get_array_name(var, access_data=False)}}.push_back(_to_record_{{varname}});
                {% endfor %}
                {{count}}[_idx-_source_start]++;
            }
            {{N}} += _num_events;
        }
    }

{% endblock %}

{% block extra_functions_cpp %}
void _debugmsg_{{codeobj_name}}()
{
	using namespace brian;
	{# We need the pointers and constants here to get the access to N working #}
    %CONSTANTS%
    {{pointers_lines|autoindent}}
	std::cout << "Number of spikes: " << {{N}} << endl;
}
{% endblock %}

{% block extra_functions_h %}
void _debugmsg_{{codeobj_name}}();
{% endblock %}

{% macro main_finalise() %}
_debugmsg_{{codeobj_name}}();
{% endmacro %}
