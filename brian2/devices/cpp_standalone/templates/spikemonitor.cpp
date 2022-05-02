{# USES_VARIABLES { N, _clock_t, count} #}
{# WRITES_TO_READ_ONLY_VARIABLES { N, count } #}
{% extends 'common_group.cpp' %}

{% block maincode %}
    //// MAIN CODE ////////////
    {#  Get the name of the array that stores these events (e.g. the spikespace array) #}
    {% set _eventspace = get_array_name(eventspace_variable) %}

    int32_t _num_events = {{_eventspace}}[_num{{eventspace_variable.name}}-1];
    {% if subgroup and not contiguous %}
    // We use the same data structure as for the eventspace to store the
    // "filtered" events, i.e. the events that are indexed in the subgroup
    int32_t _filtered_events[{{source_N}} + 1];
    _filtered_events[{{source_N}}] = 0;
    size_t _source_index_counter = 0;
    {% endif %}
    {% if subgroup %}
    // For subgroups, we do not want to record all spikes
    size_t _start_idx = _num_events;
    size_t _end_idx = _num_events;
    if (_num_events > 0)
    {
        {% if contiguous %}
        for(size_t _j=0; _j<_num_events; _j++)
        {
            const int _idx = {{_eventspace}}[_j];
            if (_idx >= _source_start) {
                _start_idx = _j;
                break;
            }
        }
        for(size_t _j=_num_events-1; _j>=_start_idx; _j--)
        {
            const int _idx = {{_eventspace}}[_j];
            if (_idx < _source_stop) {
                break;
            }
            _end_idx = _j;
        }
        _num_events = _end_idx - _start_idx;
        {% else %}
        const size_t _max_source_index = {{_source_indices}}[{{source_N}}-1];
        for (size_t _j=0; _j<_num_events; _j++)
        {
            const size_t _idx = {{_eventspace}}[_j];
            if (_idx < {{_source_indices}}[_source_index_counter])
                continue;
            if (_idx > _max_source_index)
                break;
            while ({{_source_indices}}[_source_index_counter] < _idx)
            {
                _source_index_counter++;
            }
            if (_source_index_counter < {{source_N}} &&
                _idx == {{_source_indices}}[_source_index_counter])
            {
                _source_index_counter += 1;
                _filtered_events[_filtered_events[{{source_N}}]++] = _idx;
                if (_source_index_counter == {{source_N}})
                    break;
            }
            if (_source_index_counter == {{source_N}})
                break;
        }
        _num_events = _filtered_events[{{source_N}}];
        {% endif %}
    }
    {% endif %}
    if (_num_events > 0) {
        const size_t _vectorisation_idx = 1;
        {{scalar_code|autoindent}}
        {% if subgroup %}
        {% if contiguous %}
        for(size_t _j=_start_idx; _j<_end_idx; _j++)
        {
            const size_t _idx = {{_eventspace}}[_j];
            const size_t _vectorisation_idx = _idx;
            {{vector_code|autoindent}}
            {% for varname, var in record_variables | dictsort %}
            {{get_array_name(var, access_data=False)}}.push_back(_to_record_{{varname}});
            {% endfor %}
            {{count}}[_idx-_source_start]++;
        }
        {% else %}
        for(size_t _j=0; _j < _num_events; _j++)
        {
            const size_t _idx = _filtered_events[_j];
            const size_t _vectorisation_idx = _idx;
            {{vector_code|autoindent}}
            {% for varname, var in record_variables | dictsort %}
            {{get_array_name(var, access_data=False)}}.push_back(_to_record_{{varname}});
            {% endfor %}
            {{count}}[_to_record_i]++;
        }
        {% endif %}
        {% else %}
        for (size_t _j=0; _j < _num_events; _j++)
        {
            const size_t _idx = {{_eventspace}}[_j];
            const size_t _vectorisation_idx = _idx;
            {{ vector_code|autoindent }}
            {% for varname, var in record_variables | dictsort %}
            {{get_array_name(var, access_data=False)}}.push_back(_to_record_{{varname}});
            {% endfor %}
            {{count}}[_idx]++;
        }
        {% endif %}
        {{N}} += _num_events;
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
#ifdef DEBUG
_debugmsg_{{codeobj_name}}();
#endif
{% endmacro %}
