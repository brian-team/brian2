{# USES_VARIABLES { t, _clock_t, _indices, N } #}
{# WRITES_TO_READ_ONLY_VARIABLES { t, N } #}
{% extends 'common_group.cpp' %}

{% block maincode %}
    {{_dynamic_t}}.push_back({{_clock_t}});

    const size_t _new_size = {{_dynamic_t}}.size();
    // Resize the dynamic arrays
    {% for varname, var in _recorded_variables | dictsort %}
    {% set _recorded =  get_array_name(var, access_data=False) %}
    {{_recorded}}.resize(_new_size, _num_indices);
    {% endfor %}

    // scalar code
    const size_t _vectorisation_idx = -1;
    {{scalar_code|autoindent}}

    {{ openmp_pragma('parallel-static') }}
    for (int _i = 0; _i < (int)_num_indices; _i++)
    {
        // vector code
        const size_t _idx = {{_indices}}[_i];
        const size_t _vectorisation_idx = _idx;
        {{vector_code|autoindent}}

        {% for varname, var in _recorded_variables | dictsort %}
        {% set _recorded =  get_array_name(var, access_data=False) %}
        {% if c_data_type(var.dtype) == 'bool' %}
        {{ openmp_pragma('critical') }}
        { // std::vector<bool> is not threadsafe
        {{_recorded}}(_new_size-1, _i) = _to_record_{{varname}};
        }
        {% else %}
        {{_recorded}}(_new_size-1, _i) = _to_record_{{varname}};
        {% endif %}
        {% endfor %}
    }

    {{N}} = _new_size;

{% endblock %}
