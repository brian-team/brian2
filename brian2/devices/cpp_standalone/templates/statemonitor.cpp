{# IS_OPENMP_COMPATIBLE #}
{% extends 'common_group.cpp' %}

{% block maincode %}
    {# USES_VARIABLES { t, _clock_t, _indices } #}

    {{ openmp_pragma('single') }}
    {{_dynamic_t}}.push_back(_clock_t);

    const int _new_size = {{_dynamic_t}}.size();
    // Resize the dynamic arrays
    {% for varname, var in _recorded_variables | dictsort %}
    {% set _recorded =  get_array_name(var, access_data=False) %}

    {{ openmp_pragma('single') }}
    {{_recorded}}.resize(_new_size, _num_indices);
    {% endfor %}

    // scalar code
	const int _vectorisation_idx = -1;
	{{scalar_code|autoindent}}

    {{ openmp_pragma('static') }}
    for (int _i = 0; _i < _num_indices; _i++)
    {
        // vector code
        const int _idx = {{_indices}}[_i];
        const int _vectorisation_idx = _idx;
        {% block maincode_inner %}
            {{ super() }}

            {% for varname, var in _recorded_variables | dictsort %}
            {% set _recorded =  get_array_name(var, access_data=False) %}
            {{_recorded}}(_new_size-1, _i) = _to_record_{{varname}};
            {% endfor %}
        {% endblock %}
    }

{% endblock %}
