{# IS_OPENMP_COMPATIBLE #}
{% extends 'common_group.c' %}

{% block maincode %}
    {# USES_VARIABLES { t, _clock_dt, _clock_t, _indices } #}

    const int _timestep = (int)(_clock_t/_clock_dt + 0.5);
    {{t}}[_timestep] = _clock_t;

    {{ openmp_pragma('single') }}

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
            {{_recorded}}[_timestep*_num_indices + _i] = _to_record_{{varname}};
            {% endfor %}
        {% endblock %}
    }

{% endblock %}
