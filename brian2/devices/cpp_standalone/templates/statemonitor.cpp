{% extends 'common_group.cpp' %}

{% if variables is defined %}
{% set _t = '_dynamic'+variables['_t'].arrayname %}
{% set _indices = variables['_indices'].arrayname %}
{% endif %}

{% block maincode %}
    // USES_VARIABLES { _t, _indices }

    {{_t}}.push_back(t);

    const int _new_size = {{_t}}.size();
    // Resize the dynamic arrays
    {% for _varname in _variable_names %}
    {% set _recorded = '_dynamic_2d' + variables['_recorded_'+_varname].arrayname %}
    {{_recorded}}.resize(_new_size, _num_indices);
    {% endfor %}

    for (int _i = 0; _i < _num_indices; _i++)
    {
        const int _idx = {{_indices}}[_i];
        const int _vectorisation_idx = _idx;
        {% block maincode_inner %}
            {{ super() }}

            {% for _varname in _variable_names %}
            {% set _recorded = '_dynamic_2d' + variables['_recorded_'+_varname].arrayname %}
            {{_recorded}}(_new_size-1, _i) = _to_record_{{_varname}};
            {% endfor %}
        {% endblock %}
    }

{% endblock %}

{% block extra_functions_cpp %}
void _write_{{codeobj_name}}()
{
	ofstream outfile;
	outfile.open("results/{{codeobj_name}}_t", ios::binary | ios::out);
	if(outfile.is_open())
	{
		outfile.write(reinterpret_cast<char*>(&{{_t}}[0]), {{_t}}.size()*sizeof({{_t}}[0]));
		outfile.close();
	} else
	{
		cout << "Error writing output file." << endl;
	}

	{% for _varname in _variable_names %}
	{
	    {%set _recorded = '_dynamic_2d' + variables['_recorded_'+_varname].arrayname %}
	    const int _num_indices = {{_recorded}}.m;
        const int _num_times = {{_t}}.size();
        outfile.open("results/{{codeobj_name}}_{{_varname}}", ios::binary | ios::out);
        if(outfile.is_open())
        {
            for (int s=0; s<_num_times; s++)
            {
            	outfile.write(reinterpret_cast<char*>(&{{_recorded}}(s, 0)), _num_indices*sizeof({{_recorded}}(0, 0)));
            }
            outfile.close();
        } else
        {
            cout << "Error writing output file." << endl;
        }
	}
	{% endfor %}
}

{% endblock %}

{% block extra_functions_h %}
void _write_{{codeobj_name}}();
{% endblock %}

{% macro main_finalise() %}
_write_{{codeobj_name}}();
{% endmacro %}
