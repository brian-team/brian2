{% extends 'common_group.cpp' %}

{% block maincode %}
    {# USES_VARIABLES { _t, _indices } #}

    {{_dynamic__t}}.push_back(t);

    const int _new_size = {{_dynamic__t}}.size();
    // Resize the dynamic arrays
    {% for var in _recorded_variables.values() %}
    {% set _recorded =  get_array_name(var, access_data=False) %}
    {{_recorded}}.resize(_new_size, _num_indices);
    {% endfor %}

    for (int _i = 0; _i < _num_indices; _i++)
    {
        const int _idx = {{_indices}}[_i];
        const int _vectorisation_idx = _idx;
        {% block maincode_inner %}
            {{ super() }}

            {% for varname, var in _recorded_variables.items() %}
            {% set _recorded =  get_array_name(var, access_data=False) %}
            {{_recorded}}(_new_size-1, _i) = _to_record_{{varname}};
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
		outfile.write(reinterpret_cast<char*>(&{{_dynamic__t}}[0]), {{_dynamic__t}}.size()*sizeof({{_dynamic__t}}[0]));
		outfile.close();
	} else
	{
		std::cout << "Error writing output file." << endl;
	}

	{% for varname, var in _recorded_variables.items() %}
	{
	    {% set _recorded =  get_array_name(var, access_data=False) %}
	    const int _num_indices = {{_recorded}}.m;
        const int _num_times = {{_dynamic__t}}.size();
        outfile.open("results/{{codeobj_name}}_{{varname}}", ios::binary | ios::out);
        if(outfile.is_open())
        {
            for (int s=0; s<_num_times; s++)
            {
            	outfile.write(reinterpret_cast<char*>(&{{_recorded}}(s, 0)), _num_indices*sizeof({{_recorded}}(0, 0)));
            }
            outfile.close();
        } else
        {
            std::cout << "Error writing output file." << endl;
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
