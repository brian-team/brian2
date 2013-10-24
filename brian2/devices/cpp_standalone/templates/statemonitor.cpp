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
    {{_recorded}}.resize(_num_indices, _new_size);
    {% endfor %}

    for (int _i = 0; _i < _num_indices; _i++)
    {
        const int _idx = {{_indices}}[_i];
        const int _vectorisation_idx = _idx;
        {% block maincode_inner %}
            {{ super() }}

            {% for _varname in _variable_names %}
            {% set _recorded = '_dynamic_2d' + variables['_recorded_'+_varname].arrayname %}
            {{_recorded}}(_i, _new_size-1) = _to_record_{{_varname}};
            {% endfor %}
        {% endblock %}
    }

{% endblock %}

{% block extra_functions_cpp %}
void _write_{{codeobj_name}}()
{
	ofstream outfile;
	outfile.open("results/{{codeobj_name}}_t.txt", ios::out);
	if(outfile.is_open())
	{
		for(int s=0; s<{{_t}}.size(); s++)
		{
			outfile << {{_t}}[s] << endl;
		}
		outfile.close();
	} else
	{
		cout << "Error writing output file." << endl;
	}

	{% for _varname in _variable_names %}
	{
	    {%set _recorded = '_dynamic_2d' + variables['_recorded_'+_varname].arrayname %}
	    const int _num_indices = {{_recorded}}.n;
        const int _num_times = {{_t}}.size();
        outfile.open("results/{{codeobj_name}}_{{_varname}}.txt", ios::out);
        if(outfile.is_open())
        {
            for (int s=0; s<_num_times; s++)
            {
                for (int _i = 0; _i < _num_indices; _i++)
                {
                	outfile << {{_recorded}}(_i, s) << " ";
                }
                outfile << endl;
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
