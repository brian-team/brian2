{% extends 'common_group.cpp' %}

{% block maincode %}
	{# USES_VARIABLES { rate, t, _spikespace, _clock_t, _clock_dt, _num_source_neurons } #}

	int _num_spikes = {{_spikespace}}[_num_spikespace-1];
	{{_dynamic_rate}}.push_back(1.0*_num_spikes/_clock_dt/_num_source_neurons);
	{{_dynamic_t}}.push_back(_clock_t);
{% endblock %}

{% block extra_functions_cpp %}
void _write_{{codeobj_name}}()
{
	ofstream outfile_t;
	outfile_t.open("results/{{codeobj_name}}_t", ios::binary | ios::out);
	if(outfile_t.is_open())
	{
		outfile_t.write(reinterpret_cast<char*>(&{{_dynamic_t}}[0]), {{_dynamic_t}}.size()*sizeof({{_dynamic_t}}[0]));
		outfile_t.close();
	} else
	{
		std::cout << "Error writing output file results/{{codeobj_name}}_t." << endl;
	}
	ofstream outfile_rate;
	outfile_rate.open("results/{{codeobj_name}}_rate", ios::binary | ios::out);
	if(outfile_rate.is_open())
	{
		outfile_rate.write(reinterpret_cast<char*>(&{{_dynamic_rate}}[0]), {{_dynamic_rate}}.size()*sizeof({{_dynamic_rate}}[0]));
		outfile_rate.close();
	} else
	{
		std::cout << "Error writing output file results/{{codeobj_name}}_rate." << endl;
	}
}
{% endblock %}

{% block extra_functions_h %}
void _write_{{codeobj_name}}();
{% endblock %}

{% macro main_finalise() %}
_write_{{codeobj_name}}();
{% endmacro %}
