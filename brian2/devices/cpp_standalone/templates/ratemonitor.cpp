{% extends 'common_group.cpp' %}

{% block maincode %}
	{# USES_VARIABLES { rate, t, _spikespace, _clock_t, _clock_dt, _num_source_neurons } #}

	int _num_spikes = {{_spikespace}}[_num_{{_spikespace}}-1];
	int _num_source_neurons = _num_{{_spikespace}}-1;
	{{_dynamic_rate}}.push_back(1.0*_num_spikes/_clock_dt/_num_source_neurons);
	{{_dynamic_t}}.push_back(_clock_t);
{% endblock %}

{% block extra_functions_cpp %}
void _write_{{codeobj_name}}()
{
	ofstream outfile;
	outfile.open("results/{{codeobj_name}}.txt", ios::out);
	if(outfile.is_open())
	{
		for(int s=0; s<{{_dynamic_t}}.size(); s++)
		{
			outfile << {{_dynamic_t}}[s] << ", " << {{dynamic_rate}}[s] << endl;
		}
		outfile.close();
	} else
	{
		std::cout << "Error writing output file." << endl;
	}
}
{% endblock %}

{% block extra_functions_h %}
void _write_{{codeobj_name}}();
{% endblock %}

{% macro main_finalise() %}
_write_{{codeobj_name}}();
{% endmacro %}
