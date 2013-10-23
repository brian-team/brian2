{% extends 'common_group.cpp' %}

{% if variables is defined %}
{% set _spikespace = variables['_spikespace'].arrayname %}
{% set _rate = '_dynamic'+variables['_rate'].arrayname %}
{% set _t = '_dynamic'+variables['_t'].arrayname %}
{% endif %}

{% block maincode %}
	// { USES_VARIABLES _rate, _t, _spikespace, t, dt }

	int _num_spikes = {{_spikespace}}[_num_{{_spikespace}}-1];
	int _num_source_neurons = _num_{{_spikespace}}-1;
	{{_rate}}.push_back(1.0*_num_spikes/dt/_num_source_neurons);
	{{_t}}.push_back(t);
{% endblock %}

{% block extra_functions_cpp %}
void _write_{{codeobj_name}}()
{
	ofstream outfile;
	outfile.open("results/{{codeobj_name}}.txt", ios::out);
	if(outfile.is_open())
	{
		for(int s=0; s<{{_t}}.size(); s++)
		{
			outfile << {{_t}}[s] << ", " << {{_rate}}[s] << endl;
		}
		outfile.close();
	} else
	{
		cout << "Error writing output file." << endl;
	}
}
{% endblock %}

{% block extra_functions_h %}
void _write_{{codeobj_name}}();
{% endblock %}

{% macro main_finalise() %}
_write_{{codeobj_name}}();
{% endmacro %}
