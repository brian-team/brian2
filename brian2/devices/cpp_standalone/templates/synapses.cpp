{% extends 'common_synapses.cpp' %}

{% block maincode %}
	{% if pathway is defined %}
	vector<int> *_spiking_synapses = {{pathway.name}}.queue->peek();
	const int _num_spiking_synapses = _spiking_synapses->size();
	{% endif %}
	for(int _spiking_synapse_idx=0;
		_spiking_synapse_idx<_num_spiking_synapses;
		_spiking_synapse_idx++)
	{
		const int _idx = (*_spiking_synapses)[_spiking_synapse_idx];
		const int _vectorisation_idx = _idx;
		{% for line in code_lines %}
		{{line}}
		{% endfor %}
	}
{% endblock %}

{% block extra_functions_cpp %}
void _debugmsg_{{codeobj_name}}()
{
	{% if owner is defined %}
	cout << "Number of synapses: " << _dynamic_array_{{owner.name}}__synaptic_pre.size() << endl;
	{% endif %}
}
{% endblock %}

{% block extra_functions_h %}
void _debugmsg_{{codeobj_name}}();
{% endblock %}

{% macro main_finalise() %}
_debugmsg_{{codeobj_name}}();
{% endmacro %}
