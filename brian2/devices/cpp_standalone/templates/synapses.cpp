{% extends 'common_synapses.cpp' %}

{% block maincode %}
    // This is only needed for the _debugmsg function below
    {# USES_VARIABLES { _synaptic_pre } #}
	std::vector<int32_t> *_spiking_synapses = {{pathway.name}}.queue->peek();

	// scalar code
	const int _vectorisation_idx = -1;
	{{scalar_code|autoindent}}

	const unsigned int _num_spiking_synapses = _spiking_synapses->size();
	for(unsigned int _spiking_synapse_idx=0;
		_spiking_synapse_idx<_num_spiking_synapses;
		_spiking_synapse_idx++)
	{
	    // vector code
		const int32_t _idx = (*_spiking_synapses)[_spiking_synapse_idx];
		const int32_t _vectorisation_idx = _idx;
		{{vector_code|autoindent}}
	}
{% endblock %}

{% block extra_functions_cpp %}
void _debugmsg_{{codeobj_name}}()
{
	using namespace brian;
	std::cout << "Number of synapses: " << {{_dynamic__synaptic_pre}}.size() << endl;
}
{% endblock %}

{% block extra_functions_h %}
void _debugmsg_{{codeobj_name}}();
{% endblock %}

{% macro main_finalise() %}
_debugmsg_{{codeobj_name}}();
{% endmacro %}
