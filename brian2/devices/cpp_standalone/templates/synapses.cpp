{% extends 'common_synapses.cpp' %}

{% set _non_synaptic = [] %}
{% for var in variables %}
    {% if variable_indices[var] not in ('_idx', '0') %}
        {# This is a trick to get around the scoping problem #}
        {% if _non_synaptic.append(1) %}{% endif %}
    {% endif %}
{% endfor %}

{% block maincode %}

	// This is only needed for the _debugmsg function below	
	{# USES_VARIABLES { _synaptic_pre } #}	
	
	// scalar code
	const int _vectorisation_idx = -1;
	{{scalar_code|autoindent}}

	{{ openmp_pragma('parallel') }}
	{
	std::vector<int> *_spiking_synapses = {{pathway.name}}.peek();
	const unsigned int _num_spiking_synapses = _spiking_synapses->size();

	{% if _non_synaptic %}
	{{ openmp_pragma('single') }}
	{
		for(unsigned int _spiking_synapse_idx=0;
			_spiking_synapse_idx<_num_spiking_synapses;
			_spiking_synapse_idx++)
		{
			const int _idx = (*_spiking_synapses)[_spiking_synapse_idx];
			const int _vectorisation_idx = _idx;
			{{vector_code|autoindent}}
		}
	}
	{% else %}
	{{ openmp_pragma('static') }}
	for(int _spiking_synapse_idx=0;
		_spiking_synapse_idx<_num_spiking_synapses;
		_spiking_synapse_idx++)
	{
		const int _idx = (*_spiking_synapses)[_spiking_synapse_idx];
		const int _vectorisation_idx = _idx;
		{{vector_code|autoindent}}
	}

	{% endif %}
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
