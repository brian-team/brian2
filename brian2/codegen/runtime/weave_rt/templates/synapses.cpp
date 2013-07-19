////////////////////////////////////////////////////////////////////////////
//// MAIN CODE /////////////////////////////////////////////////////////////

{% macro main() %}
	// USE_SPECIFIERS { _spiking_synapses, _synaptic_pre,_synaptic_post }

    //// SUPPORT CODE //////////////////////////////////////////////////////////
	{% for line in support_code_lines %}
	// {{line}}
	{% endfor %}

	////// HANDLE DENORMALS ///
	{% for line in denormals_code_lines %}
	{{line}}
	{% endfor %}

	////// HASH DEFINES ///////
	{% for line in hashdefine_lines %}
	{{line}}
	{% endfor %}

	///// POINTERS ////////////
	{% for line in pointers_lines %}
	{{line}}
	{% endfor %}

	//// MAIN CODE ////////////
	for(int _spiking_synapse_idx=0;
		_spiking_synapse_idx<_num_spiking_synapses;
		_spiking_synapse_idx++)
	{
		const int _neuron_idx = _spiking_synapses[_spiking_synapse_idx];
		const int _postsynaptic_idx = _synaptic_post[_neuron_idx];
		const int _presynaptic_idx = _synaptic_pre[_neuron_idx];
		const int _vectorisation_idx = _neuron_idx;
		{% for line in code_lines %}
		{{line}}
		{% endfor %}
	}
{% endmacro %}

////////////////////////////////////////////////////////////////////////////
//// SUPPORT CODE //////////////////////////////////////////////////////////

{% macro support_code() %}
	{% for line in support_code_lines %}
	{{line}}
	{% endfor %}
{% endmacro %}
