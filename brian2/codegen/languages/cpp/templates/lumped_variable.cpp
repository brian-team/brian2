////////////////////////////////////////////////////////////////////////////
//// MAIN CODE /////////////////////////////////////////////////////////////

{% macro main() %}

    // USE_SPECIFIERS { _synaptic_post, _synaptic_pre, _num_target_neurons }

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

	// Set all the target variable values to zero
	for (int _target_idx=0; _target_idx<_num_target_neurons; _target_idx++)
	    _ptr{{_target_var_array}}[_target_idx] = 0.0;

    // A bit confusing: The "neuron" index here refers to the synapses!
	for(int _neuron_idx=0; _neuron_idx<_num_synaptic_post; _neuron_idx++)
	{
	    const int _postsynaptic_idx = _synaptic_post[_neuron_idx];
	    const int _presynaptic_idx = _synaptic_pre[_neuron_idx];
		{% for line in code_lines %}
		{{line}}
		{% endfor %}
		_ptr{{_target_var_array}}[_postsynaptic_idx] += _synaptic_var;
	}
{% endmacro %}

////////////////////////////////////////////////////////////////////////////
//// SUPPORT CODE //////////////////////////////////////////////////////////

{% macro support_code() %}
	{% for line in support_code_lines %}
	{{line}}
	{% endfor %}
{% endmacro %}
