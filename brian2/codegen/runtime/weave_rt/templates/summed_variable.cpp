{% extends 'common_group.cpp' %}

{% block maincode %}
    // USES_VARIABLES { _synaptic_post, _num_target_neurons }
	//// MAIN CODE ////////////
	// Set all the target variable values to zero
	for (int _target_idx=0; _target_idx<_num_target_neurons; _target_idx++)
	    _ptr{{_target_var_array}}[_target_idx] = 0.0;

	for(int _idx=0; _idx<_num_synaptic_post; _idx++)
	{
	    {{ super() }}
		_ptr{{_target_var_array}}[_postsynaptic_idx] += _synaptic_var;
	}
{% endblock %}
