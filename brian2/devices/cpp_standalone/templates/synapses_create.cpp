{% extends 'common_synapses.cpp' %}

{% block maincode %}
	// USES_VARIABLES { _synaptic_pre, _synaptic_post, rand}
	{% if variables is defined %}
	{% set synpre = '_dynamic'+variables['_synaptic_pre'].arrayname %}
	{% set synpost = '_dynamic'+variables['_synaptic_post'].arrayname %}
	{% set synobj = owner.name %}
	{% endif %}
	int _synapse_idx = {{synpre}}.size();
	for(int i=0; i<_num_all_pre; i++)
	{
		for(int j=0; j<_num_all_post; j++)
		{
		    const int _vectorisation_idx = j;
			// Define the condition
			{% for line in code_lines %}
			{{line}}
			{% endfor %}
			// Add to buffer
			if(_cond)
			{
			    if (_p != 1.0) {
			        // We have to use _rand instead of rand to use our rand
			        // function, not the one from the C standard library
			        if (_rand(_vectorisation_idx) >= _p)
			            continue;
			    }

			    for (int _repetition=0; _repetition<_n; _repetition++) {
			    	{{synpre}}.push_back(_pre_idcs);
			    	{{synpost}}.push_back(_post_idcs);
                    _synapse_idx++;
                }
			}
		}
	}

	// now we need to resize all registered variables
	{% if owner is defined %}
	const int newsize = _dynamic{{owner.variables['_synaptic_pre'].arrayname}}.size();
	{% for variable in owner._registered_variables %}
	_dynamic{{variable.arrayname}}.resize(newsize);
	{% endfor %}
	// Also update the total number of synapses
	{{owner.name}}._N = newsize;
	{% endif %}
{% endblock %}
