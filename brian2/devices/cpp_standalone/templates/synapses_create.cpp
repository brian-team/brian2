{% extends 'common_synapses.cpp' %}

{% block maincode %}
    #include<iostream>
	// USES_VARIABLES { _synaptic_pre, _synaptic_post, rand}
	int _synapse_idx = {{_dynamic__synaptic_pre}}.size();
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
			    	{{_dynamic__synaptic_pre}}.push_back(_pre_idcs);
			    	{{_dynamic__synaptic_post}}.push_back(_post_idcs);
                    _synapse_idx++;
                }
			}
		}
	}

	// now we need to resize all registered variables
	const int newsize = {{_dynamic__synaptic_pre}}.size();
	{% for variable in owner._registered_variables %}
	{% set varname = get_array_name(variable, access_data=False) %}
	{{varname}}.resize(newsize);
	{% endfor %}
	// Also update the total number of synapses
	{{owner.name}}._N = newsize;
{% endblock %}
