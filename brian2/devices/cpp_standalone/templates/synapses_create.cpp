{% extends 'common_synapses.cpp' %}

{% block maincode %}
    #include<iostream>
	{# USES_VARIABLES { _synaptic_pre, _synaptic_post, rand,
	                    N_incoming, N_outgoing, N,
	                    N_pre, N_post, _source_offset, _target_offset } #}

    {# WRITES_TO_READ_ONLY_VARIABLES { _synaptic_pre, _synaptic_post,
                                       N_incoming, N_outgoing, N}
    #}

    {# Get N_post and N_pre in the correct way, regardless of whether they are
    constants or scalar arrays#}
    const int _N_pre = {{constant_or_scalar('N_pre', variables['N_pre'])}};
    const int _N_post = {{constant_or_scalar('N_post', variables['N_post'])}};
    {{_dynamic_N_incoming}}.resize(_N_post + _target_offset);
    {{_dynamic_N_outgoing}}.resize(_N_pre + _source_offset);

    // scalar code
    const int _vectorisation_idx = -1;
	{{scalar_code|autoindent}}
    for(int _i=0; _i<_num_all_pre; _i++)
	{
		for(int _j=0; _j<_num_all_post; _j++)
		{
		    const int _vectorisation_idx = _j;
	        {# The abstract code consists of the following lines (the first two lines
	        are there to properly support subgroups as sources/targets):
	        _pre_idx = _all_pre
	        _post_idx = _all_post
	        _cond = {user-specified condition}
	        _n = {user-specified number of synapses}
	        _p = {user-specified probability}
	        #}
			{{vector_code|autoindent}}
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
			        {{_dynamic_N_outgoing}}[_pre_idx] += 1;
			        {{_dynamic_N_incoming}}[_post_idx] += 1;
			    	{{_dynamic__synaptic_pre}}.push_back(_pre_idx);
			    	{{_dynamic__synaptic_post}}.push_back(_post_idx);
                }
			}
		}
	}

	// now we need to resize all registered variables
	const int32_t newsize = {{_dynamic__synaptic_pre}}.size();
	{% for variable in owner._registered_variables | sort(attribute='name') %}
	{% set varname = get_array_name(variable, access_data=False) %}
	{{varname}}.resize(newsize);
	{% endfor %}
	// Also update the total number of synapses
	{{N}} = newsize;

    {% if multisynaptic_index %}
    // Update the "synapse number" (number of synapses for the same
    // source-target pair)
    std::map<std::pair<int32_t, int32_t>, int32_t> source_target_count;
    for (int _i=0; _i<newsize; _i++)
    {
        // Note that source_target_count will create a new entry initialized
        // with 0 when the key does not exist yet
        const std::pair<int32_t, int32_t> source_target = std::pair<int32_t, int32_t>({{_dynamic__synaptic_pre}}[_i], {{_dynamic__synaptic_post}}[_i]);
        {{get_array_name(variables[multisynaptic_index], access_data=False)}}[_i] = source_target_count[source_target];
        source_target_count[source_target]++;
    }
    {% endif %}
{% endblock %}
