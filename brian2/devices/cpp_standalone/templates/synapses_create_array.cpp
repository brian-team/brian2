{# USES_VARIABLES { _synaptic_pre, _synaptic_post, sources, targets,
                    N_incoming, N_outgoing, N,
                    N_pre, N_post, _source_offset, _target_offset }
#}
{# WRITES_TO_READ_ONLY_VARIABLES { _synaptic_pre, _synaptic_post,
                                   N_incoming, N_outgoing, N}
#}
{% extends 'common_group.cpp' %}

{% block maincode %}

const size_t _old_num_synapses = {{N}};
const size_t _new_num_synapses = _old_num_synapses + _numsources;

{# Get N_post and N_pre in the correct way, regardless of whether they are
constants or scalar arrays#}
const size_t _N_pre = {{constant_or_scalar('N_pre', variables['N_pre'])}};
const size_t _N_post = {{constant_or_scalar('N_post', variables['N_post'])}};
{{_dynamic_N_incoming}}.resize(_N_post + _target_offset);
{{_dynamic_N_outgoing}}.resize(_N_pre + _source_offset);

for (size_t _idx=0; _idx<_numsources; _idx++) {
    {# After this code has been executed, the arrays _real_sources and
       _real_variables contain the final indices. Having any code here it all is
       only necessary for supporting subgroups #}
    {{vector_code|autoindent}}

    {{_dynamic__synaptic_pre}}.push_back(_real_sources);
    {{_dynamic__synaptic_post}}.push_back(_real_targets);
    // Update the number of total outgoing/incoming synapses per source/target neuron
    {{_dynamic_N_outgoing}}[_real_sources]++;
    {{_dynamic_N_incoming}}[_real_targets]++;
}

// now we need to resize all registered variables
const size_t newsize = {{_dynamic__synaptic_pre}}.size();
{% for varname in owner._registered_variables | variables_to_array_names(access_data=False) | sort %}
{{varname}}.resize(newsize);
{% endfor %}
// Also update the total number of synapses
{{N}} = newsize;

{% if multisynaptic_index %}
// Update the "synapse number" (number of synapses for the same
// source-target pair)
std::map<std::pair<int32_t, int32_t>, int32_t> source_target_count;
for (size_t _i=0; _i<newsize; _i++)
{
    // Note that source_target_count will create a new entry initialized
    // with 0 when the key does not exist yet
    const std::pair<int32_t, int32_t> source_target = std::pair<int32_t, int32_t>({{_dynamic__synaptic_pre}}[_i], {{_dynamic__synaptic_post}}[_i]);
    {{get_array_name(variables[multisynaptic_index], access_data=False)}}[_i] = source_target_count[source_target];
    source_target_count[source_target]++;
}
{% endif %}
{% endblock %}
