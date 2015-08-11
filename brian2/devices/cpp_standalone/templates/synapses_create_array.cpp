{# IS_OPENMP_COMPATIBLE #}
{% extends 'common_group.cpp' %}

{% block maincode %}
{# USES_VARIABLES { _synaptic_pre, _synaptic_post, sources, targets
                    N_incoming, N_outgoing, N,
                    N_pre, N_post, _source_offset, _target_offset }
#}

const int _old_num_synapses = {{N}}[0];
const int _new_num_synapses = _old_num_synapses + _numsources;

{{_dynamic_N_incoming}}.resize(N_post + _target_offset);
{{_dynamic_N_outgoing}}.resize(N_pre + _source_offset);

for (int _idx=0; _idx<_numsources; _idx++) {
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
const int newsize = {{_dynamic__synaptic_pre}}.size();
{% for variable in owner._registered_variables | sort(attribute='name') %}
{% set varname = get_array_name(variable, access_data=False) %}
{{varname}}.resize(newsize);
{% endfor %}
// Also update the total number of synapses
{{N}}[0] = newsize;
{% endblock %}