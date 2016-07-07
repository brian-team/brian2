{% extends 'common_group.cpp' %}

{% block maincode %}
{# USES_VARIABLES { _synaptic_pre, _synaptic_post, sources, targets, N,
                    N_pre, N_post, _source_offset, _target_offset }
#}
{# WRITES_TO_READ_ONLY_VARIABLES { _synaptic_pre, _synaptic_post, N_incoming, N_outgoing, N}
#}
py::tuple _arg_tuple_old(1);
py::tuple _arg_tuple_new(1);
const int _old_num_synapses = {{N}};
const int _new_num_synapses = _old_num_synapses + _numsources;
_arg_tuple_old[0] = _old_num_synapses;
_arg_tuple_new[0] = _new_num_synapses;

{{_dynamic__synaptic_pre}}.mcall("resize", _arg_tuple_new);
{{_dynamic__synaptic_post}}.mcall("resize", _arg_tuple_new);
// Get the potentially newly created underlying data arrays
int *_synaptic_pre_data = (int*)(((PyArrayObject*)(PyObject*){{_dynamic__synaptic_pre}}.attr("data"))->data);
int *_synaptic_post_data = (int*)(((PyArrayObject*)(PyObject*){{_dynamic__synaptic_post}}.attr("data"))->data);

for (int _idx=0; _idx<_numsources; _idx++) {
    {# After this code has been executed, the arrays _real_sources and
       _real_variables contain the final indices. Having any code here it all is
       only necessary for supporting subgroups #}
    {{vector_code|autoindent}}
    _synaptic_pre_data[_idx + _old_num_synapses] = _real_sources;
    _synaptic_post_data[_idx + _old_num_synapses] = _real_targets;
}

// now we need to resize all registered variables and set the total number of
// synapses (via Python)
_owner.mcall("_resize", _arg_tuple_new);

// And update N_incoming, N_outgoing and synapse_number
//_arg_tuple[0] = _old_num_synapses;
_owner.mcall("_update_synapse_numbers", _arg_tuple_old);
{% endblock %}