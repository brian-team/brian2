{% extends 'common_group.cpp' %}

{% block maincode %}
{# USES_VARIABLES { _synaptic_pre, _synaptic_post, sources, targets
                    N_incoming, N_outgoing, N,
                    N_pre, N_post, _source_offset, _target_offset }
#}

py::tuple _newlen_tuple(1);
const int _old_num_synapses = {{N}}[0];
const int _new_num_synapses = _old_num_synapses + _numsources;

_newlen_tuple[0] = _new_num_synapses;
{{_dynamic__synaptic_pre}}.mcall("resize", _newlen_tuple);
{{_dynamic__synaptic_post}}.mcall("resize", _newlen_tuple);
// Get the potentially newly created underlying data arrays
int *_synaptic_pre_data = (int*)(((PyArrayObject*)(PyObject*){{_dynamic__synaptic_pre}}.attr("data"))->data);
int *_synaptic_post_data = (int*)(((PyArrayObject*)(PyObject*){{_dynamic__synaptic_post}}.attr("data"))->data);
// Resize N_incoming and N_outgoing according to the size of the
// source/target groups
PyObject_CallMethod(_var_N_incoming, "resize", "i", N_post + _target_offset);
PyObject_CallMethod(_var_N_outgoing, "resize", "i", N_pre + _source_offset);
int *_N_incoming = (int *)(((PyArrayObject*)(PyObject*){{_dynamic_N_incoming}}.attr("data"))->data);
int *_N_outgoing = (int *)(((PyArrayObject*)(PyObject*){{_dynamic_N_outgoing}}.attr("data"))->data);

for (int _idx=0; _idx<_numsources; _idx++) {
    {# After this code has been executed, the arrays _real_sources and
       _real_variables contain the final indices. Having any code here it all is
       only necessary for supporting subgroups #}
    {{vector_code|autoindent}}
    _synaptic_pre_data[_idx + _old_num_synapses] = _real_sources;
    _synaptic_post_data[_idx + _old_num_synapses] = _real_targets;
    // Update the number of total outgoing/incoming synapses per source/target neuron
    _N_outgoing[_real_sources]++;
    _N_incoming[_real_targets]++;
}

// now we need to resize all registered variables (via Python)
_owner.mcall("_resize", _newlen_tuple);
{% endblock %}