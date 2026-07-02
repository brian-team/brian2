{# USES_VARIABLES { _synaptic_pre, _synaptic_post, sources, targets, N,
                    N_pre, N_post, _source_offset, _target_offset }
#}
{# WRITES_TO_READ_ONLY_VARIABLES { _synaptic_pre, _synaptic_post, N} #}
{% extends 'common_group.cpp' %}

{% block maincode %}
    {% set _pre_capsule = get_array_name(variables['_synaptic_pre'], access_data=False) + "_capsule" %}
    {% set _post_capsule = get_array_name(variables['_synaptic_post'], access_data=False) + "_capsule" %}

    size_t _old_num_synapses = {{ N }};
    size_t _new_num_synapses = _old_num_synapses + _numsources;

    // Resize pre/post synapse index arrays via capsules
    auto* _dyn_pre = _extract_dynamic_array_1d<int32_t>({{ _pre_capsule }});
    auto* _dyn_post = _extract_dynamic_array_1d<int32_t>({{ _post_capsule }});

    _dyn_pre->resize(_new_num_synapses);
    _dyn_post->resize(_new_num_synapses);

    int32_t* _synaptic_pre_data = _dyn_pre->get_data_ptr();
    int32_t* _synaptic_post_data = _dyn_post->get_data_ptr();

    for (size_t _idx = 0; _idx < _numsources; _idx++) {
        {{ vector_code | autoindent }}
        _synaptic_pre_data[_idx + _old_num_synapses] = _real_sources;
        _synaptic_post_data[_idx + _old_num_synapses] = _real_targets;
    }

    // Python-side resize of all registered variables and update N
    // (handled by _owner._resize and _owner._update_synapse_numbers
    // which are called from the code object's after_run or via Python)
{% endblock %}

{% block after_code %}
    // This is intentionally empty — the Python-side resize is handled
    // by the code object wrapper calling _owner._resize() after run.
    // EMPTY_CODE_BLOCK
{% endblock %}
