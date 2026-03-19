{# Rate monitor template for cppyy backend #}
{# USES_VARIABLES { N, t, rate, _clock_t, _clock_dt, _spikespace,
                    _num_source_neurons, _source_start, _source_stop } #}
{% extends 'common_group.cpp' %}

{% block maincode %}
    size_t _num_spikes = {{ _spikespace }}[_num_spikespace - 1];

    // For subgroups, filter spikes to source range
    int _start_idx = _num_spikes;
    int _end_idx = _num_spikes;
    for (size_t _j = 0; _j < _num_spikes; _j++) {
        int _idx = {{ _spikespace }}[_j];
        if (_idx >= _source_start) {
            _start_idx = _j;
            break;
        }
    }
    if (_start_idx == (int)_num_spikes) {
        _start_idx = _num_spikes;
    }
    for (size_t _j = _start_idx; _j < _num_spikes; _j++) {
        int _idx = {{ _spikespace }}[_j];
        if (_idx >= _source_stop) {
            _end_idx = _j;
            break;
        }
    }
    _num_spikes = _end_idx - _start_idx;

    // Resize t and rate arrays via capsules
    {% set _t_capsule = "_dynamic_array_" + owner.name + "_t_capsule" %}
    {% set _rate_capsule = "_dynamic_array_" + owner.name + "_rate_capsule" %}
    auto* _dyn_t = _extract_dynamic_array_1d<double>({{ _t_capsule }});
    auto* _dyn_rate = _extract_dynamic_array_1d<double>({{ _rate_capsule }});

    size_t _current_len = _dyn_t->size();
    size_t _new_len = _current_len + 1;

    _dyn_t->resize(_new_len);
    _dyn_rate->resize(_new_len);

    // Update N
    {{ N }} = _new_len;

    // Write values
    _dyn_t->get_data_ptr()[_new_len - 1] = {{ _clock_t }};
    _dyn_rate->get_data_ptr()[_new_len - 1] = (double)_num_spikes / {{ _clock_dt }} / _num_source_neurons;
{% endblock %}
