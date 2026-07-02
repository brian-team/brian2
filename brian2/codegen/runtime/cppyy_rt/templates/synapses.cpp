{# USES_VARIABLES { _queue_capsule } #}
{% extends 'common_group.cpp' %}

{% block template_support_code %}
#include <vector>
#include <cstdint>
{% endblock %}

{% block maincode %}
    // Extract the C++ spike queue from the capsule
    CSpikeQueue* _queue = _extract_spike_queue(_queue_capsule);

    // Peek at current timestep's spikes (synapse indices)
    std::vector<int32_t>* _spike_vector = _queue->peek();
    size_t _num_spikes = _spike_vector->size();

    if (_num_spikes == 0) {
        _queue->advance();
        return;
    }

    int32_t* _spike_data = &(*_spike_vector)[0];

    // Scalar code
    const size_t _vectorisation_idx = 1;
    {{ scalar_code | autoindent }}

    // Process each spike (synapse index)
    for (size_t _spike_idx = 0; _spike_idx < _num_spikes; _spike_idx++) {
        const int32_t _idx = _spike_data[_spike_idx];
        const size_t _vectorisation_idx = _idx;
        {{ vector_code | autoindent }}
    }

    // Advance the queue to the next timestep
    _queue->advance();
{% endblock %}
