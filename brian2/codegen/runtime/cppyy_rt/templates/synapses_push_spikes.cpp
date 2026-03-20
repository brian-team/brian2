{# USES_VARIABLES { _queue_capsule } #}
{% extends 'common_group.cpp' %}

{% block template_support_code %}
#include <vector>
#include <cstdint>
{% endblock %}

{% block before_code %}
    // Queue initialization happens in Python (_owner.initialise_queue())
    // This is handled by the code object's before_run mechanism
{% endblock %}

{% block maincode %}
    {% set eventspace = get_array_name(eventspace_variable) %}

    // Get the spike count from the last entry in the spike buffer
    int32_t _spike_count = {{ eventspace }}[_num{{ eventspace_variable.name }} - 1];

    // Extract the C++ spike queue from the capsule
    CSpikeQueue* _queue = _extract_spike_queue(_queue_capsule);

    if (_spike_count > 0) {
        _queue->push({{ eventspace }}, _spike_count);
    }
{% endblock %}
