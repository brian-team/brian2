{% extends 'common_group.pyx' %}

{% block before_code %}
    _owner.initialise_queue()
{% endblock %}

{% block maincode %}
    # Optimized: avoid the push_spikes method call overhead
    cdef object eventspace = _owner.eventspace
    cdef object queue = _owner.queue
    cdef int spike_count

    # Get the spike count from the last element
    spike_count = eventspace[eventspace.shape[0] - 1]

    if spike_count > 0:
        # Extract events and push directly
        events = eventspace[:spike_count]
        queue.push(events)
{% endblock %}
