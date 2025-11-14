{# USES_VARIABLES { _queue_capsule } #}
{% extends 'common_group.pyx' %}

{% block template_support_code %}
from cpython.pycapsule cimport PyCapsule_GetPointer
# We declare minimal C++ interface here - only methods we actually want to use
# This avoids importing the full SpikeQueue wrapper and its dependencies
cdef extern from "spikequeue.h":
    cdef cppclass CSpikeQueue:
        void push(int32_t *, int)
{% endblock %}

{% block before_code %}
    _owner.initialise_queue()
{% endblock %}

{% block maincode %}
    {% set eventspace=get_array_name(eventspace_variable)%}
    cdef int spike_count
    # Get the spike count from the last entry in the spike buffer
    spike_count = {{eventspace}}[_num{{eventspace}}-1]

    # Recover the C++ spike queue object from the PyCapsule passed at runtime
    cdef object capsule = _queue_capsule
    cdef CSpikeQueue* cpp_queue = <CSpikeQueue*>PyCapsule_GetPointer(capsule, "CSpikeQueue")

    if spike_count > 0:
    # Optimized: avoid the push_spikes method call overhead
      cpp_queue.push({{eventspace}},spike_count)
{% endblock %}
