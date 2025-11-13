{# USES_VARIABLES { _queue_capsule } #}
{% extends 'common_group.pyx' %}

{% block template_support_code %}
from cpython.pycapsule cimport PyCapsule_GetPointer
from libcpp.vector cimport vector
from cython.operator cimport dereference
# We declare minimal C++ interface here - only methods we actually want to use
# This avoids importing the full SpikeQueue wrapper and its dependencies
cdef extern from "cspikequeue.h":
    cdef cppclass CSpikeQueue:
        vector[int32_t]* peek();
        void advance();
{% endblock %}

{% block maincode %}
     # Extract the raw C++ object pointer from the Python capsule.
    cdef object capsule = _queue_capsule
    cdef CSpikeQueue* cpp_queue = <CSpikeQueue*>PyCapsule_GetPointer(capsule, "CSpikeQueue")

    # Now we call the C++ peek method directly to get the current spike vector.
    # This returns a pointer to std::vector<int32_t> containing synapse IDs
    # that are ready for processing in the current time step.
    cdef vector[int32_t]* spike_vector = cpp_queue.peek()
    cdef size_t num_spikes = dereference(spike_vector).size()

    # Early exit for empty queue - avoid all processing overhead
    if num_spikes == 0:
        cpp_queue.advance()
        return

    # Access the underlying raw data pointer of the vector
    cdef int32_t* spike_data = &dereference(spike_vector)[0]

    # scalar code
    _vectorisation_idx = 1
    {{ scalar_code | autoindent }}


    cdef size_t i = 0
    cdef int32_t synapse_id
    while i < num_spikes:
        synapse_id = spike_data[i]

        _idx = synapse_id
        _vectorisation_idx = _idx

        {{ vector_code | autoindent }}

        i += 1

    # Move the queue forward to the next time step
    cpp_queue.advance()

{% endblock %}
