{% extends 'common_group.pyx' %}

{% block template_support_code %}
# We declare minimal C++ interface here - only methods we actually want to use
# This avoids importing the full SpikeQueue wrapper and its dependencies
cdef extern from "cspikequeue.cpp":
    cdef cppclass CSpikeQueue:
        void push(int32_t *, int)

# POINTER RESURRECTION: Now we convert memory address back to usable C++ object
# So as we added {{queue_ptr}} to template namespace, it gets template-substituted with actual memory address (e.g., 105553147185984)
# Now we ensure proper pointer alignment and type safety to use the pointer:

# Step 1: Cast raw integer (from Python) to a generic void pointer.
# Direct casting to specific C++ types is unsafe from Python objects,
# so we first cast to void* to ensure proper pointer alignment.
cdef void* _void_ptr = <void*>{{queue_ptr}}
# Step 2: Now we cast it to specific C++ class pointer ...
cdef CSpikeQueue* _queue_ptr = <CSpikeQueue*>_void_ptr

{% endblock %}

{% block before_code %}
    _owner.initialise_queue()
{% endblock %}

{% block maincode %}
    {% set eventspace=get_array_name(eventspace_variable)%}

    cdef int spike_count
    # Get the spike count from the last element
    spike_count = {{eventspace}}[_num{{eventspace}}-1]


    if spike_count > 0:
    # Optimized: avoid the push_spikes method call overhead
      _queue_ptr.push({{eventspace}},spike_count)
{% endblock %}
