{% extends 'common_group.cpp' %}

{% block maincode %}
    {# USES_VARIABLES { _queue } #}
    //// MAIN CODE ////////////
    // Get the spikes
    const PyArrayObject *_spiking_synapses_obj = (PyArrayObject *)PyObject_CallMethod(_queue, "peek", "");
    const npy_int32 *_spiking_synapses = (npy_int32 *)_spiking_synapses_obj->data;
    const int _num_spiking_synapses = _spiking_synapses_obj->dimensions[0];

    // scalar code
    const int _vectorisation_idx = 1;
    {{scalar_code|autoindent}}

    for(int _spiking_synapse_idx=0;
        _spiking_synapse_idx<_num_spiking_synapses;
        _spiking_synapse_idx++)
    {
        // vector code
        const int _idx = _spiking_synapses[_spiking_synapse_idx];
        const int _vectorisation_idx = _idx;
        {{vector_code|autoindent}}
    }

    // Advance the spike queue
    PyObject_CallMethod(_queue, "advance", "");
{% endblock %}
