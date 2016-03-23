{% extends 'common.pyx' %}
{#
USES_VARIABLES { _synaptic_pre, _synaptic_post, _all_pre, _all_post, rand, N,
                 N_pre, N_post, _source_offset, _target_offset }
#}
{# WRITES_TO_READ_ONLY_VARIABLES { _synaptic_pre, _synaptic_post, N}
#}
# ITERATE_ALL { _idx }

######################## TEMPLATE SUPPORT CODE ##############################

{% block template_support_code %}
cdef int _buffer_size = 1024
cdef int[:] _prebuf = _numpy.zeros(_buffer_size, dtype=_numpy.int32)
cdef int[:] _postbuf = _numpy.zeros(_buffer_size, dtype=_numpy.int32)
cdef int _curbuf = 0

cdef void _flush_buffer(buf, dynarr, int buf_len):
    _curlen = dynarr.shape[0]
    _newlen = _curlen+buf_len
    # Resize the array
    dynarr.resize(_newlen)
    # Get the potentially newly created underlying data arrays
    data = dynarr.data
    data[_curlen:_curlen+buf_len] = buf[:buf_len]
    
{% endblock %}

######################## MAIN CODE ##############################

{% block maincode %}

    cdef int* _prebuf_ptr = &(_prebuf[0])
    cdef int* _postbuf_ptr = &(_postbuf[0])

    global _curbuf

    cdef int oldsize = len({{_dynamic__synaptic_pre}})
    cdef int newsize

    # The following variables are only used for probabilistic connections
    cdef int32_t[:] _all_j
    cdef int _num_targets
    cdef int _target

    # scalar code
    _vectorisation_idx = 1
    {{scalar_code|autoindent}}

    for _i in range(_num{{_all_pre}}):
        {% if sampling_algorithm != None %}
        _all_j = _sample_without_replacement(_num{{_all_post}}, {{p}})
        _num_targets = _all_j.shape[0]
        for _target in range(_all_j.shape[0]):
            _j = _all_j[_target]
        {% else %}
        for _j in range(_num{{_all_post}}):
        {% endif %}
            _vectorisation_idx = _j

            {# The abstract code consists of the following lines (the first two lines
            are there to properly support subgroups as sources/targets):
            _pre_idx = _all_pre
            _post_idx = _all_post
            _cond = {user-specified condition}
            _n = {user-specified number of synapses}
            _p = {user-specified probability}
            #}
            
            # vector code
            {{vector_code|autoindent}}
            
            # add to buffer
            if _cond:
                {% if sampling_algorithm == None %}
                if _p!=1.0:
                    if _rand(_vectorisation_idx)>=_p:
                        continue
                {% endif %}
                for _repetition in range(_n):
                    _prebuf_ptr[_curbuf] = _pre_idx
                    _postbuf_ptr[_curbuf] = _post_idx
                    _curbuf += 1
                    # Flush buffer
                    if _curbuf==_buffer_size:
                        _flush_buffer(_prebuf, {{_dynamic__synaptic_pre}}, _curbuf)
                        _flush_buffer(_postbuf, {{_dynamic__synaptic_post}}, _curbuf)
                        _curbuf = 0
                        
    # Final buffer flush
    _flush_buffer(_prebuf, {{_dynamic__synaptic_pre}}, _curbuf)
    _flush_buffer(_postbuf, {{_dynamic__synaptic_post}}, _curbuf)
    _curbuf = 0  # reset the buffer for the next run

    newsize = len({{_dynamic__synaptic_pre}})
    # now we need to resize all registered variables and set the total number
    # of synapse (via Python)
    _owner._resize(newsize)

    # And update N_incoming, N_outgoing and synapse_number
    _owner._update_synapse_numbers(oldsize)

{% endblock %}
