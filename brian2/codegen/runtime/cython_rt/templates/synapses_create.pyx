{% extends 'common.pyx' %}
{#
USES_VARIABLES { _synaptic_pre, _synaptic_post, _all_pre, _all_post, rand,
                 N_incoming, N_outgoing, N,
                 N_pre, N_post, _source_offset, _target_offset }
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

    # Resize N_incoming and N_outgoing according to the size of the
    # source/target groups
    _var_N_incoming.resize(N_post + _target_offset)
    _var_N_outgoing.resize(N_pre + _source_offset)
    cdef {{cpp_dtype(variables['N_incoming'].dtype)}}[:] _N_incoming = {{_dynamic_N_incoming}}.data.view(_numpy.{{numpy_dtype(variables['N_incoming'].dtype)}})
    cdef {{cpp_dtype(variables['N_outgoing'].dtype)}}[:] _N_outgoing = {{_dynamic_N_outgoing}}.data.view(_numpy.{{numpy_dtype(variables['N_outgoing'].dtype)}})

    # scalar code
    _vectorisation_idx = 1
    {{scalar_code|autoindent}}

    for _i in range(_num{{_all_pre}}):
        for _j in range(_num{{_all_post}}):
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
                if _p!=1.0:
                    if _rand(_vectorisation_idx)>=_p:
                        continue
                for _repetition in range(_n):
                    _N_outgoing[_pre_idx] += 1
                    _N_incoming[_post_idx] += 1
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
    # now we need to resize all registered variables (via Python)
    _owner._resize(newsize)
    # Set the total number of synapses
    {{N}}[0] = newsize

{% endblock %}
