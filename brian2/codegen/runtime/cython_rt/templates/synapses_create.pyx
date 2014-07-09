{% extends 'common.pyx' %}
{#
USES_VARIABLES { _synaptic_pre, _synaptic_post, _all_pre, _all_post, rand,
                 N_incoming, N_outgoing }
#}
# ITERATE_ALL { _idx }

######################## TEMPLATE SUPPORT CODE ##############################

{% block template_support_code %}

cdef int _buffer_size = 1024

_prebuf = _numpy.zeros(_buffer_size, dtype=int)
_postbuf = _numpy.zeros(_buffer_size, dtype=int)
cdef int _curbuf = 0

cdef void _flush_buffer(buf, dynarr, N):
    _curlen = dynarr.shape[0]
    _newlen = _curlen+N
    # Resize the array
    dynarr.resize(_newlen)
    # Get the potentially newly created underlying data arrays
    data = dynarr.data
    for i in range(N):
        data[_curlen+i] = buf[i]

{% endblock %}

######################## MAIN CODE ##############################

{% block maincode %}

    global _curbuf
    
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
                    if rand(_vectorisation_idx)>=_p:
                    #if rand(1)>=_p:
                        continue
                    for _repetition in range(_n):
                        {{N_outgoing}}[_pre_idx] += 1
                        {{N_incoming}}[_post_idx] += 1
                        _prebuf[_curbuf] = _pre_idx
                        _postbuf[_curbuf] = _post_idx
                        _curbuf += 1
                        # Flush buffer
                        if _curbuf==_buffer_size:
                            _flush_buffer(_prebuf, {{_dynamic__synaptic_pre}}, _curbuf)
                            _flush_buffer(_postbuf, {{_dynamic__synaptic_post}}, _curbuf)
                            _curbuf = 0
                        
        # Final buffer flush
        _flush_buffer(_prebuf, {{_dynamic__synaptic_pre}}, _curbuf)
        _flush_buffer(_postbuf, {{_dynamic__synaptic_post}}, _curbuf)
    
        newsize = len({{_dynamic__synaptic_pre}})
        # now we need to resize all registered variables (via Python)
        _owner._resize(newsize)

{% endblock %}
