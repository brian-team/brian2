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
    cdef bool _jump_algo
    cdef double _log1p, _pconst
    cdef int _jump

    # scalar code
    _vectorisation_idx = 1
    {{scalar_code['setup_iterator']|autoindent}}
    {{scalar_code['create_j']|autoindent}}
    {{scalar_code['create_cond']|autoindent}}
    {{scalar_code['update_post']|autoindent}}

    for _i in range(_num{{_all_pre}}):
        {% if not postsynaptic_condition %}
        {{vector_code['create_cond']|autoindent}}
        if not _cond:
            continue
        {% endif %}
        {{vector_code['setup_iterator']|autoindent}}
        {% if iterator_func=='range' %}
        for {{iteration_variable}} in range(_iter_low, _iter_high, _iter_step):
        {% elif iterator_func=='sample' %}
        if _iter_p==0:
            continue
        _jump_algo = _iter_p<0.25
        if _jump_algo:
            _log1p = log(1-_iter_p)
        else:
            _log1p = 1.0 # will be ignored
        _pconst = 1.0/_log1p
        {{iteration_variable}} = _iter_low-1
        while {{iteration_variable}}+1<_iter_high:
            {{iteration_variable}} += 1
            if _jump_algo:
                _jump = int(floor(log(_rand(_vectorisation_idx))*_pconst))*_iter_step
                {{iteration_variable}} += _jump
                if {{iteration_variable}}>=_iter_high:
                    break
            else:
                if _rand(_vectorisation_idx)>=_iter_p:
                    continue
        {% endif %}

            {{vector_code['create_j']|autoindent}}
            if _j<0 or _j>=N_post:
                {% if skip_if_invalid %}
                continue
                {% else %}
                raise IndexError("index j=%d outside allowed range from 0 to %d" % (_j, N_post-1))
                {% endif %}
            {% if postsynaptic_condition %}
            {{vector_code['create_cond']|autoindent}}
            {% endif %}
            {% if if_expression!='True' and postsynaptic_condition %}
            if not _cond:
                continue
            {% endif %}
            {{vector_code['update_post']|autoindent}}

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
