{# USES_VARIABLES { _synaptic_pre, _synaptic_post, rand, N,
                 N_pre, N_post, _source_offset, _target_offset } #}
{# WRITES_TO_READ_ONLY_VARIABLES { _synaptic_pre, _synaptic_post, N} #}
{# ITERATE_ALL { _idx } #}
{% extends 'common_group.pyx' %}

######################## TEMPLATE SUPPORT CODE ##############################

{% block template_support_code %}
cdef int _buffer_size = 1024
cdef int[:] _prebuf = _numpy.zeros(_buffer_size, dtype=_numpy.int32)
cdef int[:] _postbuf = _numpy.zeros(_buffer_size, dtype=_numpy.int32)
cdef int _curbuf = 0
cdef int _raw_pre_idx
cdef int _raw_post_idx

cdef void _flush_buffer(buf, dynarr, int buf_len):
    cdef size_t _curlen = dynarr.shape[0]
    cdef size_t _newlen = _curlen+buf_len
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

    cdef size_t oldsize = len({{_dynamic__synaptic_pre}})
    cdef size_t newsize

    # The following variables are only used for probabilistic connections
    {% if iterator_func=='sample' %}
    cdef int _iter_sign
    {% if iterator_kwds['sample_size'] == 'fixed' %}
    cdef bool _selection_algo
    cdef set[int] _selected_set = set[int]()
    cdef set[int].iterator _selected_it
    cdef int _n_selected
    cdef int _n_dealt_with
    cdef int _n_total
    cdef double _U
    {% else %}
    cdef bool _jump_algo
    cdef double _log1p, _pconst
    cdef size_t _jump
    {% endif %}
    {% endif %}

    {# For a connect call j='k+i for k in range(0, N_post, 2) if k+i < N_post'
    "j" is called the "result index" (and "_post_idx" the "result index array", etc.)
    "i" is called the "outer index" (and "_pre_idx" the "outer index array", etc.)
    "k" is called the inner variable #}

    # scalar code
    _vectorisation_idx = 1
    {{scalar_code['setup_iterator']|autoindent}}
    {{scalar_code['generator_expr']|autoindent}}
    {{scalar_code['create_cond']|autoindent}}
    {{scalar_code['update']|autoindent}}

    for _{{outer_index}} in range({{outer_index_size}}):
        _raw{{outer_index_array}} = _{{outer_index}} + {{outer_index_offset}}

        {% if not result_index_condition %}
        {{vector_code['create_cond']|autoindent}}
        if not _cond:
            continue
        {% endif %}
        {{vector_code['setup_iterator']|autoindent}}
        {% if iterator_func=='range' %}
        for {{inner_variable}} in range(_iter_low, _iter_high, _iter_step):
        {% elif iterator_func=='sample' %}
        {% if iterator_kwds['sample_size'] == 'fixed' %}
        # Note that the following code is written in a slightly convoluted way,
        # but we have to plug it together with the following code that checks
        # for the fulfillment of the condition.
        _n_selected = 0
        _n_dealt_with = 0
        with _cython.cdivision(True):
            if _iter_step > 0:
                _n_total = (_iter_high - _iter_low - 1) // _iter_step + 1
            else:
                _n_total = (_iter_low - _iter_high - 1) // -_iter_step + 1
            # Value determined by benchmarking, see github PR #1280
            _selection_algo = 1.0*_iter_size / _n_total > 0.06
        if _iter_size > _n_total:
            {% if skip_if_invalid %}
            _iter_size = _n_total
            {% else %}
            raise IndexError(f"Requested sample size {_iter_size} is bigger than the "
                             f"population size {_n_total}.")
            {% endif %}
        elif _iter_size < 0:
            {% if skip_if_invalid %}
            continue
            {% else %}
            raise IndexError(f"Requested sample size {_iter_size} is negative.")
            {% endif %}
        if _selection_algo:
            {{inner_variable}} = _iter_low - _iter_step
        else:
            # For the tracking algorithm, we have to first create all values
            # to make sure they will be iterated in sorted order
            _selected_set.clear()
            while _n_selected < _iter_size:
                _r = <int> (_rand(_vectorisation_idx) * _n_total)
                while not _selected_set.insert(_r).second:  # .second will be False if duplicate
                    _r = <int> (_rand(_vectorisation_idx) * _n_total)
                _n_selected += 1
            _n_selected = 0
            _selected_it = _selected_set.begin()

        while _n_selected < _iter_size:
            if _selection_algo:
                {{inner_variable}} += _iter_step
                # Selection sampling technique
                # See section 3.4.2 of Donald E. Knuth, AOCP, Vol 2,
                # Seminumerical Algorithms
                _n_dealt_with += 1
                _U = _rand(_vectorisation_idx)
                if (_n_total - _n_dealt_with) * _U >= _iter_size - _n_selected:
                    continue
            else:
                {{inner_variable}} = _iter_low + _deref(_selected_it)*_iter_step
                _preinc(_selected_it)
            _n_selected += 1

        {% else %}
        if _iter_p==0:
            continue
        if _iter_step < 0:
            _iter_sign = -1
        else:
            _iter_sign = 1
        _jump_algo = _iter_p<0.25
        if _jump_algo:
            _log1p = log(1-_iter_p)
        else:
            _log1p = 1.0 # will be ignored
        _pconst = 1.0/_log1p
        {{inner_variable}} = _iter_low-_iter_step
        while _iter_sign*({{inner_variable}} + _iter_step) < _iter_sign*_iter_high:
            {{inner_variable}} += _iter_step
            if _jump_algo:
                _jump = <int>(log(_rand(_vectorisation_idx))*_pconst)*_iter_step
                {{inner_variable}} += _jump
                if _iter_sign*{{inner_variable}} >= _iter_sign*_iter_high:
                    break
            else:
                if _rand(_vectorisation_idx)>=_iter_p:
                    continue
        {% endif %}
        {% endif %}

            {{vector_code['generator_expr']|autoindent}}
            _raw{{result_index_array}} = _{{result_index}} + {{result_index_offset}}

            {% if result_index_condition %}
            {% if result_index_used %}
            {# The condition could index outside of array range #}
            if _{{result_index}}<0 or _{{result_index}}>={{result_index_size}}:
                {% if skip_if_invalid %}
                continue
                {% else %}
                # Note that with Jinja using a lot of curly braces, it is a better
                # solution to use the outdated % syntax instead of f-strings here.
                raise IndexError("index {{result_index}}=%d outside allowed range from 0 to %d" % (_{{result_index}}, {{result_index_size}}-1))
                {% endif %}
            {% endif %}
            {{vector_code['create_cond']|autoindent}}
            {% endif %}
            {% if if_expression!='True' and result_index_condition %}
            if not _cond:
                continue
            {% endif %}
            {% if not result_index_used %}
            {# Otherwise, we already checked before #}
            if _{{result_index}}<0 or _{{result_index}}>={{result_index_size}}:
                {% if skip_if_invalid %}
                continue
                {% else %}
                raise IndexError("index j=%d outside allowed range from 0 to %d" % (_{{result_index}}, {{result_index_size}}-1))
                {% endif %}
            {% endif %}
            {{vector_code['update']|autoindent}}

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
