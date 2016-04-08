{% extends 'common_group.cpp' %}

{% block maincode %}
    {#
    USES_VARIABLES { _synaptic_pre, _synaptic_post, _all_pre, _all_post, rand,
                     N_incoming, N_outgoing, N,
                     N_pre, N_post, _source_offset, _target_offset}
    #}
    {# WRITES_TO_READ_ONLY_VARIABLES { _synaptic_pre, _synaptic_post,
                      N_incoming, N_outgoing, N}
    #}
    srand((unsigned int)time(NULL));
    const int _buffer_size = 1024;
    int *const _prebuf = new int[_buffer_size];
    int *const _postbuf = new int[_buffer_size];
    int _curbuf = 0;
    const int _N_pre = {{constant_or_scalar('N_pre', variables['N_pre'])}};
    const int _N_post = {{constant_or_scalar('N_post', variables['N_post'])}};

    const int oldsize = {{_dynamic__synaptic_pre}}.size();

    // scalar code
	const int _vectorisation_idx = 1;
    {{scalar_code['setup_iterator']|autoindent}}
    {{scalar_code['create_j']|autoindent}}
    {{scalar_code['create_cond']|autoindent}}
    {{scalar_code['update_post']|autoindent}}
    for(int _i=0; _i<_num_all_pre; _i++)
    {
        bool __cond, _cond;
        {% if not postsynaptic_condition %}
        {
            {{vector_code['create_cond']|autoindent}}
            __cond = _cond;
        }
        _cond = __cond;
        if(!_cond) continue;
        {% endif %}
        // Some explanation of this hackery. The problem is that we have multiple code blocks.
        // Each code block is generated independently of the others, and they declare variables
        // at the beginning if necessary (including declaring them as const if their values don't
        // change). However, if two code blocks follow each other in the same C++ scope then
        // that causes a redeclaration error. So we solve it by putting each block inside a
        // pair of braces to create a new scope specific to each code block. However, that brings
        // up another problem: we need the values from these code blocks. I don't have a general
        // solution to this problem, but in the case of this particular template, we know which
        // values we need from them so we simply create outer scoped variables to copy the value
        // into. Later on we have a slightly more complicated problem because the original name
        // _j has to be used, so we create two variables __j, _j at the outer scope, copy
        // _j to __j in the inner scope (using the inner scope version of _j), and then
        // __j to _j in the outer scope (to the outer scope version of _j). This outer scope
        // version of _j will then be used in subsequent blocks.
        long _uiter_low;
        long _uiter_high;
        long _uiter_step;
        {% if iterator_func=='sample' %}
        double _uiter_p;
        {% endif %}
        {
            {{vector_code['setup_iterator']|autoindent}}
            _uiter_low = _iter_low;
            _uiter_high = _iter_high;
            _uiter_step = _iter_step;
            {% if iterator_func=='sample' %}
            _uiter_p = _iter_p;
            {% endif %}
        }
        {% if iterator_func=='range' %}
        for(int {{iteration_variable}}=_uiter_low; {{iteration_variable}}<_uiter_high; {{iteration_variable}}+=_uiter_step)
        {
        {% elif iterator_func=='sample' %}
        if(_uiter_p==0) continue;
        const bool _jump_algo = _uiter_p<0.25;
        double _log1p;
        if(_jump_algo)
            _log1p = log(1-_uiter_p);
        else
            _log1p = 1.0; // will be ignored
        const double _pconst = 1.0/_log1p;
        for(int {{iteration_variable}}=_uiter_low; {{iteration_variable}}<_uiter_high; {{iteration_variable}}++)
        {
            if(_jump_algo) {
                const double _r = _rand(_vectorisation_idx);
                if(_r==0.0) break;
                const int _jump = floor(log(_r)*_pconst)*_uiter_step;
                {{iteration_variable}} += _jump;
                if({{iteration_variable}}>=_uiter_high) continue;
            } else {
                if(_rand(_vectorisation_idx)>=_uiter_p) continue;
            }
        {% endif %}
            long __j, _j, _pre_idx, __pre_idx;
            {
                {{vector_code['create_j']|autoindent}}
                __j = _j; // pick up the locally scoped _j and store in __j
                __pre_idx = _pre_idx;
            }
            _j = __j; // make the previously locally scoped _j available
            _pre_idx = __pre_idx;
            if(_j<0 || _j>=_N_post)
            {
                {% if skip_if_invalid %}
                continue;
                {% else %}
                PyErr_SetString(PyExc_IndexError, "index j outside allowed range");
                throw 1;
                {% endif %}
            }
            {% if postsynaptic_condition %}
            {
                {{vector_code['create_cond']|autoindent}}
                __cond = _cond;
            }
            _cond = __cond;
            {% endif %}

            {% if if_expression!='True' and postsynaptic_condition %}
            if(!_cond) continue;
            {% endif %}

            {{vector_code['update_post']|autoindent}}

            for (int _repetition=0; _repetition<_n; _repetition++) {
                _prebuf[_curbuf] = (int)_pre_idx;
                _postbuf[_curbuf] = (int)_post_idx;
                _curbuf++;
                // Flush buffer
                if(_curbuf==_buffer_size)
                {
                    _flush_buffer(_prebuf, {{_dynamic__synaptic_pre}}, _curbuf);
                    _flush_buffer(_postbuf, {{_dynamic__synaptic_post}}, _curbuf);
                    _curbuf = 0;
                }
            }
        }
    }
    // Final buffer flush
    _flush_buffer(_prebuf, {{_dynamic__synaptic_pre}}, _curbuf);
    _flush_buffer(_postbuf, {{_dynamic__synaptic_post}}, _curbuf);

    const int newsize = {{_dynamic__synaptic_pre}}.size();
    // now we need to resize all registered variables and set the total number
    // of synapses (via Python)
    py::tuple _arg_tuple(1);
    _arg_tuple[0] = newsize;
    _owner.mcall("_resize", _arg_tuple);

    // And update N_incoming, N_outgoing and synapse_number
    _arg_tuple[0] = oldsize;
    _owner.mcall("_update_synapse_numbers", _arg_tuple);

    delete [] _prebuf;
    delete [] _postbuf;
{% endblock %}

{% block support_code_block %}
// Flush a buffered segment into a dynamic array
void _flush_buffer(int *buf, py::object &dynarr, int N)
{
    int _curlen = dynarr.attr("shape")[0];
    int _newlen = _curlen+N;
    // Resize the array
    py::tuple _newlen_tuple(1);
    _newlen_tuple[0] = _newlen;
    dynarr.mcall("resize", _newlen_tuple);
    // Get the potentially newly created underlying data arrays
    int *data = (int*)(((PyArrayObject*)(PyObject*)dynarr.attr("data"))->data);
    // Copy the values across
    for(int i=0; i<N; i++)
    {
        data[_curlen+i] = buf[i];
    }
}

{{ super() }}

{% endblock %}
