{% extends 'common.pyx' %}

{# USES_VARIABLES { N } #}

{% block maincode %}

    {# This is copied from the weave equivalent, seems to work but confusing #}
    {%set c_type = c_data_type(variables['_variable'].dtype) %}
    {%set numpy_dtype = dtype(variables['_variable'].dtype).char %}

    _vectorisation_idx = 1

    cdef int _num_elements = 0
    cdef _numpy.ndarray[{{c_type}}, ndim=1, mode='c'] _elements = _numpy.zeros(N, dtype='{{numpy_dtype}}')
    cdef {{c_type}}[:] _elements_view = _elements
    
    {{scalar_code|autoindent}}

    for _idx in range(N):
        _vectorisation_idx = _idx
        
        {{vector_code|autoindent}}
        
        if _cond:
            _elements_view[_num_elements] = _variable
            _num_elements += 1
    
    return _elements[:_num_elements]

{% endblock %}
