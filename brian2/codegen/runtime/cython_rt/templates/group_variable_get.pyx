{# Note that we use this template only for subexpressions -- for normal arrays
   we do not generate any code but simply access the data in the underlying
   array directly. See RuntimeDevice.get_with_array #}

{% extends 'common.pyx' %}

{# USES_VARIABLES { _group_idx } #}

{% block maincode %}

    {# This is copied from the weave equivalent, seems to work but confusing #}
    {%set c_type = c_data_type(variables['_variable'].dtype) %}
    {%set numpy_dtype = dtype(variables['_variable'].dtype).char %}

    _vectorisation_idx = 1
    
    cdef int _num_elements = 0
    cdef _numpy.ndarray[{{c_type}}, ndim=1, mode='c'] _elements = _numpy.zeros(_num{{_group_idx}}, dtype='{{numpy_dtype}}')
    cdef {{c_type}}[:] _elements_view = _elements
    
    {{scalar_code|autoindent}}

    for _idx_group_idx in range(_num{{_group_idx}}):
        _idx = {{_group_idx}}[_idx_group_idx]
        _vectorisation_idx = _idx
        
        {{vector_code|autoindent}}
        
        _elements_view[_idx_group_idx] = _variable
    
    return _elements

{% endblock %}
