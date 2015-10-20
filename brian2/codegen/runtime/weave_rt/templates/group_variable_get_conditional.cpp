{% extends 'common_group.cpp' %}
{# USES_VARIABLES { N } #}

{% block maincode %}
    //// MAIN CODE ////////////
    // This allows everything to work correctly for synapses where N is not a
    // constant
    const int _N = {{constant_or_scalar('N', variables['N'])}};

    {%set c_type = c_data_type(variables['_variable'].dtype) %}
    {%set numpy_dtype = dtype(variables['_variable'].dtype).char %}
    {%set numpy_type_int = dtype(variables['_variable'].dtype).num %}
    // {{c_type}} {{numpy_dtype}} {{numpy_type_int}}
    int _cpp_numelements = 0;
    // Container for all the potential values
    {{c_type}}* _elements = ({{c_type}}*)malloc(sizeof({{c_type}}) * _N);

    // scalar code
	const int _vectorisation_idx = 1;
	{{scalar_code|autoindent}}

    for(int _idx=0; _idx<_N; _idx++)
    {
        // vector code
        const int _vectorisation_idx = _idx;
        {{ super() }}
        if(_cond) {
            // _variable is set in the abstract code, see Group._get_with_code
            _elements[_cpp_numelements++] = _variable;
        }
    }
    npy_intp _dims[] = {_cpp_numelements};
    PyObject *_numpy_elements = PyArray_SimpleNewFromData(1, _dims, {{numpy_type_int}}, _elements);
    return_val = _numpy_elements;
{% endblock %}
