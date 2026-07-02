{# Get indices matching a condition template for cppyy backend.
 #
 # Because cppyy functions are extern "C" void (no return value), we can't
 # return the array directly like Cython does.  Instead, the generator adds
 # two extra output parameters to the function signature:
 #
 #   int* _return_values_buf   -- pre-allocated buffer of size N (filled here)
 #   int* _return_values_n     -- 1-element array; C++ writes the match count
 #
 # These are injected by:
 #   CppyyCodeGenerator.determine_keywords()  -- adds them to function_params
 #   CppyyCodeObject.variables_to_namespace() -- allocates the numpy arrays
 #   CppyyCodeObject._build_param_mapping()   -- mirrors the two extra entries
 #   CppyyCodeObject.run_block()              -- slices and returns the result
 #}
{# USES_VARIABLES { N, _indices } #}
{% extends 'common_group.cpp' %}

{% block maincode %}
    const size_t _vectorisation_idx = 1;

    {{ scalar_code | autoindent }}

    const int _N = {{ constant_or_scalar('N', variables['N']) }};
    int _num_matches = 0;
    for (int _idx = 0; _idx < _N; _idx++) {
        const size_t _vectorisation_idx = _idx;

        {{ vector_code | autoindent }}

        if (_cond) {
            _return_values_buf[_num_matches++] = _idx;
        }
    }
    _return_values_n[0] = _num_matches;
{% endblock %}
