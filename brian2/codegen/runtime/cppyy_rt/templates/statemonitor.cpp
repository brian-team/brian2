{# USES_VARIABLES { t, _clock_t, _indices, N } #}
{# WRITES_TO_READ_ONLY_VARIABLES { t, N } #}
{% extends 'common_group.cpp' %}

{% block maincode %}
    // ── Extract DynamicArray objects from capsules ──
    // These are the SAME C++ objects that Cython created. The capsule
    // holds a void* to the DynamicArray1D<double> that the RuntimeDevice
    // allocated. We cast it back to the correct type and can call resize(),
    // get_data_ptr(), etc. — all in C++, no Python overhead.

    {% set _t_capsule = "_dynamic_array_" + owner.name + "_t_capsule" %}
    auto* _dyn_t = _extract_dynamic_array_1d<double>({{ _t_capsule }});

    // Get current size and compute new size
    size_t _old_len = _dyn_t->size();
    size_t _new_len = _old_len + 1;

    // Resize the time array — this may reallocate the underlying buffer
    _dyn_t->resize(_new_len);

    // Write the current clock time into the last element
    _dyn_t->get_data_ptr()[_new_len - 1] = {{ _clock_t }};

    // ── Resize each recorded variable's 2D array ──
    {% for varname, var in _recorded_variables | dictsort %}
    {% set _rec_capsule = get_array_name(var, access_data=False) + "_capsule" %}
    {% set _rec_ctype = c_data_type(var.dtype) %}
    {
        auto* _dyn_{{ varname }} = _extract_dynamic_array_2d<{{ _rec_ctype }}>({{ _rec_capsule }});
        _dyn_{{ varname }}->resize_along_first(_new_len);
    }
    {% endfor %}

    // ── Scalar code (runs once) ──
    const size_t _vectorisation_idx = -1;
    {{ scalar_code | autoindent }}


    for (int _i = 0; _i < _num_indices; _i++) {
        const size_t _idx = {{ _indices }}[_i];
        const size_t _vectorisation_idx = _idx;
        {{ vector_code | autoindent }}

        // Write recorded values into the last row of each 2D array.
        // After resize, get_data_ptr() returns the (potentially new) buffer,
        // and we index using stride * row + col to handle over-allocation.
        {% for varname, var in _recorded_variables | dictsort %}
        {% set _rec_capsule = get_array_name(var, access_data=False) + "_capsule" %}
        {% set _rec_ctype = c_data_type(var.dtype) %}
        {
            auto* _dyn = _extract_dynamic_array_2d<{{ _rec_ctype }}>({{ _rec_capsule }});
            _dyn->operator()(_new_len - 1, _i) = _to_record_{{ varname }};
        }
        {% endfor %}
    }

    // Update N (the number of recorded timesteps)
    {{ N }} = _new_len;

{% endblock %}
