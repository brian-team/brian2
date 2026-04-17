{# USES_VARIABLES { _synaptic_pre, _synaptic_post, rand, N,
                 N_pre, N_post, _source_offset, _target_offset } #}
{# WRITES_TO_READ_ONLY_VARIABLES { _synaptic_pre, _synaptic_post, N} #}
{# ITERATE_ALL { _idx } #}
{% extends 'common_group.cpp' %}

{% block template_support_code %}
#include <cstring>

const int _buffer_size = 1024;

inline void _flush_buffer(int32_t* buf, DynamicArray1D<int32_t>* dynarr, int buf_len) {
    size_t _curlen = dynarr->size();
    dynarr->resize(_curlen + buf_len);
    memcpy(dynarr->get_data_ptr() + _curlen, buf, buf_len * sizeof(int32_t));
}
{% endblock %}

{% block maincode %}
    {% set _pre_capsule = get_array_name(variables['_synaptic_pre'], access_data=False) + "_capsule" %}
    {% set _post_capsule = get_array_name(variables['_synaptic_post'], access_data=False) + "_capsule" %}

    auto* _dyn_pre = _extract_dynamic_array_1d<int32_t>({{ _pre_capsule }});
    auto* _dyn_post = _extract_dynamic_array_1d<int32_t>({{ _post_capsule }});

    int32_t _prebuf[1024];
    int32_t _postbuf[1024];
    int _curbuf = 0;

    // scalar code
    const size_t _vectorisation_idx = 1;
    {{scalar_code['setup_iterator']|autoindent}}
    {{scalar_code['generator_expr']|autoindent}}
    {{scalar_code['create_cond']|autoindent}}
    {{scalar_code['update']|autoindent}}

    const int _N_outer = {{ constant_or_scalar(outer_index_size, variables[outer_index_size]) }};
    const int _N_result = {{ constant_or_scalar(result_index_size, variables[result_index_size]) }};
    for (int _{{outer_index}} = 0; _{{outer_index}} < _N_outer; _{{outer_index}}++) {
        int _raw{{outer_index_array}} = _{{outer_index}} + {{outer_index_offset}};

        {% if not result_index_condition %}
        {{vector_code['create_cond']|autoindent}}
        if (!_cond) continue;
        {% endif %}
        {{vector_code['setup_iterator']|autoindent}}
        {% if iterator_func=='range' %}
        for (int {{inner_variable}} = _iter_low; {{inner_variable}} < _iter_high; {{inner_variable}} += _iter_step) {
        {% elif iterator_func=='sample' %}
        {% if iterator_kwds['sample_size'] == 'fixed' %}
        {
            // Fixed-size sample: use selection sampling (Knuth AOCP Vol 2 3.4.2)
            int _n_selected = 0;
            int _n_dealt_with = 0;
            int _n_total;
            if (_iter_step > 0)
                _n_total = (_iter_high - _iter_low - 1) / _iter_step + 1;
            else
                _n_total = (_iter_low - _iter_high - 1) / (-_iter_step) + 1;

            if (_iter_size > _n_total) {
                {% if skip_if_invalid %}
                _iter_size = _n_total;
                {% else %}
                // Error case — but we continue
                _iter_size = _n_total;
                {% endif %}
            }
            if (_iter_size < 0) {
                {% if skip_if_invalid %}
                continue;
                {% else %}
                continue;
                {% endif %}
            }

            int {{inner_variable}} = _iter_low - _iter_step;
            while (_n_selected < _iter_size) {
                {{inner_variable}} += _iter_step;
                _n_dealt_with++;
                double _U = _rand(_vectorisation_idx);
                if ((_n_total - _n_dealt_with) * _U >= _iter_size - _n_selected) {
                    continue;
                }
                _n_selected++;
        {% else %}
        {
            // Probabilistic sample
            if (_iter_p == 0) continue;
            int _iter_sign = (_iter_step < 0) ? -1 : 1;
            bool _jump_algo = (_iter_p < 0.25);
            double _log1p = _jump_algo ? log(1.0 - _iter_p) : 1.0;
            double _pconst = 1.0 / _log1p;
            int {{inner_variable}} = _iter_low - _iter_step;
            while (_iter_sign * ({{inner_variable}} + _iter_step) < _iter_sign * _iter_high) {
                {{inner_variable}} += _iter_step;
                if (_jump_algo) {
                    int _jump = (int)(log(_rand(_vectorisation_idx)) * _pconst) * _iter_step;
                    {{inner_variable}} += _jump;
                    if (_iter_sign * {{inner_variable}} >= _iter_sign * _iter_high)
                        break;
                } else {
                    if (_rand(_vectorisation_idx) >= _iter_p) continue;
                }
        {% endif %}
        {% endif %}

            {{vector_code['generator_expr']|autoindent}}
            int _raw{{result_index_array}} = _{{result_index}} + {{result_index_offset}};

            {% if result_index_condition %}
            {% if result_index_used %}
            if (_{{result_index}} < 0 || _{{result_index}} >= _N_result) {
                {% if skip_if_invalid %}
                continue;
                {% else %}
                continue;
                {% endif %}
            }
            {% endif %}
            // create_cond and update both declare _post_idx (or _pre_idx) as
            // const variables. Scope create_cond to prevent redefinition errors.
            bool _create_cond_result = true;
            {
                {{vector_code['create_cond']|autoindent}}
                _create_cond_result = (bool)_cond;
            }
            {% endif %}
            {% if if_expression!='True' and result_index_condition %}
            if (!_create_cond_result) continue;
            {% endif %}
            {% if not result_index_used %}
            if (_{{result_index}} < 0 || _{{result_index}} >= _N_result) {
                {% if skip_if_invalid %}
                continue;
                {% else %}
                continue;
                {% endif %}
            }
            {% endif %}
            {{vector_code['update']|autoindent}}

            for (int _repetition = 0; _repetition < _n; _repetition++) {
                _prebuf[_curbuf] = _pre_idx;
                _postbuf[_curbuf] = _post_idx;
                _curbuf++;
                if (_curbuf == _buffer_size) {
                    _flush_buffer(_prebuf, _dyn_pre, _curbuf);
                    _flush_buffer(_postbuf, _dyn_post, _curbuf);
                    _curbuf = 0;
                }
            }
        {% if iterator_func=='range' %}
        }
        {% else %}
            }
        }
        {% endif %}
    }

    // Final flush of remaining buffered synapses
    if (_curbuf > 0) {
        _flush_buffer(_prebuf, _dyn_pre, _curbuf);
        _flush_buffer(_postbuf, _dyn_post, _curbuf);
    }
{% endblock %}
