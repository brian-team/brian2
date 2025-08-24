{# USES_VARIABLES { N, _spikespace, t } #}
{# ALLOWS_SCALAR_WRITE #}

// Support code
{{support_code_lines}}

// Scalar code
{% if scalar_code %}
{
    const int _vectorisation_idx = -1;
    {{scalar_code|autoindent}}
}
{% endif %}

// Vector code - check threshold condition
{% if vector_code %}
{
    const int _N = {{constant_or_scalar('N', variables['N'])}};

    // Check threshold for each neuron
    for(int _idx=0; _idx<_N; _idx++)
    {
        const int _vectorisation_idx = _idx;
        {{vector_code|autoindent}}
    }
}
{% endif %}
