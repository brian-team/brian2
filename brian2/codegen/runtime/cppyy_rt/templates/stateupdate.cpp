{# USES_VARIABLES { N, dt } #}
{# ALLOWS_SCALAR_WRITE #}

// Scalar code
{% if scalar_code %}
{
    const int _vectorisation_idx = -1;
    {{scalar_code|autoindent}}
}
{% endif %}

// Vector code
{% if vector_code %}
{
    const int _N = {{constant_or_scalar('N', variables['N'])}};
    const double _dt = {{constant_or_scalar('dt', variables['dt'])}};

    for(int _idx=0; _idx<_N; _idx++)
    {
        const int _vectorisation_idx = _idx;
        {{vector_code|autoindent}}
    }
}
{% endif %}
